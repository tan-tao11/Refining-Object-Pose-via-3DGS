import sys
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from einops import rearrange

from src.utils.position_encoding import PositionEncodingSine
from .backbone import (
    build_backbone,
    _extract_backbone_feats,
    _get_feat_dims,
)
from .pose_utils import rotation_6d_to_matrix, update_pose


def _load_checkpoint(path):
    """Load a checkpoint that may have been saved with pytorch_lightning,
    without requiring it to be installed."""
    import pickle
    import io

    class _PermissiveUnpickler(pickle.Unpickler):
        """Falls back to a dummy class for any module that cannot be
        imported (e.g. pytorch_lightning)."""
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except (ModuleNotFoundError, AttributeError):
                return type(name, (), {})

    with open(path, 'rb') as f:
        data = torch.load(
            io.BytesIO(f.read()),
            map_location='cpu',
            pickle_module=type(
                '_PickleShim', (),
                {'Unpickler': _PermissiveUnpickler,
                 'load': pickle.load,
                 'dumps': pickle.dumps,
                 'loads': pickle.loads,
                 'dump': pickle.dump,
                 'HIGHEST_PROTOCOL': pickle.HIGHEST_PROTOCOL,
                 'UnpicklingError': pickle.UnpicklingError},
            ),
        )
    return data


class PatchEmbed(nn.Module):
    """Tokenize a 2-D feature map into patch tokens via strided convolution."""

    def __init__(self, in_channels, d_model, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, d_model,
            kernel_size=patch_size, stride=patch_size,
        )

    def forward(self, x):
        return self.proj(x)


class PreNormEncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer (compatible with PyTorch >=1.9)."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src):
        # src: (L, N, d_model)  — sequence-first
        x = self.norm1(src)
        x2, _ = self.self_attn(x, x, x)
        src = src + self.dropout1(x2)
        x = self.norm2(src)
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        src = src + self.dropout2(x2)
        return src


class TransformerEncoder(nn.Module):
    """Stack of PreNormEncoderLayers with final LayerNorm."""

    def __init__(self, d_model, nhead, dim_feedforward, dropout, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            PreNormEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src):
        # src: (N, L, d_model) — batch-first input
        x = src.transpose(0, 1)  # -> (L, N, d_model)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x.transpose(0, 1)  # -> (N, L, d_model)


class Refining_Model(nn.Module):
    """Pose refinement module.

    Pipeline
    --------
    1. CNN backbone (reused from matching stage) extracts a *fine* feature map
       from the query image.
    2. A feature map is rendered from the 3-D Gaussians at the initial pose
       (provided externally via ``data['rendered_feature_map']``).
    3. The two feature maps are concatenated along channels and divided into
       patches (tokenised).  Each token receives 2-D sinusoidal positional
       encoding.
    4. The token sequence is fed into **two independent** transformer encoders
       — one for rotation, one for translation.
    5. Global-average-pooled encoder outputs are projected by linear heads to
       predict the rotation residual  delta_R  (6-D representation [Zhou 2019])
       and the translation residual  delta_t  (SITE representation [Li 2019]).
    6. The initial pose is updated:  R+ = delta_R @ R,  t+ via SITE addition.

    Expected ``cfg`` attributes
    ---------------------------
    backbone          – backbone config (same as matching stage)
    render_feat_dim   – channel dim of the rendered GS feature map
    patch_size        – spatial patch size for tokenisation
    d_model           – transformer hidden dimension
    nhead             – number of attention heads
    d_ffn             – feed-forward network intermediate dimension
    dropout           – dropout rate
    num_rot_layers    – number of rotation encoder layers
    num_trans_layers  – number of translation encoder layers
    positional_encoding.enable        – bool
    positional_encoding.pos_emb_shape – [H, W] max shape
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # ---- CNN backbone (reused from matching stage) ----
        self.backbone = build_backbone(cfg.backbone)

        # Fine feature channel dim inferred from backbone config
        feat_dims = _get_feat_dims(cfg.backbone)
        fine_feat_dim = feat_dims[-1]

        # ---- Patch embedding (concatenated features → tokens) ----
        in_channels = fine_feat_dim + cfg.render_feat_dim
        self.patch_embed = PatchEmbed(in_channels, cfg.d_model, cfg.patch_size)

        # ---- 2-D sinusoidal positional encoding ----
        if cfg.positional_encoding.enable:
            self.pos_encoding = PositionEncodingSine(
                cfg.d_model,
                max_shape=cfg.positional_encoding.pos_emb_shape,
            )
        else:
            self.pos_encoding = None

        # ---- Rotation transformer encoder ----
        self.rotation_encoder = TransformerEncoder(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.d_ffn,
            dropout=cfg.dropout,
            num_layers=cfg.num_rot_layers,
        )

        # ---- Translation transformer encoder ----
        self.translation_encoder = TransformerEncoder(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.d_ffn,
            dropout=cfg.dropout,
            num_layers=cfg.num_trans_layers,
        )

        # ---- Projection heads ----
        # 6-D rotation residual  (two ortho-normal column vectors)
        self.rotation_head = nn.Linear(cfg.d_model, 6)
        # SITE translation residual  (tx/tz, ty/tz, log tz)
        self.translation_head = nn.Linear(cfg.d_model, 3)

        self._init_heads()

        # ---- Pretrained backbone loading ----
        self.backbone_pretrained = cfg.backbone.pretrained
        if self.backbone_pretrained is not None:
            logger.info(
                f'Load pretrained backbone from {self.backbone_pretrained}'
            )
            ckpt = _load_checkpoint(self.backbone_pretrained)['state_dict']
            for k in list(ckpt.keys()):
                if 'backbone' in k:
                    newk = k[k.find('backbone') + len('backbone') + 1:]
                    ckpt[newk] = ckpt[k]
                ckpt.pop(k)
            self.backbone.load_state_dict(ckpt)

            if cfg.backbone.pretrained_fix:
                for p in self.backbone.parameters():
                    p.requires_grad = False

    # ------------------------------------------------------------------
    def _init_heads(self):
        """Initialise projection heads so the model predicts *no change*
        at the start of training (identity rotation residual, zero SITE
        residual)."""
        # Rotation: identity rotation → 6D = [1,0,0, 0,1,0]
        nn.init.zeros_(self.rotation_head.weight)
        self.rotation_head.bias.data.copy_(
            torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        )
        # Translation: zero residual in SITE space → no translation change
        nn.init.zeros_(self.translation_head.weight)
        nn.init.zeros_(self.translation_head.bias)

    # ------------------------------------------------------------------
    def forward(self, data):
        """
        Args:
            data (dict):
                query_image            – (N, 1, H, W)
                rendered_feature_map   – (N, C_gs, H_r, W_r) from 3DGS
                initial_R              – (N, 3, 3)
                initial_t              – (N, 3)

        Returns:
            data (dict) updated with:
                R_refined     – (N, 3, 3)
                t_refined     – (N, 3)
                delta_R_6d    – (N, 6)  for loss computation
                delta_t_site  – (N, 3)  for loss computation
        """
        if self.backbone_pretrained and self.cfg.backbone.pretrained_fix:
            self.backbone.eval()

        # 1. Extract fine feature map from query image
        query_feats = self.backbone(data['query_image'])
        query_feats = _extract_backbone_feats(query_feats, self.cfg.backbone)
        query_feat_fine = query_feats[-1]  # (N, C_fine, H_f, W_f)

        # 2. Rendered feature map (produced by 3DGS renderer externally)
        rendered_feat = data['rendered_feature_map']  # (N, C_gs, H_r, W_r)

        # Align spatial resolution to the backbone fine feature map
        if rendered_feat.shape[2:] != query_feat_fine.shape[2:]:
            rendered_feat = F.interpolate(
                rendered_feat,
                size=query_feat_fine.shape[2:],
                mode='bilinear',
                align_corners=True,
            )

        # 3. Concatenate along channel dimension
        concat_feat = torch.cat([query_feat_fine, rendered_feat], dim=1)

        # 4. Patch tokenisation
        tokens = self.patch_embed(concat_feat)  # (N, d, Hp, Wp)

        # 5. Add 2-D positional encoding
        if self.pos_encoding is not None:
            tokens = self.pos_encoding(tokens)

        # Flatten spatial dims → sequence: (N, Hp*Wp, d_model)
        tokens = rearrange(tokens, 'n c h w -> n (h w) c')

        # 6. Rotation encoder → projection head
        rot_out = self.rotation_encoder(tokens)        # (N, L, d)
        rot_global = rot_out.mean(dim=1)                # (N, d)
        delta_R_6d = self.rotation_head(rot_global)     # (N, 6)

        # 7. Translation encoder → projection head
        trans_out = self.translation_encoder(tokens)    # (N, L, d)
        trans_global = trans_out.mean(dim=1)            # (N, d)
        delta_t_site = self.translation_head(trans_global)  # (N, 3)

        # 8. Update pose
        R_refined, t_refined = update_pose(
            data['initial_R'], data['initial_t'],
            delta_R_6d, delta_t_site,
        )

        data.update({
            'R_refined': R_refined,
            't_refined': t_refined,
            'delta_R_6d': delta_R_6d,
            'delta_t_site': delta_t_site,
        })

        return data
