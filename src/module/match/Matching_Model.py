import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from einops.einops import rearrange, repeat
from src.utils.load_model import load_backbone
from src.utils.position_encoding import KeypointEncoding_linear, PositionEncodingSine
from src.utils.normalize import normalize_3d_keypoints
from .loftr_module import LocalFeatureTransformer, FinePreprocess
from .coarse_matching import AdaptiveImageFeatureAggregation, CoarseMatching
from .fine_matching import FineMatching
from .backbone import (
    _extract_backbone_feats,
    _get_feat_dims,
)


class Matching_Model(nn.Module):
    def __init__(self, cfg, profiler=None):
        super(Matching_Model, self).__init__()

        self.cfg = cfg
        self.profiler = profiler

        # Backbone network for extracting query image features
        self.backbone = load_backbone(self.cfg.backbone)

        # Position encoder
        # Image
        if self.cfg.positional_encoding.enable:
            self.img_pos_enc = PositionEncodingSine(
                self.cfg.loftr_coarse.d_model,
                max_shape=self.cfg.positional_encoding.pos_emb_shape,
            )
        else:
            self.img_pos_enc = None

        # 3D keypoints
        if self.cfg.keypoints_encoding.enable:
            if cfg.keypoints_encoding.type == 'mlp_linear':
                encoding_func = KeypointEncoding_linear
            else:
                raise NotImplementedError
        
            self.kpt_3d_pos_enc = encoding_func(
                inp_dim=3,
                feature_dim=self.cfg.keypoints_encoding.descriptor_dim,
                layers=self.cfg.keypoints_encoding.keypoints_encoder,
                norm_method=self.cfg.keypoints_encoding.norm_method,
            )
        else:
            self._kpt_3d_pos_enc = None

        # Self- and Cross-attention between query image and reference 3d keypoints
        self.loftr_coarse = LocalFeatureTransformer(self.cfg.loftr_coarse,)
        self.coarse_query_aggregator = AdaptiveImageFeatureAggregation(
            self.cfg.loftr_coarse.d_model
        )

        self.coarse_matching = CoarseMatching(
            self.cfg.coarse_matching,
            profiler=self.cfg.profiler
        )

        self.fine_preprocess = FinePreprocess(
            self.cfg.loftr_fine,
            cf_res=self.cfg.backbone.resolution,
            feat_ids=self.cfg.backbone.resnetfpn.output_layers,
            feat_dims=_get_feat_dims(self.cfg.backbone),
        )

        self.loftr_fine = LocalFeatureTransformer(self.cfg.loftr_fine)
        self.fine_matching = FineMatching(self.cfg.fine_matching)

        self.backbone_pretrained = self.cfg.backbone.pretrained
        if self.cfg.backbone.pretrained is not None:
            logger.info(
                f'Load pretrained backbone from {self.cfg.backbone.pretrained}'
            )
            ckpt = torch.load(self.cfg.backbone.pretrained, 'cpu')['state_dict']
            for k in list(ckpt.keys()):
                if "backbone" in k:
                    newk = k[k.find("backbone") + len("backbone") + 1 :]
                    ckpt[newk] = ckpt[k]
                ckpt.pop(k)
            self.backbone.load_state_dict(ckpt)

            if self.cfg.backbone.pretrained_fix:
                for p in self.backbone.parameters():
                    p.requires_grad = False

    def forward(self, data):
        """
        Update:
            data (dict): {
                keypoints3d: [N, n1, 3]
                descriptors3d_db: [N, dim, n1]
                scores3d_db: [N, n1, 1]

                query_image: (N, 1, H, W)
                query_image_scale: (N, 2)
                query_image_mask(optional): (N, H, W)
            }
        """
        # Set the backbone to evaluation mode if it's pretrained and fixed
        if (
            self.backbone_pretrained 
            and self.cfg.backbone.pretrained_fix
        ):
            self.backbone.eval()
        
        # Update data dict with batch size and query image spatial dimensions
        data.update(
            {
                "bs": data['query_image'].shape[0],
                "q_hw_i": data['query_image'].shape[2:],
            }
        )

        # Extract features from the query image using the backbone
        query_feature_map = self.backbone(data['query_image'])

        # Split the feature map into coarse and fine features
        query_feat_b_c, query_feat_f = _extract_backbone_feats(
            query_feature_map, self.cfg.backbone
        )

        # Update data dict with coarse and fine feature spatial dimensions
        data.update(
            {
                'q_hw_c': query_feat_b_c.shape[2:],
                'q_hw_f': query_feat_f.shape[2:],
            }
        )

        # Apply dense positional encoding to coarse features if enabled
        query_feat_c = rearrange(
            self.img_pos_enc(query_feat_b_c)
            if self.img_pos_enc is not None 
            else query_feat_b_c,
            "n c h w -> n (h w) c",
        )

        # Normalize 3D keypoints
        kpts3d = normalize_3d_keypoints(data['keypoints3d']) #(B, N, 3)

        # Encode 3D keypoints with descriptors
        desc3d_db = (
            self.kpt_3d_pos_enc(
                kpts3d,
                data['descriptors3d_db']
                if 'descriptors3d_coarse_db' not in data
                else data['descriptors3d_coarse_db'],
            )
            if self.kpt_3d_pos_enc is not None
            else data['descriptors3d_db']
            if "descriptors3d_coarse_db" not in data
            else data["descriptors3d_coarse_db"]
        )  #(B, c, N)

        # Flatten query image mask if it exists
        query_mask = None
        if "query_image_mask" in data:
            query_mask = F.interpolate(
                data["query_image_mask"].float().unsqueeze(1),
                size=data["q_hw_c"],
                mode="nearest",
            ).flatten(-2).squeeze(1) > 0

        query_feat_c, query_mask_reduced, reduced_q_hw_c = self.coarse_query_aggregator(
            query_feat_c,
            data["q_hw_c"],
            query_mask=query_mask,
        )

        # Perform self- and coarse-level attention
        desc3d_db, query_feat_c = self.loftr_coarse(
            desc3d_db,
            query_feat_c,
            query_mask=query_mask_reduced,
        )
        query_feat_c = self.coarse_query_aggregator.restore(
            query_feat_c,
            reduced_q_hw_c,
            data["q_hw_c"],
        )
        data.update(
            {
                "coarse_desc3d_db": desc3d_db,
                "coarse_query_feat_c": query_feat_c,
            }
        )

        # Match coarse-level features
        self.coarse_matching(desc3d_db, query_feat_c, data, mask_query=query_mask)

        # If fine matching is not enabled, update data with matched keypoints and return
        if not self.cfg.fine_matching.enable:
            data.update(
                {
                    'mkpts_3d_db': data['mkpts_3d_db'],
                    'mkpts_query_f': data['mkpts_query_c'],
                }
            )
            return
        
        # Fine-level preprocessing to select descriptors and unfold query features
        (
            desc3d_db_selected,
            query_feat_f_unfolded,
        ) = self.fine_preprocess(
            data,
            data['descriptors3d_db'],
            query_feat_f,
        )

        # If there are coarse-level predictions and fine-level is enabled
        if (
            query_feat_f_unfolded.shape[0] != 0
            and self.cfg.loftr_fine.enable
        ):
            # Perform self- and cross-attention for fine-level features
            desc3d_db_selected, query_feat_f_unfolded = self.loftr_fine(
                desc3d_db_selected, query_feat_f_unfolded
            )
        else:
            # Transpose descriptors if no fine-level predictions
            desc3d_db_selected = torch.einsum(
                "bdn->bnd", desc3d_db_selected
            )  # [N, L, C]

        # Perform fine-level matching
        self.fine_matching(desc3d_db_selected, query_feat_f_unfolded, data)

        return data
