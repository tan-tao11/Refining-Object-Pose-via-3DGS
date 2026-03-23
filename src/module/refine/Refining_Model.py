import torch
import torch.nn as nn
from loguru import logger
from src.utils.position_encoding import KeypointEncoding_linear, PositionEncodingSine
from .loftr_module import LocalFeatureTransformer
from .backbone import (
    build_backbone,
    _extract_backbone_feats,
    _get_feat_dims,
)


class Refining_Model(nn.Module): 
    def __init__(self, cfg):
        super(Refining_Model, self).__init__()

        self.cfg = cfg
        
        # Used to extract 2D query image feature
        self.backbone = build_backbone(self.cfg.backbone)

        # Positional encoding for query image and rendering
        if self.cfg.posetional_encoding.enable:
            self.img_pos_enc = PositionEncodingSine(
                self.cfg.transformer.d_model,
                max_shape=self.cfg.positional_encoding.pose_emb_shape,
            )
        else:
            self.img_pose_enc = None

        # Self- and Cross-attention between query image and rendering
        self.transformer = LocalFeatureTransformer(self.cfg.transformer)

        # Load pretrained weight
        self.backbone_pretrained = self.cfg.backbone.pretrained
        if self.backbone_pretrained is not None:
            logger.info(
                f'Load pretrained backbone from {self.backbone_pretrained}'
            )
            ckpt = torch.load(self.backbone_pretrained, 'cpu')['state_dict']
            for k in list(ckpt.keys()):
                if 'backbone' in k:
                    newk = k[k.find('backbone') + len('backbone') + 1 :]
                    ckpt[newk] = ckpt[k]
                ckpt.pop(k)
            self.backbone.load_state_dict(ckpt)

            # Fix the pretrained weight during training
            if self.cfg.backbone.pretrained_fix:
                for p in self.backbone.parameters():
                    p.requires_grad = False
    
    def forward(self, data):
        pass
