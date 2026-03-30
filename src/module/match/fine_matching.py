from loguru import logger

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid
from einops.einops import rearrange

class FineMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._type = config['s2d']['type']
        self.fine_3d_proj = None
        self.fine_3d_gate = None
        self.fine_2d_proj = None
        self.fine_2d_gate = None
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:  # pyre-ignore
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:  # pyre-ignore
                nn.init.constant_(m.bias, 0)

    def _build_fusion_layers(self, coarse_3d_dim, fine_dim, coarse_2d_dim, device):
        if self.fine_3d_proj is None:
            self.fine_3d_proj = nn.Linear(coarse_3d_dim, fine_dim, bias=False).to(device)
            self.fine_3d_gate = nn.Sequential(
                nn.Linear(coarse_3d_dim + fine_dim, fine_dim),
                nn.GELU(),
                nn.Linear(fine_dim, fine_dim),
                nn.Sigmoid(),
            ).to(device)
            self.fine_3d_proj.apply(self._init_weights)
            self.fine_3d_gate.apply(self._init_weights)

        if self.fine_2d_proj is None:
            self.fine_2d_proj = nn.Sequential(
                nn.Conv2d(coarse_2d_dim, fine_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(fine_dim),
                nn.GELU(),
            ).to(device)
            self.fine_2d_gate = nn.Sequential(
                nn.Conv2d(fine_dim * 2, fine_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(fine_dim),
                nn.GELU(),
                nn.Conv2d(fine_dim, fine_dim, kernel_size=1),
                nn.Sigmoid(),
            ).to(device)
            self.fine_2d_proj.apply(self._init_weights)
            self.fine_2d_gate.apply(self._init_weights)

    def _adaptive_fine_feature_fusion(self, feat_db_3d, feat_query_unfolded, data):
        if "coarse_desc3d_db" not in data or "coarse_query_feat_c" not in data:
            return feat_db_3d, feat_query_unfolded

        M, WW, fine_dim = feat_query_unfolded.shape
        if M == 0:
            return feat_db_3d, feat_query_unfolded

        coarse_desc3d = data["coarse_desc3d_db"][data["b_ids"], data["i_ids"]]
        coarse_query_feat = data["coarse_query_feat_c"]
        coarse_3d_dim = coarse_desc3d.shape[-1]
        coarse_2d_dim = coarse_query_feat.shape[-1]
        self._build_fusion_layers(coarse_3d_dim, fine_dim, coarse_2d_dim, feat_query_unfolded.device)

        feat_db_3d_center = self.select_left_point(feat_db_3d, data)
        coarse_desc3d_proj = self.fine_3d_proj(coarse_desc3d)
        fuse_3d_gate = self.fine_3d_gate(torch.cat([coarse_desc3d, feat_db_3d_center], dim=-1))
        new_center = fuse_3d_gate * feat_db_3d_center + (1 - fuse_3d_gate) * coarse_desc3d_proj
        center_idx = feat_db_3d.shape[1] // 2
        feat_db_3d = torch.cat([
            feat_db_3d[:, :center_idx, :],
            new_center.unsqueeze(1),
            feat_db_3d[:, center_idx + 1:, :],
        ], dim=1)

        h_c, w_c = data["q_hw_c"]
        h_f, w_f = data["q_hw_f"]
        W = int(math.sqrt(WW))
        stride = h_f // h_c

        coarse_query_map = rearrange(coarse_query_feat, "n (h w) c -> n c h w", h=h_c, w=w_c)
        coarse_query_map = self.fine_2d_proj(coarse_query_map)
        coarse_query_map = F.interpolate(
            coarse_query_map,
            size=(h_f, w_f),
            mode="bilinear",
            align_corners=False,
        )
        coarse_query_unfolded = F.unfold(
            coarse_query_map,
            kernel_size=(W, W),
            stride=stride,
            padding=W // 2,
        )
        coarse_query_unfolded = rearrange(
            coarse_query_unfolded,
            "n (c ww) l -> n l ww c",
            ww=W ** 2,
        )
        coarse_query_unfolded = coarse_query_unfolded[data["b_ids"], data["j_ids"]]

        feat_query_map = rearrange(feat_query_unfolded, "m (h w) c -> m c h w", h=W, w=W)
        coarse_query_map_local = rearrange(coarse_query_unfolded, "m (h w) c -> m c h w", h=W, w=W)
        fuse_2d_gate = self.fine_2d_gate(torch.cat([feat_query_map, coarse_query_map_local], dim=1))
        feat_query_map = (
            fuse_2d_gate * feat_query_map + (1 - fuse_2d_gate) * coarse_query_map_local
        )
        feat_query_unfolded = rearrange(feat_query_map, "m c h w -> m (h w) c")

        return feat_db_3d, feat_query_unfolded

    def forward(self, feat_db_3d, feat_query_unfolded, data):
        """
        Args:
            feat_db_3d (torch.Tensor): [M, N+1, C] (N is nearest feature: N % 2 = 0)
            feat_query_unfold (torch.Tensor): [M, WW, C]
            data (dict)
        Update:
            data (dict):{
                'expec_f' (torch.Tensor): [M, 3],
                'mkpts_3d_db' (torch.Tensor): [M, 3],
                'mkpts_query_f' (torch.Tensor): [M, 2]}
        """
        M, WW, C = feat_query_unfolded.shape
        W = int(math.sqrt(WW))
        scale = data['q_hw_i'][0] / data['q_hw_f'][0]
        self.M, self.W, self.WW, self.C, self.scale = M, W, WW, C, scale

        # corner case: if no coarse matches found
        if M == 0:
            assert self.training == False, "M is always >0, when training, see coarse_matching.py"
            logger.warning('No matches found in coarse-level.')
            _out_dim = 3 if self._type == 'heatmap' else 2
            data.update({
                'expec_f': torch.empty(0, _out_dim, device=feat_db_3d.device),
                'mkpts_3d_db': data['mkpts_3d_db'],
                'mkpts_query_f': data['mkpts_query_c'],
            })
            return

        feat_db_3d, feat_query_unfolded = self._adaptive_fine_feature_fusion(
            feat_db_3d,
            feat_query_unfolded,
            data,
        )

        feat_db_3d_selected = self.select_left_point(feat_db_3d, data) # [M, C]
        
        coords_normed = self.predict_s2d(feat_db_3d_selected, feat_query_unfolded, data)
        # compute absolute kpt coords
        self.build_mkpts(coords_normed, data)
        
    def select_left_point(self, feat_f0, data):
        L = feat_f0.shape[1]
        assert L % 2 == 1
        
        feat_f0_picked = feat_f0[:, L//2, :]
        return feat_f0_picked
        
    def predict_s2d(self, feat_f0_picked, feat_f1, data):
        # compute normalized coords ([-1, 1]) of right patches
        if self._type == 'heatmap':
            coords_normed = self._s2d_heatmap(feat_f0_picked, feat_f1, data)
        else:
            raise NotImplementedError()
        return coords_normed
        
    def _s2d_heatmap(self, feat_f0_picked, feat_f1, data):
        W, WW, C = self.W, self.WW, self.C
        
        sim_matrix = torch.einsum('mc,mrc->mr', feat_f0_picked, feat_f1)
        softmax_temp = 1. / C**.5
        heatmap = torch.softmax(softmax_temp * sim_matrix, dim=1).view(-1, W, W)

        # compute coordinates from heatmap
        coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]  # [M, 2]
        grid_normalized = create_meshgrid(W, W, True, heatmap.device).reshape(1, -1, 2)  # [1, WW, 2]

        # compute std over <x, y>
        var = torch.sum(grid_normalized**2 * heatmap.view(-1, WW, 1), dim=1) - coords_normalized**2  # [M, 2]
        std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1)  # [M]  clamp needed for numerical stability

        data.update({'expec_f': torch.cat([coords_normalized, std.unsqueeze(1)], -1)})
        return coords_normalized
        
    @torch.no_grad() 
    def build_mkpts(self, coords_normed, data):
        W, WW, C, scale = self.W, self.WW, self.C, self.scale
        
        # mkpts0_f
        mkpts3d_db = data['mkpts_3d_db']

        # mkpts1_f
        query_scale = scale * data['query_image_scale'][data['b_ids']][:, [1, 0]] if 'query_image_scale' in data else scale
        mkpts_query_f = data['mkpts_query_c'] + (coords_normed * (W // 2) * query_scale)[:len(data['mkpts_query_c'])]

        data.update({
            "mkpts_3d_db": mkpts3d_db,
            "mkpts_query_f": mkpts_query_f
        })
