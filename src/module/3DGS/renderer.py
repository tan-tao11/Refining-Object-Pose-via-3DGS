import viser
import nerfview
import torch
import time
import argparse
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from torch import Tensor
from gsplat.rendering import rasterization


class renderer(nn.Module):
    def __init__(self, ckpt_path):
        super(renderer, self).__init__()
        # Load splats from checkpoint
        self.splats = self.load_ckpt(ckpt_path)

    def load_ckpt(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path)
        splats = ckpt["splats"]

        return splats
    
    def forward(self, c2o, K, img_wh):
        # Render image using given camera
        W, H = img_wh
        
        render_features = self.rasterize_splats(
            camtoworlds=c2o[None],
            Ks=K[None],
            width=W,
            height=H, 
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )

        return render_features

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict, Optional[torch.Tensor]]: # type: ignore
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
            
        # Rendering Gaussian features
        render_features = None
        
        gs_features = self.splats["gs_features"]
        rasterize_mode = "classic"
        if True:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]
            render_colors, render_alphas, info = rasterization(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                colors=colors,
                viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
                Ks=Ks,  # [C, 3, 3]
                width=width,
                height=height,
                packed=False,
                absgrad=(
                    False
                ),
                sparse_grad=False,
                rasterize_mode=rasterize_mode,
                distributed=False,
                camera_model="pinhole",
                sh_degree=3,  # active all SH degrees
                **kwargs,
            )
        render_features, _, _info = rasterization(
            means=means.detach(),
            quats=quats.detach(),
            scales=scales.detach(),
            opacities=opacities.detach(),
            colors=gs_features,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=False,
            absgrad=(
                False
            ),
            sparse_grad=False,
            rasterize_mode=rasterize_mode,
            distributed=False,
            camera_model="pinhole",
            **kwargs,
        )

        return render_features

if __name__ == "__main__":
    ckpt_path = 'output/gs_models/OnePose/train_data/0410-huiyuan-box/ckpts/ckpt_7499_rank0.pt'
    intrin_file = 'dataset_local/OnePose/train_data/0410-huiyuan-box/huiyuan-1/intrin_ba/60.txt'
    pose_file = 'dataset_local/OnePose/train_data/0410-huiyuan-box/huiyuan-1/poses_ba/60.txt'
    my_renderer = renderer(ckpt_path)
    img_wh = [512, 512]
    c2o = torch.eye(4, dtype=torch.float32).cuda()

    with open(intrin_file, 'r') as f:
        intrin_data = np.loadtxt(f)
    K = torch.from_numpy(intrin_data).to(torch.float32).cuda()
    with open(pose_file, 'r') as f:
        pose_data = np.loadtxt(f)
    c2o = torch.from_numpy(pose_data).to(torch.float32).cuda()
    c2o[:3, 3] = c2o[:3, 3]
    c2o = torch.inverse(c2o)

    my_renderer(c2o, K, img_wh)