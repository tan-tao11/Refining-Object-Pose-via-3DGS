import viser
import nerfview
import torch
import time
import argparse
from typing import Dict, List, Optional, Tuple, Union
from torch import Tensor
from gsplat.rendering import rasterization

sh_degree = 3

class viewer():
    def __init__(self, port, device, ckpt_path):
        self.device = device

        # Load splats from checkpoint
        self.splats = self.load_ckpt(ckpt_path)

        self.server = viser.ViserServer(port=port, verbose=False)
        self.viewer = nerfview.Viewer(
            server=self.server,
            render_fn=self._viewer_render_fn,
            mode="training",
        )

    def load_ckpt(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path)
        splats = ckpt["splats"]

        return splats

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, _, _,_ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=sh_degree,  # active all SH degrees
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()
    
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

        colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

        rasterize_mode = "classic"
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
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
            
        # Rendering Gaussian features
        render_features = None
        # if self.cfg.feature_rendering:
        #     gs_features = self.splats["gs_features"]
        #     kwargs.pop('sh_degree')
        #     render_features, _, _info = rasterization(
        #     means=means.detach(),
        #     quats=quats.detach(),
        #     scales=scales.detach(),
        #     opacities=opacities.detach(),
        #     colors=gs_features,
        #     viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
        #     Ks=Ks,  # [C, 3, 3]
        #     width=width,
        #     height=height,
        #     packed=self.cfg.packed,
        #     absgrad=(
        #         self.cfg.strategy.absgrad
        #         if isinstance(self.cfg.strategy, DefaultStrategy)
        #         else False
        #     ),
        #     sparse_grad=self.cfg.sparse_grad,
        #     rasterize_mode=rasterize_mode,
        #     distributed=self.world_size > 1,
        #     camera_model=self.cfg.camera_model,
        #     **kwargs,
        # )
        #     if masks is not None:
        #         render_features[~masks] = 0
        return render_colors, render_alphas, info, render_features
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3DGS online viewer")
    parser.add_argument("--port", type=str, default="6010")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--ckpt", type=str)

    args = parser.parse_args()

    port = args.port
    device = args.device
    ckpt_path = args.ckpt

    print('Starting 3DGS viewer...')
    my_viewer = viewer(port, device, ckpt_path)
    
    while True:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)