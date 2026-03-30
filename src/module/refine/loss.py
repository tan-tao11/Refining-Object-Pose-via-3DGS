import torch
import torch.nn as nn

from .pose_utils import matrix_to_rotation_6d, translation_to_site


class RefineLoss(nn.Module):
    """Loss for pose refinement: rotation + translation in their native
    residual parameterisations."""

    def __init__(self, config):
        super().__init__()
        self.rot_weight = config.rot_weight
        self.trans_weight = config.trans_weight

    def forward(self, data):
        R_gt = data['R_gt']          # (N, 3, 3)
        t_gt = data['t_gt']          # (N, 3)
        R_init = data['initial_R']   # (N, 3, 3)
        t_init = data['initial_t']   # (N, 3)

        delta_R_6d_pred = data['delta_R_6d']     # (N, 6)
        delta_t_site_pred = data['delta_t_site']  # (N, 3)

        # GT rotation residual: delta_R_gt @ R_init = R_gt  =>  delta_R_gt = R_gt @ R_init^T
        delta_R_gt = R_gt @ R_init.transpose(-1, -2)
        delta_R_6d_gt = matrix_to_rotation_6d(delta_R_gt)  # (N, 6)

        # GT translation residual in SITE space
        delta_t_site_gt = translation_to_site(t_gt) - translation_to_site(t_init)

        loss_rot = torch.nn.functional.l1_loss(delta_R_6d_pred, delta_R_6d_gt)
        loss_trans = torch.nn.functional.l1_loss(delta_t_site_pred, delta_t_site_gt)

        loss = self.rot_weight * loss_rot + self.trans_weight * loss_trans

        data.update({
            'loss': loss,
            'loss_rot': loss_rot.detach(),
            'loss_trans': loss_trans.detach(),
        })
