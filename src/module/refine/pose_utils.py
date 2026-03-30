import torch
import torch.nn.functional as F


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to 3x3 rotation matrix.

    Based on Zhou et al., "On the Continuity of Rotation Representations
    in Neural Networks", CVPR 2019.

    Args:
        d6: (*, 6) 6D rotation representation (two column vectors of R).

    Returns:
        (*, 3, 3) rotation matrix.
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-2)


def matrix_to_rotation_6d(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to 6D representation.

    Args:
        R: (*, 3, 3) rotation matrix.

    Returns:
        (*, 6) 6D rotation representation.
    """
    return R[..., :2, :].clone().reshape(*R.shape[:-2], 6)


def translation_to_site(t: torch.Tensor) -> torch.Tensor:
    """Convert translation to SITE (Scale-Invariant Translation Estimation).

    SITE(t) = (tx/tz, ty/tz, log(tz)), which decouples the bearing direction
    from the depth and is invariant to image cropping and camera intrinsic
    changes.

    Args:
        t: (*, 3) translation vector.

    Returns:
        (*, 3) SITE representation.
    """
    tz = t[..., 2:3]
    return torch.cat([
        t[..., :2] / (tz + 1e-8),
        torch.log(tz.clamp(min=1e-8)),
    ], dim=-1)


def site_to_translation(site: torch.Tensor) -> torch.Tensor:
    """Convert SITE representation back to translation.

    Args:
        site: (*, 3) SITE representation (tx/tz, ty/tz, log(tz)).

    Returns:
        (*, 3) translation vector.
    """
    tz = torch.exp(site[..., 2:3])
    return torch.cat([site[..., :2] * tz, tz], dim=-1)


def update_pose(
    R_init: torch.Tensor,
    t_init: torch.Tensor,
    delta_R_6d: torch.Tensor,
    delta_t_site: torch.Tensor,
):
    """Update initial pose with predicted residuals.

    Rotation update:  R+ = delta_R @ R
    Translation update in SITE space:  SITE(t_init) + delta_site -> t+

    Args:
        R_init: (N, 3, 3) initial rotation matrix.
        t_init: (N, 3) initial translation vector.
        delta_R_6d: (N, 6) rotation residual in 6D representation.
        delta_t_site: (N, 3) translation residual in SITE space.

    Returns:
        R_refined: (N, 3, 3) refined rotation matrix.
        t_refined: (N, 3) refined translation vector.
    """
    delta_R = rotation_6d_to_matrix(delta_R_6d)
    R_refined = delta_R @ R_init

    site_init = translation_to_site(t_init)
    t_refined = site_to_translation(site_init + delta_t_site)

    return R_refined, t_refined
