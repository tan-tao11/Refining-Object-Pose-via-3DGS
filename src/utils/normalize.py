import torch

def normalize_2d_keypoints(kpts, image_shape):
    """ Normalize 2d keypoints locations based on image shape
    kpts: [b, n, 2]
    image_shape: [b, 2]
    """
    height, width = image_shape[0]
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


def normalize_3d_keypoints(kpts):
    """ Normalize 3d keypoints locations based on the tight box
    kpts: [b, n, 3]
    """
    # Compute per-sample tight box extents and center so that every sample in
    # the batch is normalized with its own scale (samples may be different objects).
    extents = kpts.max(dim=1).values - kpts.min(dim=1).values  # [B, 3]
    center = torch.mean(kpts, dim=1)  # [B, 3]
    scaling = extents.max(dim=1, keepdim=True).values * 0.6  # [B, 1]
    kpts_rescaled = (kpts - center[:, None, :]) / scaling[:, None, :]
    return kpts_rescaled