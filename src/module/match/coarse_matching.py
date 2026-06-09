import torch
import torch.nn as nn
from torch.nn import functional as F
from einops.einops import rearrange


def mask_border(m, b: int, v):
    """Zero out a margin of width b on the coarse feature map."""
    if b <= 0:
        return
    m[:, :, :b] = v
    m[:, :, :, :b] = v
    m[:, :, -b:] = v
    m[:, :, :, -b:] = v


def mask_border_with_padding(m, bd, v, p_m0, p_m1):
    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd] = v
    m[:, :, :, :, :bd] = v

    h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
    h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()
    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        m[b_idx, h0 - bd :] = v
        m[b_idx, :, w0 - bd :] = v
        m[b_idx, :, :, h1 - bd :] = v
        m[b_idx, :, :, :, w1 - bd :] = v


def calc_max_candidates(p_m0, p_m1):
    """Calculate the max candidates of all pairs within a batch"""
    h0s, w0s = p_m0.sum(1).max(-1)[0], p_m0.sum(-1).max(-1)[0]
    h1s, w1s = p_m1.sum(1).max(-1)[0], p_m1.sum(-1).max(-1)[0]
    max_cand = torch.sum(torch.min(torch.stack([h0s * w0s, h1s * w1s], -1), -1)[0])
    return max_cand


def build_feat_normalizer(method, **kwargs):
    if method == "sqrt_feat_dim":
        return lambda feat: feat / feat.shape[-1] ** 0.5
    elif method == "none" or method is None:
        return lambda feat: feat
    elif method == "temparature":
        return lambda feat: feat / kwargs["temparature"]
    else:
        raise ValueError


class CoarseMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feat_normalizer = build_feat_normalizer(config["feat_norm_method"])

        self.type = config["type"]
        if self.type == "dual-softmax":
            self.temperature = config['dual_softmax']['temperature']
        elif self.type == "sigmoid":
            self.temperature = config['sigmoid']['temperature']
        else:
            raise NotImplementedError()

        self.thr = config["thr"]
        self.border_rm = config["border_rm"]
        self.train_coarse_percent = config["train"]["train_coarse_percent"]
        self.train_pad_num_gt_min = config["train"]["train_pad_num_gt_min"]

    def forward(self, feat_db_3d, feat_query, data, mask_query=None):
        """
        Args:
            feat_db_3d (torch.Tensor): [N, L, C]
            feat_query (torch.Tensor): [N, S, C]
            data (dict)
            mask_query (torch.Tensor): [N, S] (optional)
        Update:
            data (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'mkpts_3d_db' (torch.Tensor): [M, 3],
                'mkpts_query_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
            NOTE: M' != M during training.
        """
        N, L = feat_db_3d.size(0), feat_db_3d.size(1)
        S = feat_query.size(1)

        # normalize
        feat_db_3d, feat_query = map(self.feat_normalizer, [feat_db_3d, feat_query])

        if self.type == "dual-softmax":
            sim_matrix = (
                torch.einsum("nlc,nsc->nls", feat_db_3d, feat_query) / (self.temperature + 1e-4)
            )
            if mask_query is not None:
                fake_mask3D = torch.ones((N, L), dtype=torch.bool, device=mask_query.device)
                valid_sim_mask = fake_mask3D[..., None] * mask_query[:, None]
                _inf = torch.zeros_like(sim_matrix)
                _inf[~valid_sim_mask.bool()] = -1e9
                del valid_sim_mask
                sim_matrix += _inf
            conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)
        elif self.type == "sigmoid":
            sim_matrix = (
                torch.einsum("nlc,nsc->nls", feat_db_3d, feat_query) / (self.temperature + 1e-4)
            )
            conf_matrix = F.softmax(sim_matrix, 2)
        else:
            raise NotImplementedError

        data.update({"conf_matrix": conf_matrix})

        if self.type == "sigmoid":
            conf_matrix_softmax = conf_matrix*F.softmax(sim_matrix, 1)
            data.update(**self.get_coarse_match(conf_matrix_softmax, data))
        else:
            data.update(**self.get_coarse_match(conf_matrix, data))

    @torch.no_grad()
    def get_coarse_match(self, conf_matrix, data):
        """
        Args:
            conf_matrix (torch.Tensor): [N, L, S]
            data (dict): with keys ['hw1_i', 'hw1_c']
        Returns:
            coarse_matches (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'm_bids' (torch.Tensor): [M],
                'mkpts_3d_db' (torch.Tensor): [M, 3],
                'mkpts_query_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
        """
        axes_lengths = {"h1c": data["q_hw_c"][0], "w1c": data["q_hw_c"][1]}

        device = conf_matrix.device
        # confidence thresholding
        mask = conf_matrix > self.thr
        mask = rearrange(
            mask, "b n_point_cloud (h1c w1c) -> b n_point_cloud h1c w1c", **axes_lengths
        )
        if "mask0" not in data:
            mask_border(mask, self.border_rm, False)
        else:
            raise NotImplementedError
        mask = rearrange(
            mask, "b n_point_cloud h1c w1c -> b n_point_cloud (h1c w1c)", **axes_lengths
        )

        # mutual nearest
        if self.type == "dual-softmax":
            mask = (
                mask
                * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0])
                * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])
            )
        elif self.type == "sigmoid":
            mask = mask * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0])
        else:
            raise NotImplementedError

        # At most one True per (b, i) row after mutual check; take argmax over query tokens.
        mask_v, all_j_ids = mask.max(dim=2)
        b_ids, i_ids = torch.where(mask_v)
        j_ids = all_j_ids[b_ids, i_ids]
        mconf = conf_matrix[b_ids, i_ids, j_ids]

        # Training: subsample predicted matches and pad with GT for fine-level supervision
        if self.training and self.config['train']['train_padding']:
            if "mask0" not in data:
                num_candidates_max = mask.size(0) * min(mask.size(1), mask.size(2))
            else:
                raise NotImplementedError("mask0 padding path is not implemented")
            max_num_matches_train = int(num_candidates_max * self.train_coarse_percent)
            num_matches_pred = len(b_ids)
            assert (
                self.train_pad_num_gt_min < max_num_matches_train
            ), "min-num-gt-pad should be less than num-train-matches"

            # pred_indices is to select from prediction
            if num_matches_pred <= max_num_matches_train - self.train_pad_num_gt_min:
                pred_indices = torch.arange(num_matches_pred, device=device)
            else:
                pred_indices = torch.randint(
                    num_matches_pred,
                    (max_num_matches_train - self.train_pad_num_gt_min,),
                    device=device,
                )

            spv_b_ids, spv_i_ids, spv_j_ids = torch.where(data['conf_matrix_gt'])
            assert len(spv_b_ids) != 0
            gt_pad_indices = torch.randint(
                len(spv_b_ids),
                (max(max_num_matches_train - num_matches_pred, self.train_pad_num_gt_min),),
                device=device,
            )
            mconf_gt = torch.zeros(len(spv_b_ids), device=device)

            b_ids, i_ids, j_ids, mconf = map(
                lambda x, y: torch.cat([x[pred_indices], y[gt_pad_indices]], dim=0),
                *zip(
                    [b_ids, spv_b_ids],
                    [i_ids, spv_i_ids],
                    [j_ids, spv_j_ids],
                    [mconf, mconf_gt],
                ),
            )

        coarse_matches = {"b_ids": b_ids, "i_ids": i_ids, "j_ids": j_ids}

        # Map coarse grid indices to full-resolution query image coordinates
        scale = data["q_hw_i"][0] / data["q_hw_c"][0]
        scale_total = scale * data["query_image_scale"][b_ids][:, [1, 0]] if "query_image_scale" in data else scale
        mkpts_query = (
            torch.stack([j_ids % data["q_hw_c"][1], j_ids // data["q_hw_c"][1]], dim=1)
            * scale_total
        )
        mkpts_3d_db = data["keypoints3d"][b_ids, i_ids]

        # Subset with non-zero confidence (excludes GT padding) for visualization
        coarse_matches.update(
            {
                "gt_mask": mconf == 0,
                "m_bids": b_ids[mconf != 0],
                "mkpts_3d_db": mkpts_3d_db[mconf != 0],
                "mkpts_query_c": mkpts_query[mconf != 0],
                "mconf": mconf[mconf != 0],
            }
        )

        return coarse_matches
