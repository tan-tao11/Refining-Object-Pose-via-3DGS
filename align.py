import os
import glob
import signal
import sys
import time
import torch
import datetime
import numpy as np
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
import os.path as osp
from loguru import logger
from omegaconf import OmegaConf
from datetime import timedelta
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from src.module.refine.Refining_Model import Refining_Model
from src.module.refine.loss import RefineLoss
from src.data.refine.OnePose import OnePoseRefineDataset


def _load_gs_splats(ckpt_path, device):
    """Load GS splats from checkpoint and move to device."""
    ckpt = torch.load(ckpt_path, map_location=device)
    return {k: v.to(device) for k, v in ckpt['splats'].items()}


@torch.no_grad()
def _render_gs_features_pytorch(splats, viewmat, K, W, H):
    """Pure-PyTorch feature splatting: project Gaussian centres onto the
    image plane and accumulate features weighted by opacity.

    This avoids the gsplat CUDA kernel (which requires matching CUDA
    runtime / driver versions) while producing a reasonable feature map
    for training the refinement head.
    """
    means = splats['means']          # (N, 3)
    opacities = torch.sigmoid(splats['opacities'])  # (N,)
    features = splats['gs_features']  # (N, C)

    # World → camera: p_cam = R @ p_w + t
    R_cam = viewmat[:3, :3]        # (3, 3)
    t_cam = viewmat[:3, 3]         # (3,)
    pts_cam = (R_cam @ means.T).T + t_cam  # (N, 3)

    z = pts_cam[:, 2]
    visible = z > 0.1

    # Project to pixel coordinates
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * pts_cam[:, 0] / z + cx
    v = fy * pts_cam[:, 1] / z + cy

    in_bounds = visible & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[in_bounds].long()
    v = v[in_bounds].long()
    w = opacities[in_bounds].unsqueeze(-1)       # (M, 1)
    f = features[in_bounds]                       # (M, C)

    feat_dim = features.shape[1]
    feat_map = torch.zeros(H * W, feat_dim, device=means.device)
    weight_map = torch.zeros(H * W, 1, device=means.device)

    idx = (v * W + u).unsqueeze(-1)  # (M, 1)

    feat_map.scatter_add_(0, idx.expand_as(f), f * w)
    weight_map.scatter_add_(0, idx, w)

    feat_map = feat_map / (weight_map + 1e-8)
    return feat_map.view(H, W, feat_dim)  # (H, W, C)


_MAX_CACHE = 2  # keep at most N GS models in GPU memory


def render_features_for_batch(data, renderer_cache, device):
    """Render GS feature maps for every sample in the batch."""
    batch_size = data['query_image'].shape[0]
    rendered_list = []

    for i in range(batch_size):
        gs_path = data['gs_model'][i]
        if gs_path not in renderer_cache:
            if len(renderer_cache) >= _MAX_CACHE:
                oldest = next(iter(renderer_cache))
                del renderer_cache[oldest]
                torch.cuda.empty_cache()
            renderer_cache[gs_path] = _load_gs_splats(gs_path, device)
        splats = renderer_cache[gs_path]

        R = data['initial_R'][i]
        t = data['initial_t'][i]
        c2w = torch.eye(4, device=device)
        c2w[:3, :3] = R
        c2w[:3, 3] = t
        viewmat = torch.inverse(c2w)  # world-to-camera

        K = data['K'][i].clone()
        img_wh = data['img_wh']
        W, H = int(img_wh[0][i]) // 2, int(img_wh[1][i]) // 2
        K[0] = K[0] / 2.0
        K[1] = K[1] / 2.0

        feat = _render_gs_features_pytorch(splats, viewmat, K, W, H)
        feat = feat.permute(2, 0, 1)  # (C_gs, H, W)
        rendered_list.append(feat)

    return torch.stack(rendered_list, dim=0)  # (N, C_gs, H, W)


def train_refine_worker(gpu_id: int, config: OmegaConf):
    num_threads = 4
    os.environ["OMP_NUM_THREADS"] = str(num_threads)

    world_size = len(config.gpus.split(","))
    dist.init_process_group(
        backend="nccl" if dist.is_nccl_available() else "gloo",
        timeout=timedelta(seconds=7200000),
        rank=gpu_id,
        world_size=world_size,
    )
    torch.cuda.set_device(gpu_id)
    device = f'cuda:{gpu_id}'

    # ---- Model ----
    model = Refining_Model(config.model).to(device)
    if config.model.get('ckpt') and config.model.ckpt is not None:
        ckpt = torch.load(config.model.ckpt, map_location='cpu')['model_state_dict']
        model.load_state_dict(ckpt)
        logger.info(f'Loaded checkpoint from {config.model.ckpt}')

    # ---- Dataset / Dataloader ----
    dataset = OnePoseRefineDataset(
        anno_file=config.dataset.train_anno_file,
        img_resize=list(config.dataset.img_resize),
        sigma_rot=config.dataset.sigma_rot,
        sigma_trans=config.dataset.sigma_trans,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, shuffle=True, rank=gpu_id,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.data_workers,
        pin_memory=True,
        sampler=sampler,
        persistent_workers=True,
    )

    # ---- Optimizer / Scheduler ----
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)
    max_epochs = config.train.max_epochs
    total_steps = max_epochs * len(data_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-6,
    )

    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[gpu_id], find_unused_parameters=False,
    )
    model.train()

    # ---- Loss ----
    loss_fn = RefineLoss(config.train.loss)

    # ---- GS renderer cache (lazily loaded) ----
    renderer_cache = {}

    # ---- Logging ----
    if gpu_id == 0:
        progress_bar = tqdm(total=total_steps, ncols=100)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{config.exp_name}_{timestamp}"
        writer = SummaryWriter(log_dir=osp.join(config.train.save_dir, exp_name))

    dist.barrier()

    step = 0
    for epoch in range(max_epochs):
        sampler.set_epoch(epoch)
        for _, data in enumerate(data_loader):
            # Move tensors to device
            data = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in data.items()
            }

            # Render GS feature maps for the batch
            data['rendered_feature_map'] = render_features_for_batch(
                data, renderer_cache, device,
            )

            optimizer.zero_grad()

            data = model(data)
            loss_fn(data)

            training_loss = data['loss']
            training_loss.backward()
            optimizer.step()
            scheduler.step()

            if gpu_id == 0:
                step += 1
                progress_bar.update(1)
                progress_bar.set_description(
                    f'Epoch={epoch}/{max_epochs} '
                    f'LR={scheduler.get_last_lr()[0]:.2e} '
                    f'Loss={training_loss.item():.4f}'
                )

                writer.add_scalar('Train/loss', training_loss.item(), step)
                writer.add_scalar('Train/loss_rot', data['loss_rot'].item(), step)
                writer.add_scalar('Train/loss_trans', data['loss_trans'].item(), step)
                writer.add_scalar('Train/lr', scheduler.get_last_lr()[0], step)

                if step % config.train.save_itr == 0:
                    save_model(config, step, f'{step:06d}', model, optimizer, scheduler)

            dist.barrier()


def save_model(cfg, step, mod, model, optimizer, scheduler):
    if not osp.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)
    torch.save({
        'step': step,
        'model_state_dict': model.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, f'{cfg.save_dir}/ckpt_{mod}.pt')


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def signal_handler(sig, frame):
    print("Received CTRL+C, terminating all processes...")
    cleanup()
    sys.exit(0)


def train_refine_model(config):
    signal.signal(signal.SIGINT, signal_handler)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12347'
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus

    try:
        mp.spawn(
            train_refine_worker,
            nprocs=len(config.gpus.split(',')),
            args=(config,),
        )
    except Exception as e:
        print(f"Exception occurred: {e}")
        cleanup()


# ======================================================================
#  Testing
# ======================================================================

def _geodesic_angle(R_pred, R_gt):
    """Geodesic rotation error in degrees between two rotation matrices."""
    R_diff = R_pred @ R_gt.T
    cos_val = (R_diff.trace() - 1.0) / 2.0
    cos_val = cos_val.clamp(-1.0, 1.0)
    return torch.acos(cos_val).item() * 180.0 / np.pi


@torch.no_grad()
def test_refine_model(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
    os.environ["OMP_NUM_THREADS"] = "4"
    device = 'cuda:0'

    # ---- Model ----
    model = Refining_Model(config.model).to(device)
    if config.model.ckpt is not None:
        ckpt = torch.load(config.model.ckpt, map_location='cpu')['model_state_dict']
        model.load_state_dict(ckpt)
        logger.info(f'Loaded checkpoint from {config.model.ckpt}')
    model.eval()

    # ---- Dataset ----
    dataset = OnePoseRefineDataset(
        anno_file=config.dataset.test_anno_file,
        img_resize=list(config.dataset.img_resize),
        sigma_rot=config.dataset.sigma_rot,
        sigma_trans=config.dataset.sigma_trans,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
    )

    renderer_cache = {}
    R_errs_init, t_errs_init = [], []
    R_errs_ref, t_errs_ref = [], []
    timings = []

    for batch_idx, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100):
        data = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in data.items()
        }

        data['rendered_feature_map'] = render_features_for_batch(
            data, renderer_cache, device,
        )

        t0 = time.time()
        data = model(data)
        torch.cuda.synchronize()
        timings.append((time.time() - t0) * 1000)

        R_gt = data['R_gt'][0]
        t_gt = data['t_gt'][0]

        R_errs_init.append(_geodesic_angle(data['initial_R'][0], R_gt))
        t_errs_init.append((data['initial_t'][0] - t_gt).norm().item())

        R_errs_ref.append(_geodesic_angle(data['R_refined'][0], R_gt))
        t_errs_ref.append((data['t_refined'][0] - t_gt).norm().item())

    # ---- Print results ----
    def _stats(vals, name):
        a = np.array(vals)
        print(f"  {name:20s}  mean={a.mean():.4f}  med={np.median(a):.4f}  "
              f"std={a.std():.4f}  min={a.min():.4f}  max={a.max():.4f}")

    print("\n===== Refine Test Results =====")
    print(f"  Samples: {len(R_errs_ref)}")
    print(f"  Avg inference time: {np.mean(timings):.1f} ms")
    print("\n-- Initial pose --")
    _stats(R_errs_init, "R_err (deg)")
    _stats(t_errs_init, "t_err (m)")
    print("\n-- Refined pose --")
    _stats(R_errs_ref, "R_err (deg)")
    _stats(t_errs_ref, "t_err (m)")

    thresholds_R = [1, 3, 5, 10]
    thresholds_t = [0.01, 0.03, 0.05, 0.1]
    R_arr = np.array(R_errs_ref)
    t_arr = np.array(t_errs_ref)

    print("\n-- Accuracy (refined) --")
    for thr in thresholds_R:
        acc = (R_arr < thr).mean() * 100
        print(f"  R < {thr:2d} deg: {acc:6.2f}%")
    for thr in thresholds_t:
        acc = (t_arr < thr).mean() * 100
        print(f"  t < {thr:.2f} m:  {acc:6.2f}%")
    print("=" * 35)


# ======================================================================
#  Joint Testing  (Match → PnP → Refine)
# ======================================================================

def _resolve_gs_model(image_path, gs_base_dir):
    """Derive the GS checkpoint path from the query image path.

    Directory convention:
        image : {data_root}/{split}_data/{obj_name}/{seq}/color/xxx.png
        GS    : {gs_base_dir}/{obj_name}/ckpts/*.pt   (take the last one)
    """
    parts = image_path.replace("\\", "/").split("/")
    try:
        idx = next(i for i, p in enumerate(parts) if p.endswith("_data"))
        obj_name = parts[idx + 1]
    except StopIteration:
        obj_name = parts[-4]

    gs_ckpts = sorted(glob.glob(osp.join(gs_base_dir, obj_name, "ckpts", "*.pt")))
    if not gs_ckpts:
        return None
    return gs_ckpts[-1]


def _render_single_gs_feature(splats, R_init, t_init, K, img_resize, device):
    """Render a GS feature map for one sample given the initial pose."""
    c2w = torch.eye(4, device=device)
    c2w[:3, :3] = R_init
    c2w[:3, 3] = t_init
    viewmat = torch.inverse(c2w)

    W, H = img_resize[0] // 2, img_resize[1] // 2
    K_half = K.clone()
    K_half[0] = K_half[0] / 2.0
    K_half[1] = K_half[1] / 2.0

    feat = _render_gs_features_pytorch(splats, viewmat, K_half, W, H)
    return feat.permute(2, 0, 1)  # (C, H, W)


@torch.no_grad()
def test_joint_model(config):
    """End-to-end evaluation: Match (initial pose via PnP) → Refine."""
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
    os.environ["OMP_NUM_THREADS"] = "4"
    device = "cuda:0"

    match_cfg = config.match
    refine_cfg = config.refine

    # ---- Load Match model ----
    from src.module.match.Matching_Model import Matching_Model
    from src.data.match.OnePose import OnePoseDataset
    from src.utils.metric_utils import ransac_PnP, query_pose_error, aggregate_metrics

    match_model = Matching_Model(match_cfg.model)
    if match_cfg.model.ckpt is not None:
        ckpt = torch.load(match_cfg.model.ckpt, map_location="cpu")["model_state_dict"]
        match_model.load_state_dict(ckpt)
        logger.info(f"[Match] Loaded checkpoint from {match_cfg.model.ckpt}")
    match_model.cuda().eval()

    # ---- Load Refine model ----
    refine_model = Refining_Model(refine_cfg.model).to(device)
    if refine_cfg.model.ckpt is not None:
        ckpt = torch.load(refine_cfg.model.ckpt, map_location="cpu")["model_state_dict"]
        refine_model.load_state_dict(ckpt)
        logger.info(f"[Refine] Loaded checkpoint from {refine_cfg.model.ckpt}")
    refine_model.eval()

    # ---- Dataset (uses Match dataset to get 2D-3D correspondences) ----
    dataset = OnePoseDataset(
        match_cfg,
        device=device,
        anno_file=match_cfg.dataset.test_anno_file,
        shape3d=match_cfg.dataset.shape3d_val,
        load_pose_gt=True,
        load_3d_coarse_feature=match_cfg.dataset.load_3d_coarse,
        image_warp_adapt=False,
        match_type=match_cfg.dataset.match_type,
        split="val",
        percent=0.2,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=1, shuffle=False,
    )

    gs_base_dir = config.gs_base_dir
    img_resize = list(refine_cfg.dataset.img_resize)
    eval_cfg = match_cfg.train.eval_metrics

    renderer_cache = {}
    R_errs_match, t_errs_match = [], []
    R_errs_refine, t_errs_refine = [], []
    timings_match, timings_refine = [], []
    n_skipped = 0

    for batch_idx, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100):
        data = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in data.items()
        }

        # ---- Stage 1: Match → 2D-3D correspondences ----
        t0 = time.time()
        data = match_model(data)
        torch.cuda.synchronize()
        timings_match.append((time.time() - t0) * 1000)

        # ---- PnP to get initial pose ----
        m_bids = data["m_bids"].cpu().numpy()
        mkpts_3d = data["mkpts_3d_db"].cpu().numpy()
        mkpts_query = data["mkpts_query_f"].cpu().numpy()
        img_orig_size = (
            torch.tensor(data["q_hw_i"]).numpy()
            * data["query_image_scale"].cpu().numpy()
        )
        query_K = data["query_intrinsic"].cpu().numpy()
        query_pose_gt = data["query_pose_gt"].cpu().numpy()

        mask = m_bids == 0
        mkpts_query_f = mkpts_query[mask]
        pose_pred_3x4, pose_pred_homo, inliers, state = ransac_PnP(
            query_K[0],
            mkpts_query_f,
            mkpts_3d[mask],
            scale=eval_cfg["point_cloud_rescale"],
            img_hw=img_orig_size[0].tolist(),
            pnp_reprojection_error=eval_cfg["pnp_reprojection_error"],
            use_pycolmap_ransac=eval_cfg["use_pycolmap_ransac"],
        )

        gt_3x4 = query_pose_gt[0][:3]
        if pose_pred_3x4 is None or not state:
            R_errs_match.append(np.inf)
            t_errs_match.append(np.inf)
            R_errs_refine.append(np.inf)
            t_errs_refine.append(np.inf)
            continue

        R_err_m, t_err_m = query_pose_error(
            pose_pred_3x4, gt_3x4, unit=eval_cfg.get("model_unit", "m"),
        )
        R_errs_match.append(R_err_m)
        t_errs_match.append(t_err_m)

        # ---- Resolve GS model for this object ----
        image_path = data["query_image_path"]
        if isinstance(image_path, (list, tuple)):
            image_path = image_path[0]
        gs_path = _resolve_gs_model(image_path, gs_base_dir)
        if gs_path is None:
            logger.warning(f"No GS model found for {image_path}, skip refine")
            R_errs_refine.append(R_err_m)
            t_errs_refine.append(t_err_m)
            n_skipped += 1
            continue

        # ---- Load GS splats (with cache) ----
        if gs_path not in renderer_cache:
            if len(renderer_cache) >= _MAX_CACHE:
                oldest = next(iter(renderer_cache))
                del renderer_cache[oldest]
                torch.cuda.empty_cache()
            renderer_cache[gs_path] = _load_gs_splats(gs_path, device)
        splats = renderer_cache[gs_path]

        # ---- Prepare refine input ----
        R_init = torch.from_numpy(pose_pred_homo[:3, :3].astype(np.float32)).to(device)
        t_init = torch.from_numpy(pose_pred_homo[:3, 3].astype(np.float32)).to(device)

        query_img_raw = data["query_image"]  # (1, 1, H, W)
        if query_img_raw.shape[2:] != tuple(img_resize):
            query_img_raw = torch.nn.functional.interpolate(
                query_img_raw, size=img_resize, mode="bilinear", align_corners=True,
            )

        K_refine = data["query_intrinsic"][0].float().to(device)

        rendered_feat = _render_single_gs_feature(
            splats, R_init, t_init, K_refine, img_resize, device,
        )

        refine_data = {
            "query_image": query_img_raw,                   # (1, 1, H, W)
            "rendered_feature_map": rendered_feat.unsqueeze(0),  # (1, C, Hh, Wh)
            "initial_R": R_init.unsqueeze(0),               # (1, 3, 3)
            "initial_t": t_init.unsqueeze(0),               # (1, 3)
        }

        # ---- Stage 2: Refine ----
        t1 = time.time()
        refine_data = refine_model(refine_data)
        torch.cuda.synchronize()
        timings_refine.append((time.time() - t1) * 1000)

        R_ref = refine_data["R_refined"][0].cpu().numpy()
        t_ref = refine_data["t_refined"][0].cpu().numpy()
        pose_ref_3x4 = np.concatenate([R_ref, t_ref[:, None]], axis=-1)

        R_err_r, t_err_r = query_pose_error(
            pose_ref_3x4, gt_3x4, unit=eval_cfg.get("model_unit", "m"),
        )
        R_errs_refine.append(R_err_r)
        t_errs_refine.append(t_err_r)

    # ---- Report ----
    def _stats(vals, name):
        a = np.array([v for v in vals if v < 1e6])
        if len(a) == 0:
            print(f"  {name:20s}  (no valid samples)")
            return
        print(f"  {name:20s}  mean={a.mean():.4f}  med={np.median(a):.4f}  "
              f"std={a.std():.4f}  min={a.min():.4f}  max={a.max():.4f}")

    print("\n" + "=" * 50)
    print("  Joint Test Results  (Match + Refine)")
    print("=" * 50)
    print(f"  Total samples : {len(R_errs_match)}")
    print(f"  GS skipped    : {n_skipped}")
    if timings_match:
        print(f"  Match  avg time: {np.mean(timings_match):.1f} ms")
    if timings_refine:
        print(f"  Refine avg time: {np.mean(timings_refine):.1f} ms")
    if timings_match and timings_refine:
        total_avg = np.mean(timings_match) + np.mean(timings_refine)
        print(f"  Total  avg time: {total_avg:.1f} ms")

    print("\n-- Match only (PnP) --")
    _stats(R_errs_match, "R_err (deg)")
    _stats(t_errs_match, "t_err (cm)")

    print("\n-- Match + Refine --")
    _stats(R_errs_refine, "R_err (deg)")
    _stats(t_errs_refine, "t_err (cm)")

    # Accuracy at thresholds (using match eval convention: cm & degree)
    pose_thresholds = eval_cfg.get("pose_thresholds", [1, 3, 5])

    print("\n-- Accuracy comparison --")
    print(f"  {'Threshold':>20s}  {'Match':>8s}  {'Refined':>8s}  {'Delta':>8s}")
    R_m = np.array(R_errs_match)
    t_m = np.array(t_errs_match)
    R_r = np.array(R_errs_refine)
    t_r = np.array(t_errs_refine)
    for thr in pose_thresholds:
        acc_m = np.mean((R_m < thr) & (t_m < thr)) * 100
        acc_r = np.mean((R_r < thr) & (t_r < thr)) * 100
        delta = acc_r - acc_m
        sign = "+" if delta >= 0 else ""
        print(f"  {thr}cm@{thr}deg:         {acc_m:7.2f}%  {acc_r:7.2f}%  {sign}{delta:.2f}%")

    print("=" * 50)
