import os
import os.path as osp
import signal
import sys
import time
import datetime
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt 
from loguru import logger
from omegaconf import OmegaConf
from datetime import timedelta
from tqdm import tqdm
from io import BytesIO
from PIL import Image
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from src.module.match.Matching_Model import Matching_Model
from src.data.match.OnePose import OnePoseDataset
from src.utils.common_utils import generate_uuid_string
from src.module.match.loss.losses import Loss
from src.module.match.utils.fine_supervision import fine_supervision
from src.utils.plot_utils import draw_reprojection_pair
from src.utils.metric_utils import aggregate_metrics, compute_query_pose_errors

@torch.no_grad()
def validation(model, config, writer, step=0, gpu_id=0):
    model.eval()
    device = f'cuda:{gpu_id}'
    torch.cuda.set_device(gpu_id)

    dataset = OnePoseDataset(
        config,
        device=f'cuda:{gpu_id}',
        anno_file=config.dataset.val_anno_file,
        shape3d=config.dataset.shape3d_val,
        load_pose_gt=True,
        load_3d_coarse_feature=config.dataset.load_3d_coarse,
        image_warp_adapt=False,
        match_type=config.dataset.match_type,
        split="val",
        percent=0.1,
        )
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
    )

    num_val_batches = len(data_loader)
    val_plot_interval = max(num_val_batches // 5, 1)
    figures = []
    metrics_dict = {}
    for batch_idx, data in tqdm(enumerate(data_loader), ncols=100):
        data = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in data.items()}
        # Forward
        data = model(data)

        # Compute pose errors
        compute_query_pose_errors(data, configs=config.train.eval_metrics)
        metrics = {
            "R_errs": data["R_errs"],
            "t_errs": data["t_errs"],
            "inliers": data["inliers"],
        }

        # Visualization
        figure = {"evaluation": []}
        if batch_idx % val_plot_interval == 0:
            figure = draw_reprojection_pair(data, visual_color_type="conf")
            figures.append(fig2numpy(figure['evaluation'][0]))
        
        for k, v in metrics.items():
            if metrics_dict.get(k) is None:
                metrics_dict[k] = []
            metrics_dict[k].append(v[0])   

    figures = torch.from_numpy(np.stack(figures)).permute(0, 3, 1, 2)
    image_grid = make_grid(figures, nrow=len(figures))
    writer.add_image('Validation_vis/pred', image_grid, step)

    val_metrics_4tb = aggregate_metrics(
        metrics_dict, config.train.eval_metrics.pose_thresholds
    )
    for k, v in val_metrics_4tb.items():
        writer.add_scalar(f'Validation_scalar/{k}', v, step)

    for k, v in metrics_dict.items():
        if len(v) != 0:
            try:
                v = [x for x in v if x < 200]
                writer.add_scalar(f'Validation_scalar/{k}', sum(v)/len(v), step)
            except:
                pass
    

def train_match_worker(gpu_id: int, config: OmegaConf):
    """Train the matching model on the given GPU (distributed worker)."""
    # Configure environment and threads
    num_threads = 4
    os.environ.update({
        "OMP_NUM_THREADS": str(num_threads),
        # "OPENBLAS_NUM_THREADS": str(num_threads),
        # "MKL_NUM_THREADS": str(num_threads),
        # "VECLIB_MAXIMUM_THREADS": str(num_threads),
        # "NUMEXPR_NUM_THREADS": str(num_threads),
        # "TORCH_NCCL_BLOCKING_WAIT": "0"
    })
    # torch.set_num_threads(num_threads)

    # Initialize distributed processing
    world_size = len(config.gpus.split(","))
    dist.init_process_group(
        backend="nccl" if dist.is_nccl_available() else "gloo",
        timeout=timedelta(seconds=7200000),
        rank=gpu_id,
        world_size=world_size
    )
    torch.cuda.set_device(gpu_id)

    # Load models and dataset
    model = Matching_Model(config.model).cuda()
    if config.model.ckpt is not None:
        ckpt = torch.load(config.model.ckpt)['model_state_dict']
        model.load_state_dict(ckpt)
        logger.info(f'Load checkpoint from {config.model.ckpt}!')

    dataset =  OnePoseDataset(
        cfg=config,
        device=f'cuda:{gpu_id}',
        anno_file=config.dataset.train_anno_file,
        shape3d=config.dataset.shape3d_train,
        load_pose_gt=True,
        load_3d_coarse_feature=config.dataset.load_3d_coarse,
        image_warp_adapt=config.dataset.train_image_warp_adapt,
        match_type=config.dataset.match_type,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, rank=gpu_id)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.data_workers,
        pin_memory=True,
        sampler=sampler,
        persistent_workers=True
    )

    # Initialize optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, 
    #     step_size=config.train.step_size, 
    #     gamma=config.train.gamma
    #     )
    max_epochs = config.train.max_epochs
    total_steps = max_epochs * len(data_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[gpu_id],
        find_unused_parameters=False
    )
    model.train()

    # Initialize loss function
    loss = Loss(config.train.loss)

    # Training loop setup
    if gpu_id == 0:
        progress_bar = tqdm(total=total_steps, ncols=100)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"train_match_0416_{timestamp}"
        writer = SummaryWriter(log_dir=osp.join(config.train.save_dir, exp_name))
        validation(model.module, config, writer, gpu_id=0)
        model.train()

    dist.barrier()

    step = 0
    for epoch in range(max_epochs):
        # Shuffle the data every epoch
        sampler.set_epoch(epoch)
        for _, data in enumerate(data_loader):
            optimizer.zero_grad()

            # Prepare input data
            data = model(data)

            # Get gt fine supervison source
            fine_supervision(data, config)

            # Compute training losses
            loss(data)

            # backward and optimize
            training_loss = data['loss']
            training_loss.backward()
            optimizer.step()
            scheduler.step()

            if gpu_id == 0:
                step += 1
                progress_bar.update(1)
                progress_bar.set_description(
                    f'Epoch={epoch}/{max_epochs} LR={scheduler.get_last_lr()[0]:.2e} Loss={training_loss.item():.4f}'
                )

                if config.train.enable_plotting and step % config.train.vis_itr == 0:
                    figures = draw_reprojection_pair(data, visual_color_type='conf')
                    vis_figures('Train', writer, figures, step)

                if step % config.train.save_itr == 0:
                    print('save checkpoint')
                    save_model(config, step, f'{step:06d}', model, optimizer, scheduler)

                if step % config.train.val_itr == 0 and step != 0:
                    validation(model.module, config, writer, step, gpu_id=0)
                    model.train()

                writer.add_scalar('Train/loss', training_loss.item(), step)
                writer.add_scalar('Train/learning_rate', scheduler.get_last_lr()[0], step)
            
            dist.barrier()

def fig2numpy(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)

    image_array = np.array(image)

    return image_array

def vis_figures(task, writer, figures, step=0):
    for name, fig in figures.items():
        fig = fig[0]
        image_array = fig2numpy(fig)
        writer.add_image(f'{task}/{name}', image_array, step, dataformats='HWC')

        plt.clf()

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

def train_match_model(config):
    signal.signal(signal.SIGINT, signal_handler)

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = f'12349'
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus

    try:
        mp.spawn(train_match_worker, nprocs=len(config.gpus.split(',')), args=(config,))
    except Exception as e:
        print(f"Exception occurred: {e}")
        cleanup()

# Test entry for the matching model
@torch.no_grad()
def test_match_model(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
    # Configure environment and threads
    num_threads = 4
    os.environ.update({
        "OMP_NUM_THREADS": str(num_threads),
        # "OPENBLAS_NUM_THREADS": str(num_threads),
        # "MKL_NUM_THREADS": str(num_threads),
        # "VECLIB_MAXIMUM_THREADS": str(num_threads),
        # "NUMEXPR_NUM_THREADS": str(num_threads),
        # "TORCH_NCCL_BLOCKING_WAIT": "0"
    })
    
    # Load models and dataset
    model = Matching_Model(config.model)
    if config.model.ckpt is not None:
        ckpt = torch.load(config.model.ckpt)['model_state_dict']
        model.load_state_dict(ckpt)
        logger.info(f'Load checkpoint from {config.model.ckpt}!')
    model.cuda()
    model.eval()
    
    dataset = OnePoseDataset(
        config,
        device=f'cuda:0',
        anno_file=config.dataset.test_anno_file,
        shape3d=config.dataset.shape3d_val,
        load_pose_gt=True,
        load_3d_coarse_feature=config.dataset.load_3d_coarse,
        image_warp_adapt=False,
        match_type=config.dataset.match_type,
        split="val",
        percent=0.2,
        )
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
    )
    
    num_val_batches = len(data_loader)
    metrics_dict = {}
    for batch_idx, data in tqdm(enumerate(data_loader), ncols=100):
        data = {key: value.to('cuda:0') if isinstance(value, torch.Tensor) else value for key, value in data.items()}
        # Forward
        time_start = time.time()
        data = model(data)

        # Compute pose errors
        time_end = time.time()
        run_time = (time_end - time_start) * 1000
        compute_query_pose_errors(data, configs=config.train.eval_metrics)
        metrics = {
            "R_errs": data["R_errs"],
            "t_errs": data["t_errs"],
            "inliers": data["inliers"],
        }
        
        for k, v in metrics.items():
            if metrics_dict.get(k) is None:
                metrics_dict[k] = []
            metrics_dict[k].append(v[0])   

    for k, v in metrics_dict.items():
        if len(v) != 0:
            try:
                v = [x for x in v if x < 200]
                print(f'{k}: ', sum(v)/len(v))
            except:
                pass
            
    val_metrics_4tb = aggregate_metrics(
        metrics_dict, config.train.eval_metrics.pose_thresholds
    )
    for k, v in val_metrics_4tb.items():
        print(f'{k}: ', v)