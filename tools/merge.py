import json
import os
import glob
import os.path as osp

from pathlib import Path
from loguru import logger

from src.utils.path_utils import get_test_seq_path, get_gt_pose_path_by_color
import argparse
from omegaconf import OmegaConf


def merge_train_core(
    anno_2d_file,
    avg_anno_3d_file,
    idxs_file,
    img_id,
    ann_id,
    images,
    annotations,
):
    """ Merge training annotations of different objects"""

    with open(anno_2d_file, "r") as f:
        annos_2d = json.load(f)

    for anno_2d in annos_2d:
        img_id += 1
        info = {
            "id": img_id,
            "img_file": anno_2d["img_file"],
        }
        images.append(info)

        ann_id += 1
        anno = {
            "image_id": img_id,
            "id": ann_id,
            "pose_file": anno_2d["pose_file"],
            "anno2d_file": anno_2d["anno_file"],
            "avg_anno3d_file": avg_anno_3d_file,
            "idxs_file": idxs_file,
        }
        annotations.append(anno)

    return img_id, ann_id


def merge_val_core(
    data_dir,
    name,
    avg_anno_3d_file,
    idxs_file,
    anno_id,
    ann_id,
    images,
    annotations,
    last_n_seq_as_test=1,
    downsample=5,
):
    """ Merge validation annotaions of different objects"""
    obj_root = osp.join(data_dir, name)
    test_seq_paths = get_test_seq_path(obj_root, last_n_seq_as_test=last_n_seq_as_test)
    
    for test_seq_path in test_seq_paths:
        anno_dir = osp.join(test_seq_path, "anno_loftr_gs")
        anno_names = os.listdir(anno_dir)
        
        for anno_name in anno_names[::downsample]:
            anno_file = osp.join(anno_dir, anno_name)

            anno_id += 1
            img_file = anno_file.replace("anno_loftr_gs", "color").replace("json", "png")
            info = {"id": anno_id, "img_file": img_file}
            images.append(info)

            ann_id += 1
            pose_file = anno_file.replace("anno_loftr_gs", "poses_ba").replace("json", "txt")
            anno = {
                "image_id": anno_id,
                "id": ann_id,
                "pose_file": pose_file,
                "avg_anno3d_file": avg_anno_3d_file,
                "idxs_file": idxs_file,
            }
            annotations.append(anno)

    return anno_id, ann_id


def merge_(cfg, names, split):
    data_dir = cfg.datamodule.data_dir
    gs_dir = cfg.datamodule.gs_dir

    img_id = 0
    ann_id = 0
    images = []
    annotations = []

    all_data_names = os.listdir(gs_dir)
    id2datafullname = {
        data_name[:4]: data_name for data_name in all_data_names if "-" in data_name
    }
    for name in names:
        if len(name) == 4:
            # ID only!
            if name in id2datafullname:
                name = id2datafullname[name]
            else:
                logger.warning(f"id {name} not exist in sfm directory")
        anno_dir = osp.join(
            gs_dir,
            name,
            "anno",
        )

        logger.info(f"Merging anno dir: {anno_dir}")
        anno_2d_file = osp.join(anno_dir, "anno_2d.json")
        avg_anno_3d_file = osp.join(anno_dir, "anno_3d_average.npz")
        idxs_file = osp.join(anno_dir, "idxs.npy")

        if not osp.isfile(anno_2d_file) or not osp.isfile(avg_anno_3d_file):
            logger.info(f"No annotation in: {anno_dir}")
            continue

        if split == "train":
            img_id, ann_id = merge_train_core(
                anno_2d_file,
                avg_anno_3d_file,
                idxs_file,
                img_id,
                ann_id,
                images,
                annotations,
            )
        elif split == "val":
            img_id, ann_id = merge_val_core(
                data_dir,
                name,
                avg_anno_3d_file,
                idxs_file,
                img_id,
                ann_id,
                images,
                annotations,
                last_n_seq_as_test=cfg.val_use_last_n_seq,
                downsample=1,
            )
        else:
            raise NotImplementedError

    logger.info(f"Total num for {split}: {len(images)}")
    instances = {"images": images, "annotations": annotations}

    out_path = cfg.datamodule.out_path.format(split)
    out_dir = osp.dirname(cfg.datamodule.out_path)
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    with open(out_path, "w") as f:
        json.dump(instances, f)


def merge_anno(cfg):
    # Parse names
    names = cfg.names

    if isinstance(names, str):
        # Parse object directory
        assert isinstance(names, str)
        exception_obj_name_list = cfg.exception_obj_names
        top_k_obj = cfg.top_k_obj
        logger.info(f"Process all objects in directory:{names}")

        object_names = []
        object_names_list = os.listdir(names)[:top_k_obj]
        for object_name in object_names_list:
            if "-" not in object_name:
                continue
            if object_name in exception_obj_name_list:
                continue
            object_names.append(object_name)

        names = object_names

    merge_(cfg, cfg.names, split=cfg.split)

def merge_align_train(seq_dirs, gs_dirs, interval=1):
    annos = []
    id_ = 0

    # Process each object sequences
    for seq_dir, gs_dir in zip(seq_dirs, gs_dirs):
        seqs = os.listdir(seq_dir)
        seqs = [seq for seq in seqs if '-' in seq]
        seqs.sort(key=lambda x: int(x.split("-")[1]))
        for seq in seqs:
            color_folder = os.path.join(seq_dir, seq, 'color')
            color_files = os.listdir(color_folder)
            color_files.sort(key=lambda x: int(x.split('.')[0]))
            for color_file in color_files[::interval]:
                color_file = osp.join(color_folder, color_file)
                anno = {
                    'id': id_,
                    'img_file': color_file,
                    'pose_file': color_file.replace('color', 'poses_ba').replace('png', 'txt'),
                    'intrin_file': color_file.replace('color', 'intrin_ba').replace('png', 'txt'),
                    'gs_model': gs_dir
                }
                annos.append(anno)
                id_ += 1
    
    return annos
            

def merge_align(cfg, names, split='train'):
    data_dir = cfg.datamodule.data_dir
    gs_dir = cfg.datamodule.gs_dir

    img_id = 0
    ann_id = 0
    images = []
    annotations = []

    all_data_names = os.listdir(gs_dir)
    id2datafullname = {
        data_name[:4]: data_name for data_name in all_data_names if "-" in data_name
    }

    seq_dirs = []
    gs_dirs = []
    for name in names:
        if len(name) == 4:
            # ID only!
            if name in id2datafullname:
                name = id2datafullname[name]
            else:
                logger.warning(f"id {name} not exist in sfm directory")
        seq_dir = osp.join(
            data_dir,
            f'{split}_data',
            name,
        )
        gs_model_dir = osp.join(
            gs_dir,
            name,
            'ckpts'
        )
        gs_ckpts = glob.glob(f'{gs_model_dir}/*.pt')
        if not gs_ckpts:
            logger.warning(f"No GS checkpoint found in {gs_model_dir}, skipping {name}")
            continue
        seq_dirs.append(seq_dir)
        gs_dirs.append(gs_ckpts[-1])

    if split == "train":
        annos = merge_align_train(seq_dirs, gs_dirs, interval=5)

    logger.info(f"Total num for {split}: {len(annos)}")

    out_path = cfg.datamodule.out_path.format(split)
    out_dir = osp.dirname(cfg.datamodule.out_path)
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    with open(out_path, "w") as f:
        json.dump(annos, f)


def merge_anno_align(cfg):
    # Parse names
    names = cfg.names

    merge_align(cfg, cfg.names, split=cfg.split)





def main():
    parser = argparse.ArgumentParser(description='Merging data infos.')
    parser.add_argument("--config", type=str)

    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    if 'match' in cfg.task_name:
        merge_anno(cfg)
    if 'align' in cfg.task_name:
        merge_anno_align(cfg)


if __name__ == "__main__":
    main()
