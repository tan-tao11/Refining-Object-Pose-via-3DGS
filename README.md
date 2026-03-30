# SNOP-GS

Object pose estimation via 3D Gaussian Splatting with a two-stage pipeline: **Match** (coarse pose via 2D-3D matching + PnP) and **Refine** (pose refinement via dual-Transformer).

## Installation

```bash
conda env create -f environment.yaml
conda activate snop_gs
```

## Data Preparation

```bash
# 0. Preprocess raw data
python -m tools.data_preprocess \
    --data_root dataset/OnePose/train_data/ \
    --interval 5 --sam_model sam --data_type train

# 1. Train 3DGS models
python -m tools.train_gs_models \
    --data dataset_local/OnePose/train_data \
    --output output/gs_models/OnePose/train_data \
    --gpus 2 --threads 4

# 2. Generate 2D-3D correspondences
python -m tools.gen_real_matches \
    --data dataset_local/OnePose/train_data \
    --ckpt_root output/gs_models/OnePose/train_data \
    --save output/anno_match/OnePose/train_data \
    --data_type train

# 3. Merge annotations
python -m tools.merge --config config/preprocess/merge_annotation_train_match.yaml
python -m tools.merge --config config/preprocess/merge_annotation_train_align.yaml
```

## Training

```bash
# Match model
python train.py --training_type match --config config/experiment/train_matching.yaml

# Refine model
python train.py --training_type align --config config/experiment/train_refining.yaml
```

## Testing

```bash
# Match only
python test.py --testing_type match --config config/experiment/test_matching.yaml

# Refine only
python test.py --testing_type align --config config/experiment/test_refining.yaml

# Joint (Match → Refine)
python test.py --testing_type joint --config config/experiment/test_joint.yaml
```

Set `model.ckpt` in each config to the trained checkpoint path before testing.
