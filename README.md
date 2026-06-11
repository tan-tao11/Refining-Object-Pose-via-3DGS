# **SNOP-GS**  
**Self-refining Novel Object Pose via 3D Gaussian Splatting**  

SNOP-GS is a two-stage framework for **novel object pose estimation**, leveraging **3D Gaussian Splatting (3DGS)** to generate dense 2D-3D correspondences and refine poses via a **Transformer architecture**.

## **Key Features**
- **3DGS-based Matching:** Generates high-quality 2D-3D correspondences from 3D Gaussian Splatting models for coarse pose estimation via PnP.
- **Pose Refinement:** Refines coarse poses through a self-refining Transformer module for improved accuracy.
- **Novel Object Generalization:** Handles unseen objects without CAD models or object-specific training.

## **TODO**
1. Upload the processed data.
2. Upload the trained model weights.

## **Quick Start**
### **1. Clone Repository**
```bash
git clone --recursive https://github.com/tan-tao11/SNOP-GS.git
cd SNOP-GS
```

For an existing clone, initialize the bundled third-party dependencies:
```bash
git submodule update --init --recursive
```

### **2. Install Dependencies**
```bash
conda env create -f environment.yaml
conda activate snop_gs
pip install --no-build-isolation ./third_party/fused-ssim
pip install --no-build-isolation -e ./tools/mask/GroundingDINO
pip install -e ./tools/mask/GroundedSegmentAnything/segment_anything
```

### **3. Download Segmentation Weights**
The default preprocessing command uses GroundingDINO with SAM ViT-H. Download
the following weights and place them in `tools/mask/checkpoints/`:

- `groundingdino_swint_ogc.pth`: [GroundingDINO weights](https://github.com/IDEA-Research/GroundingDINO#checkpoint)
- `sam_vit_h_4b8939.pth`: [SAM ViT-H weights](https://github.com/facebookresearch/segment-anything#model-checkpoints)

Optional mask backends require additional weights in the same directory:

- `FastSAM-x.pt` for `--sam_model fast_sam`
- `mobile_sam.pt` for `--sam_model mobile_sam`

### **4. Prepare Datasets**
### Download [OnePose](https://github.com/zju3dv/OnePose_Plus_Plus) dataset and organize it like dataset/OnePose/train
#### Preprocess raw data
```bash
python -m tools.data_preprocess \
    --data_root dataset/OnePose/train_data/ \
    --interval 5 --sam_model sam --data_type train
```

#### Train 3DGS models
```bash
python -m tools.train_gs_models \
    --data dataset_local/OnePose/train_data \
    --output output/gs_models/OnePose/train_data \
    --gpus 2 --threads 4
```

#### Generate 2D-3D correspondences
```bash
python -m tools.gen_real_matches \
    --data dataset_local/OnePose/train_data \
    --ckpt_root output/gs_models/OnePose/train_data \
    --save output/anno_match/OnePose/train_data \
    --data_type train
```

#### Merge annotations
```bash
python -m tools.merge --config config/preprocess/merge_annotation_train_match.yaml
python -m tools.merge --config config/preprocess/merge_annotation_train_align.yaml
```

### **5. Run Training**
**Train Match model:**
```bash
python train.py --training_type match --config config/experiment/train_matching.yaml
```

**Train Refine model:**
```bash
python train.py --training_type align --config config/experiment/train_refining.yaml
```

### **6. Run Testing**
Set `model.ckpt` in each config to the trained checkpoint path before testing.

**Match only:**
```bash
python test.py --testing_type match --config config/experiment/test_matching.yaml
```

**Refine only:**
```bash
python test.py --testing_type align --config config/experiment/test_refining.yaml
```

**Joint (Match -> Refine):**
```bash
python test.py --testing_type joint --config config/experiment/test_joint.yaml
```

## Citation
If you use SNOP-GS in your research, please cite:
```bibtex
@article{tan2026snopgs,
  title     = {SNOP-GS: Self-refining Novel Object Pose via 3D Gaussian Splatting},
  author    = {Tao Tan and Hao Hu and Qiulei Dong},
  journal   = {Pattern Recognition},
  pages     = {113063},
  year      = {2026}
}
```

## Acknowledgements
Built upon the excellent work of:

- [OnePose](https://github.com/zju3dv/OnePose)
- [OnePose++](https://github.com/zju3dv/OnePose_Plus_Plus)
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)

We sincerely thank the authors for their valuable contributions.
