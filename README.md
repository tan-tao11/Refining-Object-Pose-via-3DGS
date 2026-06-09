# **SNOP-GS**  
**Self-refining Novel Object Pose via 3D Gaussian Splatting**  

SNOP-GS is a two-stage framework for **novel object pose estimation**, leveraging **3D Gaussian Splatting (3DGS)** to generate dense 2D-3D correspondences and refine poses via a **Transformer architecture**.

## **Key Features**
- **3DGS-based Matching:** Generates high-quality 2D-3D correspondences from 3D Gaussian Splatting models for coarse pose estimation via PnP.
- **Pose Refinement:** Refines coarse poses through a self-refining Transformer module for improved accuracy.
- **Novel Object Generalization:** Handles unseen objects without CAD models or object-specific training.

## **Quick Start**
### **1. Clone Repository**
```bash
git clone https://github.com/tan-tao11/SNOP-GS.git
cd SNOP-GS
```

### **2. Install Dependencies**
```bash
conda env create -f environment.yaml
conda activate snop_gs
```

### **3. Prepare Datasets**
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

### **4. Run Training**
**Train Match model:**
```bash
python train.py --training_type match --config config/experiment/train_matching.yaml
```

**Train Refine model:**
```bash
python train.py --training_type align --config config/experiment/train_refining.yaml
```

### **5. Run Testing**
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
