# TDT4265 Mini-Project: Head-and-Neck Tumor Segmentation

This repository contains code and resources for automating 3D segmentation of head-and-neck tumors in pre-radiotherapy T2-weighted MRI scans, developed as part of the TDT4265 Medical Image Computing mini-project.

---

## Project Overview

Manual delineation of gross tumor volumes (GTVₚ) and nodal metastases (GTVₙ) on head-and-neck MRI is time-consuming and prone to inter-observer variability.

This project implements an end-to-end deep learning pipeline using **MONAI** and **PyTorch** to:

1. **Explore & preprocess** 130 pre-RT MRI volumes (512×512×~85) and consensus masks  
2. **Train** a custom **nnU-Net** (3D U-Net with residual blocks & dropout) with targeted patch sampling, augmentations, and One-CycleLR  
3. **Validate** via 5-fold cross-validation, saving “best” and “last” checkpoints per fold  
4. **Infer** with sliding-window and small-component removal  
5. **Ensemble** softmax outputs of the five best models to boost robustness  

**Final performance on held-out test cases**:  
- **Mean Dice ≈ 0.72**  
  - GTVₚ ≈ 0.89  
  - GTVₙ ≈ 0.56  

---

## Usage

### 1. Exploratory Data Analysis (EDA)

- **Notebook**: `notebooks/eda.ipynb`  
- **Script**:

    python scripts/eda.py --data_dir $DATA_DIR

### 2. K-Fold Training

By default, `scripts/kfold_train.py` uses hard-coded values for `EXPERIMENT` and `DATA_DIR` near the top of the file. To customize, you can either:

- Edit the constants `EXPERIMENT` and `DATA_DIR` inside the script, or  
- Update the script to accept CLI arguments.

To kick off all five folds (once configured), run:

    python scripts/kfold_train.py

Results (checkpoints, logs) are saved under:

    results/<experiment>/fold*/

### 3. Inference & Evaluation

#### Single-model inference

1. Edit `scripts/inference.py` at its top to point at the desired fold’s checkpoint and output path.  
2. Run:

    python scripts/inference.py --single

#### Ensemble inference

1. Configure `checkpoint_glob` and `output_dir` inside `scripts/inference.py`.  
2. Run:

    python scripts/inference.py --ensemble

Metrics and prediction files will be written to the configured output directories.

---

## Repository Structure

    ├── notebooks/          # Jupyter notebooks for EDA, training, inference
    │   ├── eda.ipynb
    │   ├── train.ipynb
    │   └── inference.ipynb
    ├── scripts/            # Command-line entrypoints
    │   ├── eda.py
    │   ├── kfold_train.py
    │   └── inference.py
    ├── src/                # Source code packages
    │   ├── dataloader/     # MONAI transforms & DataLoader factories
    │   ├── training/       # Trainer and ensemble utilities
    │   ├── inference/      # Inference & evaluation functions
    │   ├── models/         # Model definitions (CustomNNUNet, etc.)
    │   └── utils/          # EDA, visualization, I/O, metrics
    ├── results/            # Outputs: checkpoints, logs, predictions
    ├── requirements.txt    # Pinned Python dependencies
    └── README.md           # Project overview and instructions
