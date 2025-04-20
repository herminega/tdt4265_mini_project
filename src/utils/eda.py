
#!/usr/bin/env python
# scripts/eda.py

"""
eda.py

Exploratory data analysis helpers for MRI volumes.
- compute_dataset_stats(): sample‑wise slice counts & voxel counts per class.
- compute_tumor_ratio(): fraction of tumor voxels vs. total.
"""


import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataloader.dataloader import get_mri_dataloader  # <-- fixed import

def gather_stats(data_dir):
    """
    Walk through all 'train/*/preRT' cases, load each image+mask via MONAI transforms,
    and compute per‐case statistics.
    Returns a pandas DataFrame with one row per patient.
    """
    # get a loader with no validation split, batch_size=1 so dataset is whole train
    train_loader, _ = get_mri_dataloader(data_dir, subset="train", batch_size=1, validation_fraction=0.0)
    ds = train_loader.dataset

    records = []
    for sample in ds:
        img = sample["image"].numpy().squeeze()
        lbl = sample["label"].numpy().squeeze()

        # intensity statistics (excluding any infinite / NaN)
        vals = img[np.isfinite(img)]
        rec = {
            "mean_int":     float(vals.mean()),
            "std_int":      float(vals.std()),
            "min_int":      float(vals.min()),
            "max_int":      float(vals.max()),
        }

        # spatial / slice stats
        rec["dim_x"], rec["dim_y"], rec["num_slices"] = img.shape
        rec["slices_with_tumor"] = int((lbl.sum(axis=(0,1))>0).sum())

        # class‐voxel counts
        total_vox = lbl.size
        for c in [0,1,2]:
            cnt = int((lbl == c).sum())
            rec[f"class{c}_voxels"] = cnt
        rec["pct_tumor"] = (rec["class1_voxels"] + rec["class2_voxels"]) / total_vox * 100

        records.append(rec)

    df = pd.DataFrame(records)
    return df

def plot_distributions(df):
    """
    Given the DataFrame from gather_stats, plot key histograms
    """
    plt.figure(figsize=(16,10))
    to_plot = [
        "mean_int", "std_int", "pct_tumor",
        "num_slices", "slices_with_tumor",
        "class1_voxels", "class2_voxels"
    ]
    n = len(to_plot)
    cols = 3
    rows = (n + cols - 1)//cols
    for i, col in enumerate(to_plot, 1):
        ax = plt.subplot(rows, cols, i)
        df[col].hist(bins=20, ax=ax)
        ax.set_title(col)
    plt.tight_layout()
    plt.show()


