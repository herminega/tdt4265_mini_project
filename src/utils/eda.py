
#!/usr/bin/env python
# scripts/eda.py
"""
eda.py

Exploratory data analysis helpers for MRI volumes.
- gather_stats(): walk train/*/preRT, load each image+mask via nibabel, compute per-case stats.
- plot_distributions(): histograms of key columns.
"""

import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def gather_stats(data_dir: str) -> pd.DataFrame:
    """
    Walk through all `train/*/preRT` cases in `data_dir`, load each T2 + mask,
    and compute per‐case statistics.  Returns a DataFrame (one row per patient).
    """
    records = []
    train_root = Path(data_dir) / "train"
    for case_dir in sorted(train_root.iterdir()):
        preRT = case_dir / "preRT"
        if not preRT.is_dir(): 
            continue
        # find your image & mask
        img_file  = next(preRT.glob("*T2.nii.gz"), None)
        mask_file = next(preRT.glob("*mask.nii.gz"), None)
        if img_file is None or mask_file is None:
            continue

        img = nib.load(str(img_file)).get_fdata().astype(np.float32)
        lbl = nib.load(str(mask_file)).get_fdata().astype(np.int32)

        rec = {}
        # intensity stats
        vals = img[np.isfinite(img)]
        rec["case"]    = case_dir.name
        rec["mean_int"]= float(vals.mean())
        rec["std_int"] = float(vals.std())
        rec["min_int"] = float(vals.min())
        rec["max_int"] = float(vals.max())

        # geometry
        x, y, z = img.shape
        rec["dim_x"], rec["dim_y"], rec["num_slices"] = x, y, z
        # how many slices contain any tumor (labels >0)?
        rec["slices_with_tumor"] = int((lbl.sum(axis=(0,1))>0).sum())

        # voxel counts per class
        total = lbl.size
        for c in (0,1,2):
            cnt = int((lbl==c).sum())
            rec[f"class{c}_voxels"] = cnt
        rec["pct_tumor"] = (rec["class1_voxels"] + rec["class2_voxels"]) / total * 100

        records.append(rec)

    df = pd.DataFrame(records)
    return df


def plot_distributions(df: pd.DataFrame) -> None:
    """
    Given the DataFrame from gather_stats, plot key histograms.
    """
    to_plot = [
        "mean_int","std_int","pct_tumor",
        "num_slices","slices_with_tumor",
        "class1_voxels","class2_voxels"
    ]
    n = len(to_plot)
    cols = 3
    rows = (n + cols - 1)//cols
    plt.figure(figsize=(4*cols,3*rows))
    for i, col in enumerate(to_plot, 1):
        ax = plt.subplot(rows, cols, i)
        df[col].hist(bins=20, ax=ax)
        ax.set_title(col)
    plt.tight_layout()
    plt.show()

