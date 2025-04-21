
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
    Walk the train/*/preRT folders, load each T2 and mask pair
    and compute per‐case stats (intensity + geometry + voxel counts).
    Returns a DataFrame with one row per patient.
    """
    data_dir = Path(data_dir) / "train"
    records = []
    for pid_folder in sorted(data_dir.iterdir()):
        preRT = pid_folder / "preRT"
        if not preRT.is_dir():
            continue
        t2 = next(preRT.glob("*T2.nii.gz"), None)
        m  = next(preRT.glob("*mask.nii.gz"), None)
        if not (t2 and m):
            continue

        img = nib.load(str(t2)).get_fdata().astype(float)
        lbl = nib.load(str(m )).get_fdata().astype(int)

        rec = {
            "patient": pid_folder.name,
            "dim_x":   img.shape[0],
            "dim_y":   img.shape[1],
            "num_slices": img.shape[2],
            "mean_int":   float(img[np.isfinite(img)].mean()),
            "std_int":    float(img[np.isfinite(img)].std()),
            "min_int":    float(np.nanmin(img)),
            "max_int":    float(np.nanmax(img)),
            "slices_with_tumor": int((lbl.sum(axis=(0,1))>0).sum()),
            "class0_vox": int((lbl==0).sum()),
            "class1_vox": int((lbl==1).sum()),
            "class2_vox": int((lbl==2).sum()),
        }
        rec["pct_tumor"] = 100*(rec["class1_vox"] + rec["class2_vox"]) / lbl.size
        records.append(rec)

    return pd.DataFrame(records)


def plot_distributions(df: pd.DataFrame):
    to_plot = [
      "mean_int","std_int","pct_tumor",
      "num_slices","slices_with_tumor"
    ]
    n = len(to_plot)
    cols = 3
    rows = (n+cols-1)//cols
    plt.figure(figsize=(4*cols,3*rows))
    for i,col in enumerate(to_plot,1):
        ax = plt.subplot(rows, cols, i)
        df[col].hist(bins=20, ax=ax)
        ax.set_title(col)
    plt.tight_layout()
    plt.show()

