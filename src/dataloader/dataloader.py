"""
dataloader.py

Defines the data ingestion and preprocessing pipelines for MRI tumor segmentation.
- Builds train / validation / test DataLoaders using MONAI’s CacheDataset + DataLoader.
- Encapsulates base transforms (loading, normalization, resizing) and augmentations
  (random crops, flips, elastic warps, intensity perturbations).
- Ensures reproducible splits via fixed RNG seed.
"""

import os
import torch
import monai
from monai.transforms import (
    LoadImaged,  
    EnsureChannelFirstd,
    NormalizeIntensityd,   
    ResizeWithPadOrCropd,
    RandAffined,
    RandBiasFieldd,
    RandFlipd, 
    RandGaussianNoised,
    RandCropByPosNegLabeld,
    Spacingd,
    CropForegroundd,
    ToTensord,
    Compose,
    Rand3DElasticd,
    RandScaleIntensityd,
)
from monai.data import Dataset, DataLoader, pad_list_data_collate, CacheDataset
import numpy as np

def base_transforms(pixdim=(1., 1., 1.), spatial_size=(192, 192, 48)):
    """
    The shared “backbone” for both train and test:
      1) load
      2) channel‑first
      3) resample to 1 × 1 × 1 mm
      4) crop to non‑zero
      5) normalize (zero‑mean / unit‑var on non‑zero voxels)
    """
    return [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"],
            pixdim=pixdim,
            mode=("bilinear", "nearest"),
        ),
        CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ]
    
def train_transforms(spatial_size=(192, 192, 48)):
    """
    Full augmentation pipeline *for training*:
      – start from base
      – randomly sample 8 patches per volume, ~⅓ containing tumor
      – a suite of spatial+intensity augmentations
      – finally pad/crop back to exactly spatial_size
      – to Tensor
    """
    return Compose(
            [
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=spatial_size,
                pos=0.33,   # ~33% of your crops will contain tumor
                neg=0.67,   # ~67% background‑only
                num_samples=3,
            ),
            RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=[0, 1, 2]),
            # ±10° rotations, ±10% scaling
            RandAffined(
                keys=["image", "label"],
                rotate_range=np.deg2rad((10, 10, 10)),
                scale_range=(0.1, 0.1, 0.1),
                prob=0.3,
            ),
            Rand3DElasticd(
                keys=["image", "label"],
                sigma_range=(10.0, 12.0),
                magnitude_range=(20.0, 40.0),
                spatial_size=spatial_size,
                mode=("bilinear", "nearest"),
                prob=0.3,
            ),
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.05),
            RandBiasFieldd(keys=["image"], prob=0.2),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.3),
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=spatial_size),
            ToTensord(keys=["image", "label"]),
        ]
    )


def val_transforms(spatial_size=(192, 192, 48)):
    """
    In validation (and test), *we do not* want any random crops or augmentations—
    we simply want the full volumes (resampled & cropped to foreground),
    then pad/crop to the same spatial size and convert to Tensor.
    """
    return Compose(
        [
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=spatial_size),
            ToTensord(keys=["image", "label"]),
        ]
    )

def test_transforms(spatial_size=(192, 192, 48)):
    """
    In validation (and test), *we do not* want any random crops or augmentations—
    we simply want the full volumes (resampled & cropped to foreground),
    then pad/crop to the same spatial size and convert to Tensor.
    """
    return Compose(
        base_transforms() + 
        [
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=spatial_size),
            ToTensord(keys=["image", "label"]),
        ]
    )


def get_mri_dataloader(
    data_dir: str,
    batch_size: int = 2,
    validation_fraction: float = 0.1,
):
    monai.utils.set_determinism(seed=0)

    # 1) build your list of files
    subset_dir = os.path.join(data_dir, "train")
    data_list = []
    for pid in sorted(os.listdir(subset_dir)):
        preRT = os.path.join(subset_dir, pid, "preRT")
        img = next((f for f in os.listdir(preRT) if f.endswith("T2.nii.gz")), None)
        msk = next((f for f in os.listdir(preRT) if f.endswith("mask.nii.gz")), None)
        if img and msk:
            data_list.append({
                "image": os.path.join(preRT, img),
                "label": os.path.join(preRT, msk),
            })

    # 2) ONE CacheDataset for the base (no random crops or heavy augs)
    base_ds = CacheDataset(
        data=data_list,
        transform=Compose(base_transforms()),   # load → channel-first → resample → crop fg → normalize
        cache_rate=1.0,
        num_workers=4,
    )

    # 3) split indices once
    N = len(base_ds)
    n_val = int(N * validation_fraction)
    g = torch.Generator().manual_seed(0)
    idxs = torch.randperm(N, generator=g)
    train_idx, val_idx = idxs[n_val:], idxs[:n_val]

    # 4) wrap the *subsets* in *lightweight* Datasets that apply your train/val pipelines
    train_ds = Dataset(
        data=[ base_ds[i] for i in train_idx ],
        transform=Compose(train_transforms())
    )
    val_ds   = Dataset(
        data=[ base_ds[i] for i in val_idx ],
        transform=Compose(val_transforms())
    )

    # 5) finally the PyTorch DataLoaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=pad_list_data_collate, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=pad_list_data_collate, num_workers=4, pin_memory=True
    )

    return train_loader, val_loader


def get_test_dataloader(
    data_dir: str,
    batch_size: int = 1,
):
    monai.utils.set_determinism(seed=0)
    
    subset_dir = os.path.join(data_dir, "test")
    data_list = []
    for pid in sorted(os.listdir(subset_dir)):
        preRT = os.path.join(subset_dir, pid, "preRT")
        img = next((f for f in os.listdir(preRT) if f.endswith("T2.nii.gz")), None)
        msk = next((f for f in os.listdir(preRT) if f.endswith("mask.nii.gz")), None)
        if img and msk:
            data_list.append({
                "image": os.path.join(preRT, img),
                "label": os.path.join(preRT, msk),
            })
            
    test_ds = CacheDataset(data=data_list, transform=test_transforms(), cache_rate=1.0, num_workers=4)
    return None, DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )


