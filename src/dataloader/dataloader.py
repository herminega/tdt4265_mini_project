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
from torch.utils.data import Subset
import numpy as np

def base_transforms(pixdim=(1.,1.,1.), spatial_size=(192,192,48)):
    return [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image","label"], pixdim=pixdim, mode=("bilinear","nearest")),
        CropForegroundd(keys=["image","label"], source_key="image", allow_smaller=True),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        #ResizeWithPadOrCropd(keys=["image","label"], spatial_size=spatial_size),
        ToTensord(keys=["image","label"]),
    ]

def train_transforms(**kwargs):
    # Define the transforms for training
        return Compose([
            # 1. Load the image and label files.
            LoadImaged(keys=["image", "label"]),
            
            # 2. Ensure channel-first ordering.
            EnsureChannelFirstd(keys=["image", "label"]),
            
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest"),
            ),
            
            # 3. Crop the image to the foreground.
            CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
            
            # 4. Normalize the image intensity.
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            
            # 5. Randomly crop the image and label to a specified size.
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(192, 192, 48),
                pos=1,
                neg=2,
                num_samples=3,
            ),
            
            # 6. Resize or pad the crop to a fixed size.
            ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=(192, 192, 48)),
            
            # 7. Augmentations
            RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=[0, 1]),
            RandAffined(
                keys=["image", "label"],
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.1, 0.1, 0.1),
                prob=0.2,
            ),
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.05),
            RandBiasFieldd(keys=["image"], prob=0.15),
            RandScaleIntensityd(keys="image", factors=0.1, prob=0.3),
            # **new**: elastic warp
            Rand3DElasticd(
                keys=["image", "label"],
                sigma_range=(4.0, 6.0),           # controls smoothness
                magnitude_range=(50, 150),       # controls strength
                prob=0.2,
                spatial_size=(192,192,48),        # warp field size
                mode=("bilinear", "nearest"),
            ),  

            # 8. Convert to tensors.
            ToTensord(keys=["image", "label"]),        
        ])

def test_transforms(**kwargs):
    return Compose(base_transforms(**kwargs))

def get_mri_dataloader(data_dir: str, subset="train", batch_size=2, validation_fraction=0.1):
    """
    Loads MRI dataset using MONAI, ensuring correct pairing of images & masks.
    - subset: "train" or "test"
    - batch_size: Number of samples per batch
    """
    monai.utils.set_determinism(seed=0)
    
    if subset == "test":
        transforms = test_transforms()
    else:
        transforms = train_transforms()
    
    # Path to train or test directory
    subset_dir = os.path.join(data_dir, subset)
    patient_folders = sorted(os.listdir(subset_dir))

    # Store paired image-mask file paths
    data_list = []
    
    for patient_id in patient_folders:
        preRT_path = os.path.join(subset_dir, patient_id, "preRT")

        # Find image & mask
        image_file = next((f for f in os.listdir(preRT_path) if f.endswith("T2.nii.gz")), None)
        mask_file = next((f for f in os.listdir(preRT_path) if f.endswith("mask.nii.gz")), None)
        
        if image_file is None:
            print(f"Warning: Skipping {patient_id} due to missing image/mask.")
            continue  # Skip this patient

        full_image_path = os.path.join(preRT_path, image_file)
        full_mask_path = os.path.join(preRT_path, mask_file)
                    
        data_list.append({
            "image": full_image_path,
            "label": full_mask_path
        })
    
    dataset = CacheDataset(data=data_list, transform=transforms, cache_rate=1.0, num_workers=4)
    
    # if test, return immediately
    if subset == "test":
        return None, DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
    
    # else: reproducible split into train/val
    num = len(dataset)
    split = int(num * validation_fraction)
    g = torch.Generator().manual_seed(0)
    idxs = torch.randperm(num, generator=g)
    train_idx, val_idx = idxs[split:], idxs[:split]

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_idx, generator=g),
        drop_last=False,
        collate_fn=pad_list_data_collate,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(val_idx, generator=g),
        drop_last=False,
        collate_fn=pad_list_data_collate,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, val_loader





