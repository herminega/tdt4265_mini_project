import os
import torch
import monai
from monai.transforms import (
    LoadImaged,  
    EnsureChannelFirstd,
    ScaleIntensityd,
    NormalizeIntensityd,   
    ResizeWithPadOrCropd,
    AsDiscreted,
    Lambdad,
    RandFlipd, 
    RandGaussianNoised,
    Rand3DElasticd,
    HistogramNormalized,
    AdjustContrastd,
    EnsureTyped,
    RandCropByLabelClassesd,
    RandCropByPosNegLabeld,
    CropForegroundd,
    ToTensord,
    Compose,
)
from monai.data import Dataset, DataLoader, pad_list_data_collate, CacheDataset
import nibabel as nib
from torch.utils.data import Subset
import numpy as np

monai.utils.set_determinism(seed=42)

def get_subset_dataloader(dataset, indices, fraction=0.5, batch_size=2):
    """
    Returns a DataLoader for a subset of the dataset.
    This function ensures we only take a subset from a predefined set of indices (train or validation).
    """
    subset_size = int(len(indices) * fraction)
    subset_indices = np.random.choice(indices, subset_size, replace=False)  # Sample from indices
    subset = Subset(dataset, subset_indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=False)


def get_mri_dataloader(data_dir: str, subset="train", batch_size=2, validation_fraction=0.1):
    """
    Loads MRI dataset using MONAI, ensuring correct pairing of images & masks.
    - subset: "train" or "test"
    - batch_size: Number of samples per batch
    """

    
    spatial_crop_size = (64, 96, 96)  # crop region
    target_size = (64, 96, 96)        # final uniform size
    
    transforms = Compose([
        LoadImaged(
            keys=["image", "label"],
            meta_keys=["image_meta_dict", "label_meta_dict"],
            image_only=False  # <--- This is key!
        ),
        
        EnsureChannelFirstd(keys=["image", "label"]),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),

        # Ensure that sampled patches contain tumor
        RandCropByLabelClassesd(
            keys=["image", "label"],
            label_key="label",
            spatial_size=spatial_crop_size,
            num_classes=3,
            num_samples=6,  # per image
            ratios=[0.4, 0.3, 0.3],
            allow_smaller = True
        ),
        ResizeWithPadOrCropd(keys=["image", "label"], spatial_size=target_size),
        AsDiscreted(keys=["label"], to_onehot=3),
        ToTensord(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"], track_meta=True),
    ])
        
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

    #Use dictionary-based Dataset
    #dataset = Dataset(data=data_list, transform=transforms)
    
    dataset = CacheDataset(data=data_list, transform=transforms, cache_rate=1.0, num_workers=4)

    
    # Split into training & validation sets
    num_samples = len(dataset)
    indices = torch.randperm(num_samples)
    split_idx = int(num_samples * validation_fraction)

    train_indices, val_indices = indices[split_idx:], indices[:split_idx]

    #train_loader = DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(train_indices), drop_last=False, collate_fn=pad_list_data_collate)
    #val_loader = DataLoader(dataset, batch_size=batch_size, sampler=torch.utils.data.SubsetRandomSampler(val_indices), drop_last=False, collate_fn=pad_list_data_collate)

    train_loader = get_subset_dataloader(dataset, train_indices, fraction=0.5, batch_size=2)
    val_loader = get_subset_dataloader(dataset, val_indices, fraction=0.5, batch_size=2)
    
    return train_loader, val_loader



