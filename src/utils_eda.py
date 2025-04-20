import numpy as np
import torch


def compute_dataset_stats(dataset) -> dict:
    """
    Compute basic statistics over a dataset of MRI volumes and segmentation masks.

    Args:
        dataset: Iterable of samples with keys 'image' and 'label', where each is a Tensor or ndarray

    Returns:
        dict: {
            'num_samples': int,
            'slice_counts': List[int],
            'mask_counts': {class_index: voxel_count},
        }
    """
    slice_counts = []
    mask_counts = {0: 0, 1: 0, 2: 0}

    for sample in dataset:
        image = sample['image']
        label = sample['label']

        # Convert tensors to numpy
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if isinstance(label, torch.Tensor):
            label = label.detach().cpu().numpy()

        # Remove channel dim if single-channel
        if image.ndim == 4 and image.shape[0] == 1:
            image = image[0]
        if label.ndim == 4 and label.shape[0] == 1:
            label = label[0]

        # Number of slices
        D = image.shape[-1]
        slice_counts.append(D)

        # Accumulate mask voxel counts per class
        for cls in mask_counts.keys():
            mask_counts[cls] += int((label == cls).sum())

    return {
        'num_samples': len(slice_counts),
        'slice_counts': slice_counts,
        'mask_counts': mask_counts,
    }


def compute_tumor_ratio(mask_counts: dict) -> float:
    """
    Compute the fraction of voxels belonging to tumor classes (1 & 2).
    """
    tumor = mask_counts.get(1, 0) + mask_counts.get(2, 0)
    total = sum(mask_counts.values())
    return tumor / total if total > 0 else 0.0