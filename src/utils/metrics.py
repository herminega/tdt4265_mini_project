"""
metrics.py

Core metric‑related helpers for training and evaluation.
- set_global_seed(): fix all RNGs for deterministic behavior.
- should_early_stop(): stop training when validation Dice plateaus.
- remove_small_cc(): drop tiny connected components from segmentation maps.
- is_best_model(): compare current vs. historical validation loss.
- freeze_encoder(): optionally freeze encoder layers for fine‑tuning.
"""

import torch
import os, random
import numpy as np
from scipy import ndimage
import numpy as np
import nibabel as nib


def should_early_stop(self, delta: float = 1e-4):
    """
    Stop if the mean‐dice hasn’t improved by > delta over the last `early_stop_count` epochs.
    """
    val_dice = self.validation_history["dice"]
    # not enough epochs yet
    if len(val_dice) < self.early_stop_count:
        return False

    # take the last `early_stop_count` dice values
    recent = list(val_dice.values())[-self.early_stop_count:]

    # if the best dice in that window is no better than the first epoch’s dice + delta, stop
    best_recent = max(recent)
    return best_recent <= (recent[0] + delta)

def log_metrics(epoch, train_metrics, val_metrics):
    print(f"\nEpoch {epoch} Summary:")
    print(f"\nTraining:")
    print(f"  Loss: {train_metrics['loss']:.4f}")
    print(f"  Mean Dice: {train_metrics['mean_dice']:.4f}")
    print(f"  Class 1 Dice: {train_metrics['dice_class1']:.4f}")
    print(f"  Class 2 Dice: {train_metrics['dice_class2']:.4f}")
    print(f"\n  Class 1 : Precision {train_metrics['precision_class1']:.4f} / Recall {train_metrics['recall_class1']:.4f}")
    print(f"  Class 2 : Precision {train_metrics['precision_class2']:.4f} / Recall {train_metrics['recall_class2']:.4f}")
    
    print(f"\nValidation:")
    print(f"  Loss: {val_metrics['loss']:.4f} ")
    print(f"  Mean Dice: {val_metrics['mean_dice']:.4f}")
    print(f"  Class 1 Dice: {val_metrics['dice_class1']:.4f}")
    print(f"  Class 2 Dice: {val_metrics['dice_class2']:.4f}")
    print(f"\n  Class 1 : Precision {val_metrics['precision_class1']:.4f} / Recall {val_metrics['recall_class1']:.4f}")
    print(f"  Class 2 : Precision {val_metrics['precision_class2']:.4f} / Recall {val_metrics['recall_class2']:.4f}")


def freeze_encoder(model, freeze=True):
    for name, param in model.named_parameters():
        if "down" in name:  # or 'encoder' for other nets
            param.requires_grad = not freeze

def is_best_model(global_step, validation_history):
    """
    Determines if the model at the given global step is the best so far,
    based on the lowest validation loss.
    """
    current_loss = validation_history["loss"][global_step]
    return current_loss == min(validation_history["loss"].values())

def remove_small_cc(seg: np.ndarray, min_voxels: int = 300) -> np.ndarray:
    """
    Remove connected components smaller than `min_voxels` from a 3D label array.
    """
    cleaned = np.zeros_like(seg)
    for cls in np.unique(seg):
        if cls == 0:
            continue
        mask = seg == cls
        labeled, num_features = ndimage.label(mask)
        sizes = ndimage.sum(mask, labeled, index=np.arange(1, num_features + 1))
        for i, size in enumerate(sizes, start=1):
            if size >= min_voxels:
                cleaned[labeled == i] = cls
    return cleaned


def set_global_seed(seed: int = 0, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def should_early_stop(self, delta: float = 1e-4):
    """
    Stop if the mean‐dice hasn’t improved by > delta over the last `early_stop_count` epochs.
    """
    val_dice = self.validation_history["dice"]
    # not enough epochs yet
    if len(val_dice) < self.early_stop_count:
        return False

    # take the last `early_stop_count` dice values
    recent = list(val_dice.values())[-self.early_stop_count:]

    # if the best dice in that window is no better than the first epoch’s dice + delta, stop
    best_recent = max(recent)
    return best_recent <= (recent[0] + delta)
        

