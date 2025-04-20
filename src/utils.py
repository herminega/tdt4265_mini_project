import torch
import os, random
import pathlib
import numpy as np
import nibabel as nib
import datetime
import json
from scipy import ndimage


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

# utils.py

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



def save_checkpoint(model, optimizer, scheduler, validation_history, loss_criterion,
                    train_loader, epoch, global_step, total_epochs, checkpoint_dir):
    """
    Save a model checkpoint, including metadata and best/last logic.
    """
    import time
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "validation_history": validation_history,
        "timestamp": time.strftime("%Y%m%d_%H%M%S")
    }

    config = {
        "learning_rate": optimizer.param_groups[0]["lr"],
        "batch_size": train_loader.batch_size,
        "loss_parameters": {
            "lambda_dice": loss_criterion.lambda_dice,
            "lambda_ce": loss_criterion.lambda_ce,
        },
        "model": model.__class__.__name__,
        "dataset": "HNTS-MRG Challenge 2024",
    }

    state["config"] = config

    # Save "last" checkpoint every N or final epoch
    if epoch % 5 == 0 or epoch == total_epochs:
        last_path = checkpoint_dir / "last.ckpt"
        torch.save(state, last_path)
        print(f"Last checkpoint saved to: {last_path}")


    # Save "best" if current mean_dice is best
    best = validation_history["dice"][global_step] == max(validation_history["dice"].values())
    if best:
        best_path = checkpoint_dir / "best.ckpt"
        torch.save(state, best_path)
        print(f"Best model updated at: {best_path} (based on highest mean Dice)")

def save_model(model, model_dir, filename=None):
    """
    Saves the final model's state_dict in the specified model directory.
    """
    model_dir = pathlib.Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}.pth"
    model_path = model_dir / filename
    torch.save(model.state_dict(), model_path)
    print(f"Final model saved at {model_path}")
    
def save_nifti(tensor, filename):
        tensor = tensor.cpu().numpy().squeeze().astype(np.uint8)
        img = nib.Nifti1Image(tensor, affine=np.eye(4))
        nib.save(img, filename)
    
def save_predictions(input_tensor, label_tensor, output_tensor, index, output_dir):
    # If label_tensor has more than one channel, assume it's one-hot encoded and use argmax.
    if label_tensor.shape[0] > 1:
        label_discrete = torch.argmax(label_tensor, dim=0)
    else:
        # Otherwise, squeeze the channel dimension.
        label_discrete = label_tensor.squeeze(0)
    
    # For predictions, since they are outputs from the network and one-hot encoded, use argmax.
    prediction_discrete = torch.argmax(output_tensor, dim=0)
    
    save_nifti(input_tensor, f"{output_dir}/image_{index}.nii.gz")
    save_nifti(label_discrete, f"{output_dir}/label_{index}.nii.gz")
    save_nifti(prediction_discrete, f"{output_dir}/prediction_{index}.nii.gz")

    
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

def save_history_log(train_history, val_history, lr_history, path):
    def to_serializable(obj):
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    history = {
        "train": to_serializable(train_history),
        "validation": to_serializable(val_history),
        "lr": to_serializable(lr_history)
    }

    with open(path, "w") as f:
        json.dump(history, f, indent=4)
        

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