"""
file_io.py

All disk I/O related to models, checkpoints, predictions, and history.
- save_checkpoint(): manage “last”/“best” ckpt saving with metadata.
- save_model(): dump final state_dict to models directory.
- save_nifti() & save_predictions(): convert tensors to NIfTI files.
- load_history(): read JSON logs of training/validation metrics.
- save_history_log(): serialize metric + LR histories to JSON.
"""


import torch
import pathlib
import numpy as np
import nibabel as nib
import datetime
import json
from pathlib import Path

def load_history(history_path: str) -> dict:
    """
    Load a JSON training history file into a Python dictionary.
    """
    path = Path(history_path)
    with path.open('r') as f:
        return json.load(f)

def load_nifti(file_path: str) -> np.ndarray:
    """
    Load a NIfTI file and return its image data as a NumPy array.
    """
    img = nib.load(str(file_path))
    return img.get_fdata()

def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, device: torch.device):
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state['model_state'])
    return model

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
        

def load_training_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
) -> tuple[int, int]:
    """
    Load model, optimizer, scheduler state, and return (epoch, global_step).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    epoch       = checkpoint["epoch"]
    global_step = checkpoint["global_step"]
    return epoch, global_step