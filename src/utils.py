import torch
import os
import pathlib
import numpy as np
import nibabel as nib
import time
import datetime
import yaml

def should_early_stop(self, delta=1e-4):
    val_loss = self.validation_history["loss"]
    if len(val_loss) < self.early_stop_count:
        return False
    recent = list(val_loss.values())[-self.early_stop_count:]
    # Stop if no epoch in the recent window lowered the validation loss by more than delta compared to the best value.
    return (min(recent) + delta) >= recent[0]


def save_checkpoint(self):
    """
    Save a checkpoint with the model's and optimizer's state, as well as metadata.
    """
    state = {
        "epoch": self.current_epoch,
        "global_step": self.global_step,
        "model_state": self.model.state_dict(),
        "optimizer_state": self.optimizer.state_dict(),
        "scheduler_state": self.scheduler.state_dict(),
        "validation_history": self.validation_history,
        "timestamp": time.strftime("%Y%m%d_%H%M%S")
    }
    config = {
        "learning_rate": self.optimizer.param_groups[0]["lr"],
        "batch_size": self.train_loader.batch_size,
        "loss_parameters": {
            "lambda_dice": self.loss_criterion.lambda_dice,
            "lambda_ce": self.loss_criterion.lambda_ce,
        },
        "model": self.model.__class__.__name__,
        "dataset": "HNTS-MRG Challenge 2024",
    }
    state["config"] = config

    filename = f"checkpoint_{self.global_step:06d}_{state['timestamp']}.ckpt"
    filepath = self.checkpoint_dir / filename
    self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, filepath)
    print(f"Checkpoint saved at step {self.global_step} to {filepath}")

    best_model = self.validation_history["loss"][self.global_step] == min(self.validation_history["loss"].values())
    if best_model:
        best_filepath = self.checkpoint_dir / "best.ckpt"
        torch.save(state, best_filepath)
        print(f"Best model updated at {best_filepath}")

def save_model(model, model_dir, filename=None):
    """
    Saves the final model's state_dict in the specified model directory.
    """
    model_dir = pathlib.Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = f"dynunet_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}.pth"
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


def freeze_encoder(self, freeze=True):
    for name, param in self.model.named_parameters():
        if "down" in name:
            param.requires_grad = not freeze