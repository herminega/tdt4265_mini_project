import torch
import os
import pathlib
import numpy as np
import nibabel as nib

def should_early_stop(self):
        val_loss = self.validation_history["loss"]
        if len(val_loss) < self.early_stop_count:
            return False
        recent = list(val_loss.values())[-self.early_stop_count:]
        return recent[0] == min(recent)


def save_checkpoint(self):
        """
        Saves model checkpoint. This is useful to resume training from a saved state.
        """
        best_model = self.validation_history["loss"][self.global_step] == min(self.validation_history["loss"].values())
        state_dict = self.model.state_dict()
        filepath = self.checkpoint_dir.joinpath(f"{self.global_step}.ckpt")

        # Ensure checkpoint directory exists
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        torch.save(state_dict, filepath)
        if best_model:
            torch.save(state_dict, self.checkpoint_dir.joinpath("best.ckpt"))

        print(f"Model checkpoint saved at step {self.global_step}")

def save_model(self, path="results/final_model.pth"):
    """
    Saves the final trained model separately from checkpoints.
    Use this to save a model after training is complete.
    """
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(self.model.state_dict(), path)
    print(f"Final model saved at {path}")
    
def save_nifti(tensor, filename):
        tensor = tensor.cpu().numpy().squeeze().astype(np.uint8)
        img = nib.Nifti1Image(tensor, affine=np.eye(4))
        nib.save(img, filename)
    
def save_predictions(self, input_tensor, label_tensor, output_tensor, meta_dict, index, output_dir):
    pred = torch.argmax(output_tensor, dim=0).cpu().numpy().astype(np.uint8)
    label = torch.argmax(label_tensor, dim=0).cpu().numpy().astype(np.uint8)

    pred_img = nib.Nifti1Image(pred, affine=meta_dict["affine"])
    label_img = nib.Nifti1Image(label, affine=meta_dict["affine"])
    input_img = nib.Nifti1Image(input_tensor.cpu().numpy().squeeze(), affine=meta_dict["affine"])

    nib.save(pred_img, os.path.join(output_dir, f"prediction_{index}.nii.gz"))
    nib.save(label_img, os.path.join(output_dir, f"label_{index}.nii.gz"))
    nib.save(input_img, os.path.join(output_dir, f"image_{index}.nii.gz"))
    
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