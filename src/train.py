import torch
import collections
import pathlib
import os
import nibabel as nib
import datetime
import numpy as np
from monai.losses import GeneralizedDiceLoss, DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from model import get_model  # Import model
from dataloader import get_mri_dataloader  # Import MONAI dataloader
from tqdm import tqdm
from torch.amp import GradScaler, autocast


class Trainer:
    def __init__(self, 
                 data_dir: str,
                 batch_size: int, 
                 learning_rate: float, 
                 early_stop_count: int, 
                 epochs: int, 
                 in_channels: int = 1, 
                 out_channels: int = 3):
        """
        Trainer class for MRI segmentation.
        """
        torch.cuda.empty_cache()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize model & move to GPU if available
        self.model = get_model(model_type="dynunet", in_channels=in_channels, out_channels=out_channels, pretrained=False).to(self.device)

        # Load Data
        self.train_loader, self.val_loader = get_mri_dataloader(data_dir, "train", batch_size, validation_fraction=0.1)
        
        ### Loss functions options ###
        class_weights = torch.tensor([0.01, 1.0, 3.0]).to(self.device)
        # 1. (Generalized Dice Loss with softmax)
        #self.loss_criterion = GeneralizedDiceLoss(softmax=True, include_background=True) 
        # 2. Class weights for handling imbalance
        #self.loss_criterion = DiceLoss(
        #    softmax=True, 
        #    weight=class_weights)
        # 3. Combined Dice + Cross-Entropy Loss
        self.loss_criterion = DiceCELoss(to_onehot_y=False, softmax=True, lambda_dice=0.7, lambda_ce=0.3)

        # Optimizer with weight decay (L2 regularization)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        # Mixed precision training
        self.scaler = GradScaler()
        self.epochs = epochs
        self.early_stop_count = early_stop_count
        self.global_step = 0

        # Training & validation history tracking
        self.train_history = dict(loss=collections.OrderedDict(), dice=collections.OrderedDict())
        self.validation_history = dict(loss=collections.OrderedDict(), dice=collections.OrderedDict())

        # Directory for saving checkpoints
        self.checkpoint_dir = pathlib.Path("checkpoints")

        # Initialize DiceMetric once (before training starts)
        self.dice_metric = DiceMetric(include_background=False, reduction="none", get_not_nans=True)

    def train_step(self, inputs, labels):
        self.model.train()
        self.optimizer.zero_grad()

        with autocast(device_type=self.device.type):
            outputs = self.model(inputs)
            loss = self.loss_criterion(outputs, labels)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item(), outputs, labels  # return predictions and targets to compute Dice later


    def save_nifti(self, tensor, filename):
        tensor = tensor.cpu().numpy().squeeze().astype(np.uint8)
        img = nib.Nifti1Image(tensor, affine=np.eye(4))
        nib.save(img, filename)
    
    def _save_predictions(self, input_tensor, label_tensor, output_tensor, index, output_dir):
        label_discrete = torch.argmax(label_tensor, dim=0)
        prediction_discrete = torch.argmax(output_tensor, dim=0)
        self.save_nifti(input_tensor, f"{output_dir}/image_{index}.nii.gz")
        self.save_nifti(label_discrete, f"{output_dir}/label_{index}.nii.gz")
        self.save_nifti(prediction_discrete, f"{output_dir}/prediction_{index}.nii.gz")

    def validate(self, save_predictions=True, output_dir="results/predictions"):
        self.model.eval()
        self.dice_metric.reset()
        os.makedirs(output_dir, exist_ok=True)
        
        total_TP = {1: 0, 2: 0}
        total_FP = {1: 0, 2: 0}
        total_FN = {1: 0, 2: 0}

        total_loss = 0
        with torch.no_grad():
            for idx, batch in enumerate(self.val_loader):
                inputs, labels = batch["image"].to(self.device), batch["label"].to(self.device)
                with autocast(self.device.type):
                    outputs = self.model(inputs)
                    loss = self.loss_criterion(outputs, labels)
                
                total_loss += loss.item()
                
                softmax_outputs = torch.softmax(outputs, dim=1)             
                self.dice_metric(softmax_outputs, labels)
                
                pred = torch.argmax(softmax_outputs, dim=1)
                true = labels.argmax(dim=1)
                #labels.shape
                print(f"Pred shape: {pred.shape}, True shape: {true.shape}")

                for cls in [1, 2]:
                    TP = ((pred == cls) & (true == cls)).sum().item()
                    FP = ((pred == cls) & (true != cls)).sum().item()
                    FN = ((pred != cls) & (true == cls)).sum().item()

                    total_TP[cls] += TP
                    total_FP[cls] += FP
                    total_FN[cls] += FN

                if save_predictions and idx < 5:
                    self._save_predictions(inputs[0], labels[0], softmax_outputs[0], idx, output_dir)
        
        for cls in [1, 2]:
            TP = total_TP[cls]
            FP = total_FP[cls]
            FN = total_FN[cls]
            precision = TP / (TP + FP + 1e-6)
            recall = TP / (TP + FN + 1e-6)
            print(f"Class {cls}: Precision {precision:.4f}, Recall {recall:.4f}")

        avg_loss = total_loss / len(self.val_loader)

        # Aggregate Dice scores across validation epoch
        per_class_dice, _ = self.dice_metric.aggregate()
        per_class_dice = per_class_dice.cpu().numpy()  # shape: (B, C)
        class_dice_means = np.nanmean(per_class_dice, axis=0)

        class1_dice = class_dice_means[0]
        class2_dice = class_dice_means[1]
        mean_dice = np.nanmean(class_dice_means)

        # Save to validation history
        self.validation_history["loss"][self.global_step] = avg_loss
        self.validation_history["dice"][self.global_step] = mean_dice
        self.validation_history.setdefault("dice_class1", {})[self.global_step] = class1_dice
        self.validation_history.setdefault("dice_class2", {})[self.global_step] = class2_dice

        print(f"\nValidation - Loss: {avg_loss:.4f}")
        print(f"  GTVp (class 1) Dice: {class1_dice:.4f}")
        print(f"  GTVn (class 2) Dice: {class2_dice:.4f}")
        print(f"  Mean Dice (tumor only): {mean_dice:.4f}")

        return avg_loss, mean_dice


    def should_early_stop(self):
        val_loss = self.validation_history["loss"]
        if len(val_loss) < self.early_stop_count:
            return False
        recent = list(val_loss.values())[-self.early_stop_count:]
        return recent[0] == min(recent)

    def train(self):
        for epoch in range(1, self.epochs + 1):
            print(f"\n=== Epoch {epoch} ===")
            epoch_loss = 0.0
            self.dice_metric.reset()

            progress_bar = tqdm(self.train_loader, desc="Training", dynamic_ncols=True)
            for batch in progress_bar:
                inputs, labels = batch["image"].to(self.device), batch["label"].to(self.device)
        
                loss, outputs, targets = self.train_step(inputs, labels)
                self.global_step += 1
                epoch_loss += loss
                
                softmax_outputs = torch.softmax(outputs, dim=1)
                self.dice_metric(softmax_outputs, targets)

                self.train_history["loss"][self.global_step] = loss
                progress_bar.set_postfix(loss=f"{loss:.4f}")

            # Aggregate Dice over the whole epoch
            per_class_dice, _ = self.dice_metric.aggregate()
            per_class_dice = per_class_dice.cpu().numpy()  # shape: (B, C)
            class_dice_means = np.nanmean(per_class_dice, axis=0)  # shape: (C,)

            class1_dice = class_dice_means[0]
            class2_dice = class_dice_means[1]
            mean_dice = np.nanmean(class_dice_means)

            avg_loss = epoch_loss / len(self.train_loader)

            print(f"\nEpoch Summary:")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  GTVp (class 1) Dice: {class1_dice:.4f}")
            print(f"  GTVn (class 2) Dice: {class2_dice:.4f}")
            print(f"  Mean Dice (tumor only): {mean_dice:.4f}")

            self.train_history["dice"][self.global_step] = mean_dice
            self.train_history.setdefault("dice_class1", {})[self.global_step] = class1_dice
            self.train_history.setdefault("dice_class2", {})[self.global_step] = class2_dice

            val_loss, val_dice = self.validate()
            self.scheduler.step()
            self.save_checkpoint()

            if self.should_early_stop():
                print("Early stopping triggered.")
                break


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

if __name__ == "__main__":
    # Set dataset directory
    """
    IDUN: /cluster/projects/vc/data/mic/open/HNTS-MRG
–   Cybele: /datasets/tdt4265/mic/open/HNTS-MRG
    """
    data_dir = "/cluster/projects/vc/data/mic/open/HNTS-MRG"

    # Initialize Trainer
    trainer = Trainer(
        data_dir=data_dir,
        batch_size=3,
        learning_rate=5e-4,
        early_stop_count=5,
        epochs=10,
        in_channels=1,
        out_channels=3,
    )

    # Train the model
    trainer.train()

    # Save the final model
    
    model_name = f"swin_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}.pth"
    trainer.save_model(f"results/models/{model_name}")
   

