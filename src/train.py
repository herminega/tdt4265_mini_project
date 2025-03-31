import torch
import time
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
        self.model = get_model(model_type="swinunetr", in_channels=in_channels, out_channels=out_channels, pretrained=True).to(self.device)
        print("Loaded type:", type(self.model))

        # Load Data
        self.train_loader, self.val_loader = get_mri_dataloader(data_dir, "train", batch_size, validation_fraction=0.1)
        
        ### Loss functions options ###
        class_weights = torch.tensor([0.1, 1.0, 1.5]).to(self.device)  # Example: Background=0.01, GTVp=1.0, GTVn=2.0
        # 1. (Generalized Dice Loss with softmax)
        #self.loss_criterion = GeneralizedDiceLoss(softmax=True, include_background=True) 
        # 2. Class weights for handling imbalance
        #self.loss_criterion = DiceLoss(
        #    softmax=True, 
        #    weight=class_weights)
        # 3. Combined Dice + Cross-Entropy Loss
        self.loss_criterion = DiceCELoss(softmax=True, lambda_dice=0.3, lambda_ce=0.5, weight=class_weights)

        # Optimizer with weight decay (L2 regularization)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)

        # Learning rate scheduler (Cosine Annealing)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        # Mixed precision training
        self.scaler = GradScaler()

        # Training settings
        self.epochs = epochs
        self.early_stop_count = early_stop_count
        self.global_step = 0
        self.start_time = time.time()

        # Training & validation history tracking
        self.train_history = dict(loss=collections.OrderedDict(), dice=collections.OrderedDict())
        self.validation_history = dict(loss=collections.OrderedDict(), dice=collections.OrderedDict())

        # Directory for saving checkpoints
        self.checkpoint_dir = pathlib.Path("checkpoints")

        # Initialize DiceMetric once (before training starts)
        self.dice_metric = DiceMetric(include_background=True, reduction="none", get_not_nans=True)

    def train_step(self, batch):
        """
        Performs forward and backward pass for one batch.
        """
        self.model.train()
        inputs, labels = batch["image"].to(self.device), batch["label"].to(self.device)
        
        self.optimizer.zero_grad()
               
        with autocast(device_type=self.device.type):  # Enable mixed precision
            outputs = self.model(inputs)
            # Compute loss using raw logits
            loss = self.loss_criterion(outputs, labels)
        
        # Backpropagation with mixed precision
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
          
        # Compute Dice score using MONAI DiceMetric
        self.dice_metric.reset()
        softmax_outputs = torch.softmax(outputs, dim=1)
        self.dice_metric(softmax_outputs, labels)
        dice_scores, _ = self.dice_metric.aggregate()
        dice_per_class = dice_scores.detach().cpu().numpy()
        overall_dice = dice_per_class.mean()

        return loss.item(), overall_dice, dice_per_class
    
    def save_nifti(self, tensor, filename):
        """ Convert PyTorch tensor to NIfTI and save it. """
        tensor = tensor.cpu().numpy().squeeze()  # Convert tensor to numpy & remove batch dim
        tensor = tensor.astype(np.uint8)  # Convert to uint8 (safe for segmentations)
    
        img = nib.Nifti1Image(tensor, affine=np.eye(4))  # Identity affine (adjust if needed)
        nib.save(img, filename)
      

    def validate(self, save_predictions=True, output_dir="results/predictions"):
        """
        Computes validation loss and per-class dice score.
        """
        self.model.eval()
        val_loss = 0
        all_dice_scores = []

        num_batches = len(self.val_loader)
        self.dice_metric.reset()
        os.makedirs(output_dir, exist_ok=True)

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                inputs, labels = batch["image"].to(self.device), batch["label"].to(self.device)

                with autocast(device_type=self.device.type):
                    outputs = self.model(inputs)
                    loss = self.loss_criterion(outputs, labels)

                val_loss += loss.item()

                softmax_outputs = torch.softmax(outputs, dim=1)
                self.dice_metric(softmax_outputs, labels)

                # Save predictions (only first few batches)
                if save_predictions and batch_idx < 5:
                    labels_discrete = torch.argmax(labels, dim=1)
                    self.save_nifti(tensor=inputs[0], filename=f"{output_dir}/image_{batch_idx}.nii.gz")
                    self.save_nifti(tensor=labels_discrete[0], filename=f"{output_dir}/label_{batch_idx}.nii.gz")
                    self.save_nifti(tensor=torch.argmax(softmax_outputs[0], dim=0), filename=f"{output_dir}/prediction_{batch_idx}.nii.gz")

        # Aggregate after the full epoch
        dice_scores, _ = self.dice_metric.aggregate()
        dice_per_class = dice_scores.detach().cpu().numpy()
        overall_dice = dice_per_class.mean()
        avg_loss = val_loss / num_batches

        # Log to history
        self.validation_history["loss"][self.global_step] = avg_loss
        self.validation_history["dice"][self.global_step] = overall_dice

        # Print summary
        print(f"\nValidation - Loss: {avg_loss:.4f}, Dice Score: {overall_dice:.4f}")
        for i, score in enumerate(dice_per_class):
            print(f"   ↪ Class {i}: Dice = {score:.4f}")

        return avg_loss, overall_dice


    def should_early_stop(self):
        """
        Checks if validation loss hasn't improved in `early_stop_count` epochs.
        """
        val_loss = self.validation_history["loss"]
        if len(val_loss) < self.early_stop_count:
            return False

        recent_losses = list(val_loss.values())[-self.early_stop_count:]
        if recent_losses[0] == min(recent_losses):  # No improvement
            print("Early stopping triggered due to no improvement.")
            return True
        return False

    def train(self):
        """
        Trains the model for `self.epochs` epochs.
        """

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0
            epoch_dice = 0
            num_batches = len(self.train_loader)    
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True, leave=True)
            for batch_idx, batch in enumerate(progress_bar):
                loss, dice, dice_per_class = self.train_step(batch)
                epoch_loss += loss
                epoch_dice += dice
                self.global_step += 1

                # Store loss history
                self.train_history["loss"][self.global_step] = loss
                self.train_history["dice"][self.global_step] = dice
                
                # Safely extract class scores (default to 0.0 if missing)
                gtvp_dice = float(dice_per_class[1]) if len(dice_per_class) > 1 else 0.0
                gtvn_dice = float(dice_per_class[2]) if len(dice_per_class) > 2 else 0.0

                progress_bar.set_postfix(
                    loss=f"{loss:.4f}",
                    dice=f"{dice:.4f}",
                    GTVp=f"{gtvp_dice:.3f}",
                    GTVn=f"{gtvn_dice:.3f}"
                )


            progress_bar.close()  # Ensure clean closing of tqdm bar

            # Compute average loss/dice for epoch
            avg_loss = epoch_loss / num_batches
            avg_dice = epoch_dice / num_batches
            print(f"Epoch {epoch} - Loss: {avg_loss:.4f}, Dice Score: {avg_dice:.4f}")
            
            # Validation step
            val_loss, val_dice = self.validate(save_predictions = True, output_dir = "results/predictions")

            self.validation_history["loss"][self.global_step] = val_loss
            self.validation_history["dice"][self.global_step] = val_dice

            # Update learning rate scheduler
            self.scheduler.step()

            # Save checkpoint & check early stopping
            self.save_checkpoint()
            if self.should_early_stop():
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
    data_dir = "/cluster/projects/vc/data/mic/open/HNTS-MRG"

    # Initialize Trainer
    trainer = Trainer(
        data_dir=data_dir,
        batch_size=2,
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
   

