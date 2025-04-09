import torch
import collections
import pathlib
import numpy as np
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.utils import one_hot
from model import get_model
from dataloader import get_mri_dataloader
from utils import (
    save_predictions,
    log_metrics,
    should_early_stop,
    save_checkpoint,
    freeze_encoder
)

class Trainer:
    def __init__(self, data_dir, batch_size, learning_rate, early_stop_count, epochs, in_channels=1, out_channels=3, checkpoint_dir=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = get_model("dynunet", in_channels, out_channels).to(self.device)
        self.train_loader, self.val_loader = get_mri_dataloader(data_dir, "train", batch_size, validation_fraction=0.1)

        class_weights = torch.tensor([0.5, 1.0, 1.2]).to(self.device)
        self.loss_criterion = DiceCELoss(to_onehot_y=True, softmax=True, smooth_dr= 0.001, lambda_dice=0.7, lambda_ce=0.3)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=8, verbose=True, min_lr=1e-6)
        # torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        # torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        #self.scaler = GradScaler()
        self.epochs = epochs
        self.early_stop_count = early_stop_count
        self.global_step = 0

        self.train_history = dict(loss=collections.OrderedDict(), dice=collections.OrderedDict())
        self.validation_history = dict(loss=collections.OrderedDict(), dice=collections.OrderedDict())
        
        if checkpoint_dir is None:
            self.checkpoint_dir = pathlib.Path("results") / "checkpoints"
        else:
            self.checkpoint_dir = checkpoint_dir

        self.checkpoint_dir = pathlib.Path("checkpoints")
        self.dice_metric = DiceMetric(include_background=False, reduction="none", get_not_nans=True)
        
    def train_step(self, inputs, labels):
        self.model.train()
        self.optimizer.zero_grad()

        #with autocast(device_type=self.device.type):
        outputs = self.model(inputs)
        loss = self.loss_criterion(outputs, labels)

        # Scale loss and perform backward pass.
        #self.scaler.scale(loss).backward()
        #self.scaler.step(self.optimizer)
        #self.scaler.update()

        # Instead of scaling the loss, we simply backpropagate:
        loss.backward()

        # Optional: apply gradient clipping to avoid exploding gradients.
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()
        
        return loss.item(), outputs, labels


    def validate(self, save=True, output_dir="results/predictions"):
        self.model.eval()
        self.dice_metric.reset()
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        total_loss, num_total, num_skipped = 0, 0, 0
        total_TP, total_FP, total_FN = {1: 0, 2: 0}, {1: 0, 2: 0}, {1: 0, 2: 0}

        with torch.no_grad():
            for idx, batch in enumerate(self.val_loader):
                inputs, labels = batch["image"].to(self.device), batch["label"].to(self.device)
                
                #with autocast(device_type=self.device.type):
                outputs = self.model(inputs)
                loss = self.loss_criterion(outputs, labels)
                
                total_loss += loss.item()

                # Detach outputs so that you do not keep the gradient history for metric computation.
                softmax_outputs = torch.softmax(outputs.detach(), dim=1)
                pred_indices = torch.argmax(softmax_outputs, dim=1)
                onehot_pred = one_hot(pred_indices.detach().unsqueeze(1), num_classes=3)
                onehot_labels = one_hot(labels.detach().long(), num_classes=3)
                self.dice_metric(onehot_pred, onehot_labels)
                
                # Also compute discrete predictions and compare for TP/FP/FN (for precision/recall).
                true = labels.squeeze(1)                     # [B,H,W,D]
                
                if true.sum() == 0:
                    print(f"[Skip] Sample {idx} has no tumor in ground truth.")
                    num_skipped += 1
                    continue
                num_total += 1

                for cls in [1, 2]:
                    TP = ((pred_indices == cls) & (true == cls)).sum().item()
                    FP = ((pred_indices == cls) & (true != cls)).sum().item()
                    FN = ((pred_indices != cls) & (true == cls)).sum().item()
                    total_TP[cls] += TP
                    total_FP[cls] += FP
                    total_FN[cls] += FN

                if save and idx < 5:
                    save_predictions(inputs[0], labels[0], softmax_outputs[0], idx, output_dir)

        print(f"[Summary] Skipped {num_skipped}/{num_total + num_skipped} validation batches (no tumor present)")

        avg_loss = total_loss / len(self.val_loader)
        per_class_dice, _ = self.dice_metric.aggregate()
        per_class_dice = per_class_dice.cpu().numpy()
        class_dice_means = np.nanmean(per_class_dice, axis=0)

        class1_dice = class_dice_means[0]
        class2_dice = class_dice_means[1]
        mean_dice = np.nanmean(class_dice_means)

        self.validation_history["loss"][self.global_step] = avg_loss
        self.validation_history["dice"][self.global_step] = mean_dice
        self.validation_history.setdefault("dice_class1", {})[self.global_step] = class1_dice
        self.validation_history.setdefault("dice_class2", {})[self.global_step] = class2_dice

        precision_class1 = total_TP[1] / (total_TP[1] + total_FP[1] + 1e-6)
        recall_class1 = total_TP[1] / (total_TP[1] + total_FN[1] + 1e-6)
        precision_class2 = total_TP[2] / (total_TP[2] + total_FP[2] + 1e-6)
        recall_class2 = total_TP[2] / (total_TP[2] + total_FN[2] + 1e-6)

        return {
            "loss": avg_loss,
            "mean_dice": mean_dice,
            "dice_class1": class1_dice,
            "dice_class2": class2_dice,
            "precision_class1": precision_class1,
            "recall_class1": recall_class1,
            "precision_class2": precision_class2,
            "recall_class2": recall_class2,
        }

    def train(self):
        for epoch in range(1, self.epochs + 1):
            print(f"\n=== Epoch {epoch} ===")
            #freeze_encoder(self.model, freeze=(epoch <= 5))

            epoch_loss = 0.0
            self.dice_metric.reset()
            progress_bar = tqdm(self.train_loader, desc="Training", dynamic_ncols=True)
            self.current_epoch = epoch

            total_TP, total_FP, total_FN = {1: 0, 2: 0}, {1: 0, 2: 0}, {1: 0, 2: 0}

            for batch in progress_bar:
                inputs, labels = batch["image"].to(self.device), batch["label"].to(self.device)
                loss, outputs, labels = self.train_step(inputs, labels)
                
                if self.global_step == 0:
                    print(f"labels shape: {labels.shape}")
                    print(f"outputs shape: {outputs.shape}")

                self.global_step += 1
                epoch_loss += loss
                
                # Detach outputs so that you do not keep the gradient history for metric computation.
                softmax_outputs = torch.softmax(outputs.detach(), dim=1)
                pred_indices = torch.argmax(softmax_outputs, dim=1)
                onehot_pred = one_hot(pred_indices.detach().unsqueeze(1), num_classes=3)
                onehot_labels = one_hot(labels.detach().long(), num_classes=3)
                self.dice_metric(onehot_pred, onehot_labels)
                
                self.train_history["loss"][self.global_step] = loss
                progress_bar.set_postfix(loss=f"{loss:.4f}")

                true = labels.squeeze(1)
                
                for cls in [1, 2]:
                    TP = ((pred_indices == cls) & (true == cls)).sum().item()
                    FP = ((pred_indices == cls) & (true != cls)).sum().item()
                    FN = ((pred_indices != cls) & (true == cls)).sum().item()
                    total_TP[cls] += TP
                    total_FP[cls] += FP
                    total_FN[cls] += FN

            per_class_dice, _ = self.dice_metric.aggregate()
            per_class_dice = per_class_dice.cpu().numpy()
            class_dice_means = np.nanmean(per_class_dice, axis=0)
            class1_dice = class_dice_means[0]
            class2_dice = class_dice_means[1]
            mean_dice = np.nanmean(class_dice_means)
            avg_loss = epoch_loss / len(self.train_loader)

            precision_class1 = total_TP[1] / (total_TP[1] + total_FP[1] + 1e-6)
            recall_class1 = total_TP[1] / (total_TP[1] + total_FN[1] + 1e-6)
            precision_class2 = total_TP[2] / (total_TP[2] + total_FP[2] + 1e-6)
            recall_class2 = total_TP[2] / (total_TP[2] + total_FN[2] + 1e-6)

            self.train_history["dice"][self.global_step] = mean_dice
            self.train_history.setdefault("dice_class1", {})[self.global_step] = class1_dice
            self.train_history.setdefault("dice_class2", {})[self.global_step] = class2_dice

            train_metrics = {
                "loss": avg_loss,
                "mean_dice": mean_dice,
                "dice_class1": class1_dice,
                "dice_class2": class2_dice,
                "precision_class1": precision_class1,
                "recall_class1": recall_class1,
                "precision_class2": precision_class2,
                "recall_class2": recall_class2,
            }

            val_metrics = self.validate()
            self.scheduler.step(val_metrics["loss"])  # Adjust learning rate based on validation loss.

            # Optional: print current learning rate.
            for param_group in self.optimizer.param_groups:
                print(f"Current LR: {param_group['lr']:.6f}")
            
            save_checkpoint(self)
            log_metrics(epoch, train_metrics, val_metrics)
            
            if should_early_stop(self):
                print("Early stopping triggered.")
                break

