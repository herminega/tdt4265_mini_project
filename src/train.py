import torch
import collections
import pathlib
import numpy as np
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
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
    def __init__(self, data_dir, batch_size, learning_rate, early_stop_count, epochs, in_channels=1, out_channels=3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = get_model("nnunet", in_channels, out_channels).to(self.device)
        self.train_loader, self.val_loader = get_mri_dataloader(data_dir, "train", batch_size, validation_fraction=0.1)

        class_weights = torch.tensor([0.5, 1.0, 1.2]).to(self.device)
        self.loss_criterion = DiceCELoss(to_onehot_y=False, softmax=True, lambda_dice=0.8, lambda_ce=0.2, class_weights=class_weights)
        self.dice_loss = DiceLoss(softmax=True)
        self.ce_loss = torch.nn.CrossEntropyLoss(weight=class_weights)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6)
        # torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2)
        # torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        self.scaler = GradScaler()
        self.epochs = epochs
        self.early_stop_count = early_stop_count
        self.global_step = 0

        self.train_history = dict(loss=collections.OrderedDict(), dice=collections.OrderedDict())
        self.validation_history = dict(loss=collections.OrderedDict(), dice=collections.OrderedDict())

        self.checkpoint_dir = pathlib.Path("checkpoints")
        self.dice_metric = DiceMetric(include_background=False, reduction="none", get_not_nans=True)

    def train_step(self, inputs, labels):
        self.model.train()
        self.optimizer.zero_grad()

        with autocast(device_type=self.device.type):
            outputs = self.model(inputs)
            loss = self.loss_criterion(outputs, labels)
            
            # Compute individual components for logging/monitoring (not used in training step)
            ce = self.ce_loss(outputs, labels.argmax(dim=1))
            dice = self.dice_loss(outputs, labels)
            
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return loss.item(), outputs, labels, dice.item(), ce.item()

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
                with autocast(self.device.type):
                    outputs = self.model(inputs)
                    loss = self.loss_criterion(outputs, labels)
                total_loss += loss.item()

                softmax_outputs = torch.softmax(outputs, dim=1)
                self.dice_metric(softmax_outputs, labels)

                pred = torch.argmax(softmax_outputs, dim=1)
                true = labels.argmax(dim=1)

                if true.sum() == 0:
                    print(f"[Skip] Sample {idx} has no tumor in ground truth.")
                    num_skipped += 1
                    continue
                num_total += 1

                for cls in [1, 2]:
                    TP = ((pred == cls) & (true == cls)).sum().item()
                    FP = ((pred == cls) & (true != cls)).sum().item()
                    FN = ((pred != cls) & (true == cls)).sum().item()
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

            total_TP, total_FP, total_FN = {1: 0, 2: 0}, {1: 0, 2: 0}, {1: 0, 2: 0}

            for batch in progress_bar:
                inputs, labels = batch["image"].to(self.device), batch["label"].to(self.device)
                loss, outputs, targets, dice_loss_val, ce_loss_val = self.train_step(inputs, labels)
                
                if self.global_step == 0:
                    print(f"labels shape: {labels.shape}")
                    print(f"outputs shape: {outputs.shape}")

                self.train_history.setdefault("ce_loss", {})[self.global_step] = ce_loss_val
                self.train_history.setdefault("dice_loss", {})[self.global_step] = dice_loss_val

                self.global_step += 1
                epoch_loss += loss
                softmax_outputs = torch.softmax(outputs, dim=1)
                self.dice_metric(softmax_outputs, targets)
                self.train_history["loss"][self.global_step] = loss
                progress_bar.set_postfix(loss=f"{loss:.4f}", ce_loss=f"{ce_loss_val:.4f}", dice_loss=f"{dice_loss_val:.4f}")

                pred = torch.argmax(softmax_outputs, dim=1)
                true = targets.argmax(dim=1)
                
                for cls in [1, 2]:
                    TP = ((pred == cls) & (true == cls)).sum().item()
                    FP = ((pred == cls) & (true != cls)).sum().item()
                    FN = ((pred != cls) & (true == cls)).sum().item()
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
            #self.scheduler.step()
            #val_metrics = self.validate()

            # Update ReduceLROnPlateau with validation loss
            self.scheduler.step(val_metrics["loss"])  # Instead of self.scheduler.step()

            # Optional: print current LR
            for param_group in self.optimizer.param_groups:
                print(f"Current LR: {param_group['lr']:.6f}")
            
            save_checkpoint(self)
            log_metrics(epoch, train_metrics, val_metrics)

            if should_early_stop(self):
                print("Early stopping triggered.")
                break