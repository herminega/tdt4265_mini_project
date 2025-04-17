import torch
import collections
import pathlib
import numpy as np
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.utils import one_hot
from monai.inferers import sliding_window_inference
from model import get_model
from dataloader import get_mri_dataloader
from utils import (
    save_predictions,
    log_metrics,
    should_early_stop,
    save_checkpoint,
    freeze_encoder,
    is_best_model,
    save_history_log,
)

class Trainer:
    def __init__(
        self,
        data_dir,
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        in_channels=1,
        out_channels=3,
        checkpoint_dir=None,
        scheduler_type="plateau",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.model = get_model("nnunet", in_channels, out_channels, pretrained=False).to(self.device)
        self.train_loader, self.val_loader = get_mri_dataloader(data_dir, "train", batch_size, validation_fraction=0.1)

        class_weights = torch.tensor([0.4, 1.5, 1.5]).to(self.device)
        self.loss_criterion = DiceCELoss(
            to_onehot_y=True, softmax=True, smooth_dr=0.0001,
            weight=class_weights, lambda_dice=0.7, lambda_ce=0.3
        )

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=5e-5)
        # compute total training steps for OneCycle (num_epochs * steps_per_epoch)
        total_steps = epochs * len(self.train_loader)
        self.scheduler = self._init_scheduler(scheduler_type, total_steps, base_lr=learning_rate)
        

        self.epochs = epochs
        self.early_stop_count = early_stop_count
        self.global_step = 0
        self.train_history = {"loss": collections.OrderedDict(), "dice": collections.OrderedDict()}
        self.validation_history = {"loss": collections.OrderedDict(), "dice": collections.OrderedDict()}

        self.checkpoint_dir = checkpoint_dir or pathlib.Path("results") / "checkpoints"
        self.base_dir = self.checkpoint_dir.parent
        self.predictions_dir = self.base_dir / "predictions"
        self.logs_dir = self.base_dir / "logs"
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.dice_metric = DiceMetric(include_background=False, reduction="none", get_not_nans=True)
        
    def _init_scheduler(self, scheduler_type, total_steps, base_lr):
        if scheduler_type.lower() == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min', factor=0.5, patience=10, verbose=True, min_lr=1e-6
            )
        elif scheduler_type.lower() == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_steps)
        
        elif scheduler_type.lower() == "onecycle":
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer      = self.optimizer,
                max_lr         = 1e-3,           # your peak LR
                total_steps    = total_steps,    # epochs * steps_per_epoch
                pct_start      = 0.15,           # 15% of total_steps warming up
                anneal_strategy= "cos",          # cosine annealing down
                div_factor     = 25.0,           # start LR = max_lr / div_factor → 4e-5
                final_div_factor=1000.0,         # end   LR = max_lr / final_div_factor → 1e-6
                three_phase    = False,          # two‑phase (up then down)
                last_epoch     = -1,
            )
        else:
            raise ValueError("Unsupported scheduler_type. Use 'plateau', 'cosine' or 'onecycle'.")

    
    def train_step(self, inputs, labels):
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        
        loss = self.loss_criterion(outputs, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()
        
        return loss.item(), outputs
    

    def train(self):
        for epoch in range(1, self.epochs + 1):
            print(f"\n=== Epoch {epoch} ===")
            self.current_epoch = epoch
            self.dice_metric.reset()
            epoch_loss = 0.0
            total_TP, total_FP, total_FN = {1: 0, 2: 0}, {1: 0, 2: 0}, {1: 0, 2: 0}

            for batch in tqdm(self.train_loader, desc="Training", dynamic_ncols=True):
                inputs, labels = batch["image"].to(self.device), batch["label"].to(self.device)
                loss, outputs = self.train_step(inputs, labels)
                self.global_step += 1
                epoch_loss += loss
                
                softmax_outputs = torch.softmax(outputs.detach(), dim=1)
                pred_indices = torch.argmax(softmax_outputs, dim=1)
                onehot_pred = one_hot(pred_indices.unsqueeze(1), num_classes=3)
                onehot_labels = one_hot(labels.long(), num_classes=3)
                self.dice_metric(onehot_pred, onehot_labels)

                self.train_history["loss"][self.global_step] = loss

                true = labels.squeeze(1)
                for cls in [1, 2]:
                    TP = ((pred_indices == cls) & (true == cls)).sum().item()
                    FP = ((pred_indices == cls) & (true != cls)).sum().item()
                    FN = ((pred_indices != cls) & (true == cls)).sum().item()
                    total_TP[cls] += TP
                    total_FP[cls] += FP
                    total_FN[cls] += FN

            metrics = self._finalize_epoch(epoch_loss, total_TP, total_FP, total_FN, self.train_history, loader_len=len(self.train_loader))
            val_metrics = self.validate()
            
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                if epoch == 70:
                    print("Switching to reduced-patience scheduler (patience=5)")
                    self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        self.optimizer, mode='min', factor=0.3, patience=7, verbose=True, min_lr=1e-6
                    )
                
                if epoch == 120:
                    print("Manual LR drop to 2e-5")
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = 2e-5
                    # Optional: Freeze scheduler after manual LR drop
                    self.scheduler = None
                    
                if self.scheduler is not None:
                    self.scheduler.step(val_metrics["loss"])
            else:
                self.scheduler.step()

            
            for param_group in self.optimizer.param_groups:
                print(f"Current LR: {param_group['lr']:.6f}")

            log_metrics(epoch, metrics, val_metrics)
            save_history_log(self.train_history, self.validation_history, path=self.logs_dir / f"history_{epoch:03d}.json")
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                validation_history=self.validation_history,
                loss_criterion=self.loss_criterion,
                train_loader=self.train_loader,
                epoch=self.current_epoch,
                global_step=self.global_step,
                total_epochs=self.epochs,
                checkpoint_dir=self.checkpoint_dir
            )
            
            if should_early_stop(self):
                print("Early stopping triggered.")
                break
                   
            
    def _finalize_epoch(self, epoch_loss, TP, FP, FN, history, loader_len):
        avg_loss = epoch_loss / loader_len  # <-- now it's flexible
        per_class_dice, _ = self.dice_metric.aggregate()
        per_class_dice = per_class_dice.cpu().numpy()
        dice_class1, dice_class2 = np.nanmean(per_class_dice, axis=0)[:2]
        mean_dice = np.nanmean([dice_class1, dice_class2])

        precision_class1 = TP[1] / (TP[1] + FP[1] + 1e-6)
        recall_class1 = TP[1] / (TP[1] + FN[1] + 1e-6)
        precision_class2 = TP[2] / (TP[2] + FP[2] + 1e-6)
        recall_class2 = TP[2] / (TP[2] + FN[2] + 1e-6)

        history["dice"][self.global_step] = mean_dice
        history.setdefault("dice_class1", {})[self.global_step] = dice_class1
        history.setdefault("dice_class2", {})[self.global_step] = dice_class2

        return {
            "loss": avg_loss,
            "mean_dice": mean_dice,
            "dice_class1": dice_class1,
            "dice_class2": dice_class2,
            "precision_class1": precision_class1,
            "recall_class1": recall_class1,
            "precision_class2": precision_class2,
            "recall_class2": recall_class2,
        }
    
    def validate(self, save=True):
        self.model.eval()
        self.dice_metric.reset()
        predictions_to_save, total_loss = [], 0
        TP, FP, FN = {1: 0, 2: 0}, {1: 0, 2: 0}, {1: 0, 2: 0}
        skipped = 0

        with torch.no_grad():
            for idx, batch in enumerate(self.val_loader):
                inputs, labels = batch["image"].to(self.device), batch["label"].to(self.device)
                outputs = sliding_window_inference(
                    inputs,                                  # [B, C, H, W, D]
                    roi_size=(192, 192, 48),                 # match your train crop
                    sw_batch_size=1,                         # how many windows to process at once
                    predictor=self.model,                    # your nnunet
                    overlap=0.5,                             # 50% overlap + gaussian blending
                )
                loss = self.loss_criterion(outputs, labels)
                total_loss += loss.item()

                softmax_outputs = torch.softmax(outputs.detach(), dim=1)
                pred_indices = torch.argmax(softmax_outputs, dim=1)
                onehot_pred = one_hot(pred_indices.detach().unsqueeze(1), num_classes=3)
                onehot_labels = one_hot(labels.detach().long(), num_classes=3)
                self.dice_metric(onehot_pred, onehot_labels)

                true = labels.squeeze(1)
                if true.sum() == 0:
                    skipped += 1
                    continue

                for cls in [1, 2]:
                    TP[cls] += ((pred_indices == cls) & (true == cls)).sum().item()
                    FP[cls] += ((pred_indices == cls) & (true != cls)).sum().item()
                    FN[cls] += ((pred_indices != cls) & (true == cls)).sum().item()

                if idx < 3:
                    predictions_to_save.append((inputs[0], labels[0], softmax_outputs[0], idx))

        print(f"[Summary] Skipped {skipped}/{len(self.val_loader)} validation batches (no tumor present)")

        per_class_dice, _ = self.dice_metric.aggregate()
        per_class_dice = per_class_dice.cpu().numpy()
        dice_class1, dice_class2 = np.nanmean(per_class_dice, axis=0)[:2]
        mean_dice = np.nanmean([dice_class1, dice_class2])
        avg_loss = total_loss / len(self.val_loader)

        self.validation_history["loss"][self.global_step] = avg_loss
        self.validation_history["dice"][self.global_step] = mean_dice
        self.validation_history.setdefault("dice_class1", {})[self.global_step] = dice_class1
        self.validation_history.setdefault("dice_class2", {})[self.global_step] = dice_class2

        if save and (self.current_epoch % 5 == 0 or is_best_model(self.global_step, self.validation_history)):
            for inputs, labels, outputs, idx in predictions_to_save:
                save_predictions(inputs, labels, outputs, idx, self.predictions_dir)

        return self._finalize_epoch(total_loss, TP, FP, FN, self.validation_history, loader_len=len(self.val_loader))




