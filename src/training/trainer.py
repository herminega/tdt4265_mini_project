"""
trainer.py

High‑level training loop encapsulated in Trainer class.
- Initializes model, optimizer, scheduler, loss (Dice+CE), and data loaders.
- Implements train/validate epochs with connected‑component cleanup on predictions.
- Tracks metrics (loss, Dice, precision, recall) and logs them.
- Handles checkpointing (“last” every N epochs, “best” on validation improvement).
- Supports early stopping based on mean‑Dice plateau.
"""


import torch
import collections
import pathlib
import numpy as np
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.utils import one_hot
from monai.inferers import sliding_window_inference

from src.models.model import get_model
from src.dataloader.dataloader import get_mri_dataloader
from src.utils.file_io import save_predictions, save_checkpoint, save_history_log
from src.utils.metrics import set_global_seed, log_metrics, should_early_stop, is_best_model, remove_small_cc


class Trainer:
    """
    Trainer encapsulates the full training + validation loop for a 3D segmentation model.

    Attributes:
        model: the nn.Module being trained
        optimizer: optimizer instance
        scheduler: learning rate scheduler
        loss_criterion: combined Dice+CE loss
        train_loader, val_loader: data loaders
        device: torch device
        histories: dicts to store training/validation metrics and lr
    """
    def __init__(
        self,
        data_dir=None,
        batch_size=None,
        train_loader=None,
        val_loader=None,
        learning_rate=None,
        early_stop_count=None,
        epochs=None,
        in_channels=1,
        out_channels=3,
        checkpoint_dir=None,
        scheduler_type="onecycle",
    ):
        # set seed for reproducibility (optional if done globally)
        set_global_seed(0)

        # device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # build the model
        self.model = get_model("nnunet", in_channels, out_channels, pretrained=False).to(self.device)

        # store hyperparameters
        self.epochs = epochs
        self.early_stop_count = early_stop_count

        # prepare data loaders
        if train_loader and val_loader:
            self.train_loader, self.val_loader = train_loader, val_loader
        else:
            assert data_dir and batch_size, \
                "Provide either (train_loader & val_loader) or (data_dir & batch_size)"
            self.train_loader, self.val_loader = get_mri_dataloader(
                data_dir, batch_size=batch_size, validation_fraction=0.1
            )

        # define loss: weighted Dice + CrossEntropy
        class_weights = torch.tensor([0.4, 1.7, 1.5]).to(self.device)
        self.loss_criterion = DiceCELoss(
            to_onehot_y=True, softmax=True,
            weight=class_weights,
            smooth_dr=1e-4, lambda_dice=0.5, lambda_ce=0.5
        )

        # optimizer + scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=5e-5)
        self.scheduler = self._init_scheduler(scheduler_type, base_lr=learning_rate)

        # bookkeeping
        self.global_step = 0
        self.train_history = {"loss": collections.OrderedDict(), "dice": collections.OrderedDict()}
        self.validation_history = {"loss": collections.OrderedDict(), "dice": collections.OrderedDict()}
        self.lr_history = collections.OrderedDict()

        # checkpoint and logging directories
        self.checkpoint_dir = pathlib.Path(checkpoint_dir or "results/checkpoints")
        self.predictions_dir = self.checkpoint_dir.parent / "predictions"
        self.logs_dir = self.checkpoint_dir.parent / "logs"
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # metric accumulator
        self.dice_metric = DiceMetric(include_background=False, reduction="none", get_not_nans=True)

    def _init_scheduler(self, scheduler_type, base_lr):
        """
        Initialize learning rate scheduler.
        Supports 'plateau', 'cosine', 'onecycle'.
        """
        if scheduler_type.lower() == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5,
                patience=12, threshold=1e-4, cooldown=4,
                min_lr=1e-6, verbose=True
            )
        elif scheduler_type.lower() == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs, eta_min=1e-6
            )
        elif scheduler_type.lower() == "onecycle":
            steps_per_epoch = len(self.train_loader)
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=base_lr,
                epochs=self.epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.3,
                div_factor=25,
                final_div_factor=1e4,
            )
        else:
            raise ValueError("scheduler_type must be 'plateau', 'cosine', or 'onecycle'")

    def train_step(self, inputs, labels):
        """
        Single gradient update step.
        1) forward
        2) backward
        3) gradient clipping
        4) optimizer step

        Returns:
            loss value, raw model outputs
        """
        self.model.train()
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss_criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()
        return loss.item(), outputs

    def train(self):
        """
        Run full training loop over epochs.
        Logs metrics, saves checkpoints, and performs early stopping.
        """
        for epoch in range(1, self.epochs + 1):
            print(f"\n=== Epoch {epoch} ===")
            self.current_epoch = epoch
            self.dice_metric.reset()
            epoch_loss = 0.0
            TP, FP, FN = {1: 0, 2: 0}, {1: 0, 2: 0}, {1: 0, 2: 0}

            # training batches
            for batch in tqdm(self.train_loader, desc="Training", dynamic_ncols=True):
                img, lbl = batch["image"].to(self.device), batch["label"].to(self.device)
                loss, outputs = self.train_step(img, lbl)
                self.global_step += 1
                # step OneCycleLR per batch
                if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    self.scheduler.step()
                epoch_loss += loss

                # compute per-batch Dice
                probs = torch.softmax(outputs.detach(), dim=1)
                preds = torch.argmax(probs, dim=1)
                onehot_pred = one_hot(preds.unsqueeze(1), num_classes=3)
                onehot_lbl = one_hot(lbl.long(), num_classes=3)
                self.dice_metric(onehot_pred, onehot_lbl)

                # accumulate precision/recall counts
                true = lbl.squeeze(1)
                for cls in [1, 2]:
                    TP[cls] += ((preds == cls) & (true == cls)).sum().item()
                    FP[cls] += ((preds == cls) & (true != cls)).sum().item()
                    FN[cls] += ((preds != cls) & (true == cls)).sum().item()

                self.train_history["loss"][self.global_step] = loss

            train_metrics = self._finalize_epoch(
                epoch_loss, TP, FP, FN, self.train_history, len(self.train_loader)
            )
            val_metrics   = self.validate()

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics["mean_dice"])
            elif not isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            self.lr_history[epoch] = lr
            print(f"Current LR: {lr:.6f}")
            log_metrics(epoch, train_metrics, val_metrics)
            save_history_log(self.train_history, self.validation_history, self.lr_history,
                             path=self.logs_dir / f"history_{epoch:03d}.json")
            save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                validation_history=self.validation_history,
                loss_criterion=self.loss_criterion,
                train_loader=self.train_loader,
                epoch=epoch,
                global_step=self.global_step,
                total_epochs=self.epochs,
                checkpoint_dir=self.checkpoint_dir
            )

            if should_early_stop(self):
                print("Early stopping triggered.")
                break

    def _finalize_epoch(self, epoch_loss, TP, FP, FN, history, loader_len):
        avg_loss = epoch_loss / loader_len
        # record loss in history for both train & val
        history["loss"][self.global_step] = avg_loss
        per_class, _ = self.dice_metric.aggregate()
        per_class = per_class.cpu().numpy()
        dice1, dice2 = np.nanmean(per_class, axis=0)[:2]
        mean_dice = float(np.nanmean([dice1, dice2]))

        prec1 = TP[1] / (TP[1] + FP[1] + 1e-6)
        rec1  = TP[1] / (TP[1] + FN[1] + 1e-6)
        prec2 = TP[2] / (TP[2] + FP[2] + 1e-6)
        rec2  = TP[2] / (TP[2] + FN[2] + 1e-6)

        history.setdefault("dice_class1", {})[self.global_step] = dice1
        history.setdefault("dice_class2", {})[self.global_step] = dice2
        history["dice"][self.global_step] = mean_dice

        return {
            "loss": avg_loss,
            "mean_dice": mean_dice,
            "dice_class1": dice1,
            "dice_class2": dice2,
            "precision_class1": prec1,
            "recall_class1": rec1,
            "precision_class2": prec2,
            "recall_class2": rec2,
        }

    def validate(self, save=True):
        self.model.eval()
        self.dice_metric.reset()
        total_loss = 0.0
        TP, FP, FN = {1:0,2:0}, {1:0,2:0}, {1:0,2:0}
        skipped = 0
        preds_to_save = []

        with torch.no_grad():
            for idx, batch in enumerate(self.val_loader):
                img, lbl = batch["image"].to(self.device), batch["label"].to(self.device)
                out = sliding_window_inference(
                    img, roi_size=(192,192,48), sw_batch_size=1,
                    predictor=self.model, overlap=0.5
                )
                loss = self.loss_criterion(out, lbl)
                total_loss += loss.item()

                probs = torch.softmax(out.detach(), dim=1)
                preds = torch.argmax(probs, dim=1)

                cleaned = []
                for b in range(preds.shape[0]):
                    arr = preds[b].cpu().numpy()
                    arr = remove_small_cc(arr, min_voxels=300)
                    cleaned.append(torch.from_numpy(arr))
                preds = torch.stack(cleaned, dim=0).to(self.device)

                onehot_pred = one_hot(preds.unsqueeze(1), num_classes=3)
                onehot_lbl  = one_hot(lbl.long(),   num_classes=3)
                self.dice_metric(onehot_pred, onehot_lbl)

                true = lbl.squeeze(1)
                if true.sum() == 0:
                    skipped += 1
                    continue
                for cls in [1,2]:
                    TP[cls] += ((preds==cls)&(true==cls)).sum().item()
                    FP[cls] += ((preds==cls)&(true!=cls)).sum().item()
                    FN[cls] += ((preds!=cls)&(true==cls)).sum().item()

                if idx < 3:
                    preds_to_save.append((img[0], lbl[0], probs[0], idx))

        print(f"[Summary] Skipped {skipped}/{len(self.val_loader)} batches with no lesion")
        metrics = self._finalize_epoch(
            total_loss, TP, FP, FN, self.validation_history, loader_len=len(self.val_loader)
        )

        if save and is_best_model(self.global_step, self.validation_history):
            for img, lbl, prob, i in preds_to_save:
                save_predictions(img, lbl, prob, i, self.predictions_dir)

        return metrics




