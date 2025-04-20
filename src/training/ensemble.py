"""
ensemble.py

K‑fold and ensemble training/inference utilities.
- Loads full training set into a cached dataset.
- Performs stratified K‑fold splits, builds per‑fold DataLoaders.
- Wraps Trainer to run folds sequentially, storing per‑fold checkpoints.
- (Optionally extendable to aggregate fold results or build model ensembles.)
"""

import pathlib
import torch
from sklearn.model_selection import KFold
from monai.data import CacheDataset, DataLoader, pad_list_data_collate
from scripts.config import load_config
from dataloader.dataloader import train_transforms
from training.trainer import Trainer

def run_kfold(experiment: str, n_folds: int = 5, seed: int = 0):
    cfg = load_config(experiment)
    base = pathlib.Path("results") / experiment

    # build full cache‐dataset
    data_list = []
    for pid in sorted((cfg["data_dir"]/"train").iterdir()):
        preRT = pid/"preRT"
        img  = next(preRT.glob("*T2.nii.gz"), None)
        mask = next(preRT.glob("*mask.nii.gz"), None)
        if img and mask:
            data_list.append({"image":str(img), "label":str(mask)})

    full_ds = CacheDataset(data=data_list,
                           transform=train_transforms(),
                           cache_rate=1.0,
                           num_workers=4)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(full_ds)):
        print(f"\n=== Fold {fold+1}/{n_folds} ===")
        fold_dir = base/f"fold{fold}"
        ckpt_dir = fold_dir/"checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        train_loader = DataLoader(
            torch.utils.data.Subset(full_ds, tr_idx),
            batch_size=cfg["batch_size"], shuffle=True,
            collate_fn=pad_list_data_collate,
            num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            torch.utils.data.Subset(full_ds, va_idx),
            batch_size=cfg["batch_size"], shuffle=False,
            collate_fn=pad_list_data_collate,
            num_workers=4, pin_memory=True
        )

        trainer = Trainer(
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=cfg["learning_rate"],
            early_stop_count=cfg["early_stop_count"],
            epochs=cfg["epochs"],
            checkpoint_dir=ckpt_dir,
            in_channels=1,
            out_channels=3,
            scheduler_type=cfg.get("scheduler_type", "cosine"),
        )
        trainer.train()
        print(f"best ckpt: {ckpt_dir/'best.ckpt'}")

