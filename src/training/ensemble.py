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
from monai.data import Dataset, CacheDataset, DataLoader, pad_list_data_collate
from scripts.config import load_config
from src.dataloader.dataloader import train_transforms, base_transforms, val_transforms
from src.training.trainer import Trainer

def run_kfold(experiment: str, n_folds: int = 5, seed: int = 0):
    cfg  = load_config(experiment)
    base = pathlib.Path("results") / experiment

    # 1) build list of file‐dicts
    data_list = []
    for pid in sorted((pathlib.Path(cfg.data_dir)/"train").iterdir()):
        preRT = pid/"preRT"
        img   = next(preRT.glob("*T2.nii.gz"), None)
        msk   = next(preRT.glob("*mask.nii.gz"), None)
        if img and msk:
            data_list.append({"image": str(img), "label": str(msk)})

    # 2) cache only the base transforms (no random crops yet)
    base_ds = CacheDataset(
        data=data_list,
        transform=base_transforms(),
        cache_rate=1.0,
        num_workers=4,
    )

    # 3) get K‐fold splits
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(kf.split(base_ds)):
        print(f"\n=== Fold {fold+1}/{n_folds} ===")
        ckpt_dir = base/f"fold{fold}"/"checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # 4) build lightweight train/val Datasets *on top* of base_ds
        train_ds = Dataset(
            data=[ base_ds[i] for i in tr_idx ],
            transform=train_transforms()
        )
        val_ds   = Dataset(
            data=[ base_ds[i] for i in va_idx ],
            transform=val_transforms()
        )

        # 5) DataLoaders for each fold
        train_loader = DataLoader(
            train_ds, batch_size=2, shuffle=True,
            collate_fn=pad_list_data_collate, num_workers=4, pin_memory=True
        )
        val_loader   = DataLoader(
            val_ds,   batch_size=2, shuffle=False,
            collate_fn=pad_list_data_collate, num_workers=4, pin_memory=True
        )

        # 6) hand off to your Trainer
        trainer = Trainer(
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=cfg.learning_rate,
            early_stop_count=cfg.early_stop_count,
            epochs=cfg.epochs,
            checkpoint_dir=ckpt_dir,
            in_channels=1,
            out_channels=3,
            scheduler_type=cfg.scheduler_type,
        )
        trainer.train()
        print(f"best ckpt: {ckpt_dir/'best.ckpt'}")

        del trainer
        torch.cuda.empty_cache()

