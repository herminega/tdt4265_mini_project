# ensemble_train.py
import torch, pathlib
from sklearn.model_selection import KFold
from monai.data import CacheDataset, DataLoader, pad_list_data_collate
from dataloader import train_transforms
from train import Trainer
import yaml

# ——— Load experiment config ———
EXPERIMENT = "exp24_nnunet"   # ← change this to exp23_nnunet or exp25_xyz as needed
CONFIG_PATH = f"results/{EXPERIMENT}/config.yaml"

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)


DATA_DIR    = cfg["data_dir"]
BATCH_SIZE  = cfg["batch_size"]
LR          = cfg["learning_rate"]
EPOCHS      = cfg["epochs"]
EARLY_STOP  = cfg["early_stop_count"]
MODEL_TYPE  = cfg["model"]
SEED        = 0
N_FOLDS     = 5

# ——— Build a single cached dataset for all train cases ———
# replicate what get_mri_dataloader does internally:
data_list = []  # same logic as in dataloader.get_mri_dataloader
for pid in sorted(pathlib.Path(DATA_DIR, "train").iterdir()):
    preRT = pid / "preRT"
    img  = next(preRT.glob("*T2.nii.gz"), None)
    mask = next(preRT.glob("*mask.nii.gz"), None)
    if img and mask:
        data_list.append({"image": str(img), "label": str(mask)})

full_ds = CacheDataset(
    data=data_list,
    transform=train_transforms(),
    cache_rate=1.0,
    num_workers=4,
)

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(kf.split(full_ds)):
    print(f"\n=== Fold {fold+1}/{N_FOLDS} ===")

    # build subset loaders
    train_sub = torch.utils.data.Subset(full_ds, train_idx)
    val_sub   = torch.utils.data.Subset(full_ds, val_idx)

    train_loader = DataLoader(
        train_sub, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=pad_list_data_collate, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_sub, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=pad_list_data_collate, num_workers=4, pin_memory=True
    )

    # set up per‐fold output dir
    fold_dir = pathlib.Path("results") / EXPERIMENT / f"fold{fold}"
    ckpt_dir = fold_dir / "checkpoints"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # instantiate & train
    trainer = Trainer(
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=LR,
        early_stop_count=EARLY_STOP,
        epochs=EPOCHS,
        in_channels=1,
        out_channels=3,
        checkpoint_dir=ckpt_dir,
        scheduler_type=cfg.get("scheduler_type", "cosine"),
    )
    trainer.train()
    print(f"Fold {fold} done. Best ckpt in {ckpt_dir}/best.ckpt")

