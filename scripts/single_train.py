"""
single_train.py

CLI for a single end‑to‑end training run.
- Loads config via src/config.py
- Instantiates Trainer on the full training set
- Runs Trainer.train() once
- Saves final model
"""

import os, sys
# Insert project root (one level up from scripts/) at front of sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.config import load_config
from scripts.paths  import resolve_paths
from src.utils.metrics import set_global_seed
from src.utils.file_io import save_model
from src.training.trainer import Trainer

if __name__=="__main__":
    set_global_seed(0)
    EXP = "exp24_nnunet"          # pick experiment
    cfg = load_config(EXP)
    P   = resolve_paths(cfg)

    # ensure folders exist
    for d in ("base","checkpts","models","logs","preds"):
        P[d].mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        early_stop_count=cfg.early_stop_count,
        epochs=cfg.epochs,
        checkpoint_dir=P["checkpts"],
        in_channels=1,
        out_channels=3,
        scheduler_type=cfg.scheduler_type,
    )
    trainer.train()
    save_model(trainer.model, P["models"])
