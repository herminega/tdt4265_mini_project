# scripts/paths.py
from pathlib import Path
from .config import ExperimentConfig

def resolve_paths(cfg: ExperimentConfig):
    base = Path("../results") / cfg.name
    return {
        "base":      base,
        "checkpts":  base / "checkpoints",
        "models":    base / "models",
        "logs":      base / "logs",
        "preds":     base / "predictions",
        "fold_dir":  lambda fold: base / f"fold{fold}" / "checkpoints",
        "config":    base / "config.yaml",
    }