# scripts/kfold_train.py
from src.training.ensemble import run_kfold

if __name__ == "__main__":
    EXPERIMENT = "exp24_nnunet"
    run_kfold(EXPERIMENT, n_folds=5, seed=0)
