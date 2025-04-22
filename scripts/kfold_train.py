# scripts/kfold_train.py
import os, sys
# Insert project root (one level up from scripts/) at front of sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
    
from src.training.ensemble import run_kfold

if __name__ == "__main__":
    EXPERIMENT = "exp25_nnunet"
    run_kfold(EXPERIMENT, n_folds=5, start_fold=3, seed=0)
