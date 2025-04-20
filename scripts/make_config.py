"""
make_config.py

Utility to generate a baseline config.yaml template.
- Defines default experiment name, data paths, hyperparameters.
- Writes out results/<experiment>/config.yaml for downstream scripts.
"""
import yaml, pathlib

# 1) Experiment–specific tweak here
EXPERIMENT    = "exp24_nnunet"
DATA_DIR      = "/datasets/tdt4265/mic/open/HNTS-MRG"
BATCH_SIZE    = 3
LEARNING_RATE = 1e-3
EARLY_STOP    = 20
EPOCHS        = 200
MODEL         = "nnunet"
SCHEDULER     = "cosine"
L_DICE        = 0.5
L_CE          = 0.5

base = pathlib.Path("results") / EXPERIMENT
base.mkdir(parents=True, exist_ok=True)

config = {
  "experiment": EXPERIMENT,
  "data_dir": DATA_DIR,
  "batch_size": BATCH_SIZE,
  "learning_rate": LEARNING_RATE,
  "early_stop_count": EARLY_STOP,
  "epochs": EPOCHS,
  "model": MODEL,
  "scheduler_type": SCHEDULER,
  "loss_parameters": {
      "lambda_dice": L_DICE,
      "lambda_ce":   L_CE,
  }
}

with open(base/"config.yaml","w") as f:
    yaml.dump(config, f)

print(f"Wrote {base/'config.yaml'}")