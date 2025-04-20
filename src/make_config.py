import pathlib
import yaml

# === Experiment settings (hardcoded) ===
EXPERIMENT = "exp24_nnunet"    # Change this for each new experiment
DATA_DIR = "/datasets/tdt4265/mic/open/HNTS-MRG"
BATCH_SIZE = 3
LEARNING_RATE = 1e-3
EARLY_STOP_COUNT = 20
EPOCHS = 200
MODEL = "nnunet"             # Options: 'unet', 'dynunet', 'nnunet', 'segresnet'
SCHEDULER = "cosine"         # Options: 'plateau', 'cosine'
LAMBDA_DICE = 0.5
LAMBDA_CE = 0.5

# === Create config directory ===
base = pathlib.Path("results") / EXPERIMENT
base.mkdir(parents=True, exist_ok=True)

# === Build config dictionary ===
config = {
    "experiment": EXPERIMENT,
    "data_dir": DATA_DIR,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "early_stop_count": EARLY_STOP_COUNT,
    "epochs": EPOCHS,
    "model": MODEL,
    "scheduler_type": SCHEDULER,
    "loss_parameters": {
        "lambda_dice": LAMBDA_DICE,
        "lambda_ce": LAMBDA_CE,
    }
}

# === Write config.yaml ===
config_path = base / "config.yaml"
with open(config_path, "w") as f:
    yaml.dump(config, f)

print(f"Config written to {config_path}")

