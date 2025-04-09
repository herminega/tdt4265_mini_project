import datetime
import pathlib
import yaml
from train import Trainer
from utils import save_model

if __name__ == "__main__":
    # Define experiment-specific folder structure.
    EXPERIMENT = "exp03_DynUNet"
    BASE_SAVE_PATH = pathlib.Path("results") / EXPERIMENT
    CHECKPOINT_DIR = BASE_SAVE_PATH / "checkpoints"
    MODEL_DIR = BASE_SAVE_PATH / "models"

    # Ensure directories exist:
    for directory in [BASE_SAVE_PATH, CHECKPOINT_DIR, MODEL_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    # Save experiment configuration:
    config = {
        "experiment": EXPERIMENT,
        "data_dir": "/cluster/projects/vc/data/mic/open/HNTS-MRG",
        "batch_size": 2,
        "learning_rate": 5e-4,
        "early_stop_count": 10,
        "epochs": 60,
        "model": "nnunet",
        "loss_parameters": {
            "smooth_dr": 0.001,
            "lambda_dice": 0.75,
            "lambda_ce": 0.35
        }
    }
    with open(BASE_SAVE_PATH / "config.yaml", 'w') as f:
        yaml.dump(config, f)

    # Initialize Trainer; note that we pass the checkpoint_dir.
    trainer = Trainer(
        data_dir="/cluster/projects/vc/data/mic/open/HNTS-MRG",
        batch_size=4,
        learning_rate=1e-3,
        early_stop_count=10,
        epochs=60,
        checkpoint_dir=CHECKPOINT_DIR,  # pass the checkpoints folder
        in_channels=1,
        out_channels=3,
    )

    # Train the model.
    trainer.train()
    model = trainer.model

    # Save the final model into the models folder.
    save_model(model, MODEL_DIR)
