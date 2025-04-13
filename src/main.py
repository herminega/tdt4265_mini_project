import datetime
import pathlib
import yaml
from train import Trainer
from utils import save_model

if __name__ == "__main__":
    # Define experiment-specific folder structure.
    EXPERIMENT = "exp15_nnunet"
    BASE_SAVE_PATH = pathlib.Path("results") / EXPERIMENT
    CHECKPOINT_DIR = BASE_SAVE_PATH / "checkpoints"
    MODEL_DIR = BASE_SAVE_PATH / "models"

    # Ensure directories exist:
    for directory in [BASE_SAVE_PATH, CHECKPOINT_DIR, MODEL_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    # Save experiment configuration:
    config = {
        "experiment": EXPERIMENT,
        "data_dir": "/datasets/tdt4265/mic/open/HNTS-MRG",
        "batch_size": 3,
        "learning_rate": 1e-3,
        "early_stop_count": 15,
        "epochs": 200,
        "model": "nnunet",
        "loss_parameters": {
            "lambda_dice": 0.7,
            "lambda_ce": 0.3
        }
    }
    with open(BASE_SAVE_PATH / "config.yaml", 'w') as f:
        yaml.dump(config, f)

    # Initialize Trainer; note that we pass the checkpoint_dir.
    # Path idun: /cluster/projects/vc/data/mic/open/HNTS-MRG
    # Path cybele: /datasets/tdt4265/mic/open/HNTS-MRG
    trainer = Trainer(
        data_dir="/datasets/tdt4265/mic/open/HNTS-MRG",
        batch_size=3,
        learning_rate=1e-3,
        early_stop_count=15,
        epochs=200,
        checkpoint_dir=CHECKPOINT_DIR,  # pass the checkpoints folder
        in_channels=1,
        out_channels=3,
    )

    # Train the model.
    trainer.train()
    model = trainer.model

    # Save the final model into the models folder.
    save_model(model, MODEL_DIR)
