import datetime
from train import Trainer
from utils import save_model

if __name__ == "__main__":
    # Set dataset directory
    data_dir = "/cluster/projects/vc/data/mic/open/HNTS-MRG"

    # Initialize Trainer
    trainer = Trainer(
        data_dir=data_dir,
        batch_size=4,
        learning_rate=1e-3,
        early_stop_count=5,
        epochs=50,
        in_channels=1,
        out_channels=3,
    )

    # Train the model
    trainer.train()
    model = trainer.model

    # Save the final model
    model_name = f"dynunet_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}.pth"
    save_model(model, f"results/models/{model_name}")