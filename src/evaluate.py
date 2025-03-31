import torch
import numpy as np
import argparse
from monai.metrics import DiceMetric
from model import get_model
from dataloader import get_mri_dataloader

def evaluate_model(model_path, data_dir):
    """ Evaluates a trained model on the validation set. """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained model
    model = get_model(model_type="dynunet", in_channels=1, out_channels=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load validation data
    _, val_loader = get_mri_dataloader(data_dir, "train", batch_size=1, validation_fraction=0.1)

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    all_dice_scores = []

    with torch.no_grad():
        for batch in val_loader:
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            outputs = model(inputs)
            softmax_outputs = torch.softmax(outputs, dim=1)
            dice_metric(softmax_outputs, labels)
            dice_score = dice_metric.aggregate().item()
            all_dice_scores.append(dice_score)

    print(f"Mean Dice Score: {np.mean(all_dice_scores):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset directory")
    args = parser.parse_args()

    evaluate_model(args.model_path, args.data_dir)
    
    # How to use: 
    # python evaluate.py --model_path results/unet3d_mri.pth --data_dir /path/to/data


        