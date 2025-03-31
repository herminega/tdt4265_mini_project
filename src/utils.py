import itertools
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
import torch
import os
from model import get_model  # Import model
from dataloader import get_mri_dataloader  # Import MONAI dataloader
from tqdm import tqdm, trange
import time
import copy
import gc

# Define a search grid
bg_weights = [0.01, 0.1, 1.0]
gtvp_weights = [0.5, 1.0, 1.25]
gtvn_weights = [1.0, 2.0, 2.5]

# Create combinations
weight_combinations = list(itertools.product(bg_weights, gtvp_weights, gtvn_weights))


weight_combinations = [
    (0.1, 1.0, 1.0),
    (0.01, 1.0, 2.0),
    (0.1, 1.0, 2.5),
    (0.1, 1.25, 2.0),
    (1.0, 1.0, 1.0),  # baseline
]

def grid_search_class_weights(train_loader, val_loader, device, weight_combinations=weight_combinations):
    

    results = []
    base_model = get_model(model_type="swinunetr", in_channels=1, out_channels=3, pretrained=True).to(device)


    bar = tqdm(total=len(weight_combinations), desc="🔍 Grid Search")

    for idx, (w_bg, w_gtvp, w_gtvn) in enumerate(weight_combinations):
        print(f"\n🔍 Grid Search [{idx+1}/{len(weight_combinations)}] - Weights: BG={w_bg}, GTVp={w_gtvp}, GTVn={w_gtvn}")

        # Define the loss function
        class_weights = torch.tensor([w_bg, w_gtvp, w_gtvn]).to(device)
        loss_fn = DiceLoss(softmax=True, weight=class_weights)

        # Fresh model
        start = time.time()
        model = copy.deepcopy(base_model)
        print("Model cloned in", time.time() - start, "seconds")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        best_val_dice = 0.0

        for epoch in tqdm(range(1, 2), desc=f"📈 Epochs for combo {idx+1}/{len(weight_combinations)}", leave=False):
            model.train()
            total_loss = 0.0

            for batch in train_loader:
                inputs = batch["image"].to(device)
                labels = batch["label"].to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Validation
            model.eval()
            dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
            with torch.no_grad():
                for val_batch in val_loader:
                    val_inputs = val_batch["image"].to(device)
                    val_labels = val_batch["label"].to(device)
                    val_outputs = model(val_inputs)
                    dice_metric(val_outputs, val_labels)

            val_dice = dice_metric.aggregate().item()
            best_val_dice = max(best_val_dice, val_dice)
            print(f"Epoch {epoch}: Val Dice = {val_dice:.4f}")

        # Save result
        results.append({
            "weights": (w_bg, w_gtvp, w_gtvn),
            "val_dice": best_val_dice
        })

        torch.cuda.empty_cache()
        gc.collect()
        del model

        bar.update(1)  # ✅ Move this here — only update when combo is fully done

    bar.close()

    sorted_results = sorted(results, key=lambda x: x["val_dice"], reverse=True)
    best_weights = sorted_results[0]["weights"]
    
    print("\n✅ Best weights found:")
    print(f"   BG={best_weights[0]}, GTVp={best_weights[1]}, GTVn={best_weights[2]}")
    
    return best_weights


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    data_dir = data_dir = "/cluster/projects/vc/data/mic/open/HNTS-MRG"
    # Load data
    train_loader, val_loader =  get_mri_dataloader(data_dir, "train", batch_size=2, validation_fraction=0.1)

    # Perform grid search
    best_weights = grid_search_class_weights(train_loader, val_loader, device, weight_combinations)
    print(f"Best weights: {best_weights}")
