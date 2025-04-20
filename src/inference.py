# inference.py
import pathlib
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils import set_global_seed
set_global_seed(0)
from model import get_model
from dataloader import get_mri_dataloader
from utils import save_nifti, save_predictions
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.networks.utils import one_hot
from monai.inferers import sliding_window_inference


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, device: torch.device):
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state['model_state'])
    return model

def run_inference(
    data_dir: str,
    checkpoint: str,
    output_dir: str,
    batch_size: int = 1,
    model_type: str = 'nnunet',
    device: torch.device = None,
    save_labels: bool = True,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1) build model + load weights
    model = get_model(model_type, in_channels=1, out_channels=3, pretrained=False).to(device)
    model = load_checkpoint(checkpoint, model, device)
    model.eval()
    
    # 2) build dataloader
    _, loader = get_mri_dataloader(data_dir, subset='test', batch_size=batch_size)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 3) inference loop
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            imgs = batch['image'].to(device)
            outs = sliding_window_inference(
                inputs=imgs,
                roi_size=(192,192,48),
                sw_batch_size=1,          # how many windows per forward‐pass
                predictor=model,          # your nn.Module
                overlap=0.5,              # amount of overlap between windows
            )
            probs = torch.softmax(outs, dim=1)
            for b in range(imgs.size(0)):
                img, prob = imgs[b], probs[b]
                gid = idx * loader.batch_size + b
                if 'label' in batch and save_labels:
                    lbl = batch['label'][b].to(device)
                    save_predictions(img, lbl, prob, gid, output_dir)
                else:
                    save_nifti(img, output_dir / f"image_{gid}.nii.gz")
                    save_nifti(torch.argmax(prob, dim=0), output_dir / f"pred_{gid}.nii.gz")
    print(f"Inference done; outputs in {output_dir}")
    return model, loader



def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_criterion,
    device: torch.device
) -> dict:
    """
    Compute on the entire dataloader:
      - average loss (Dice+CE)
      - per-class Dice (classes 1 & 2)
      - mean Dice across classes
      - precision & recall per class
    """
    model.eval()
    dice_metric = DiceMetric(include_background=False, reduction="none", get_not_nans=True)

    total_loss = 0.0
    total_TP, total_FP, total_FN = {1:0, 2:0}, {1:0, 2:0}, {1:0, 2:0}
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            imgs  = batch["image"].to(device)
            lbls  = batch["label"].to(device)
            outs  = model(imgs)
            # 1) loss
            loss = loss_criterion(outs, lbls)
            total_loss += loss.item()
            n_batches  += 1

            # 2) dice accum
            probs = torch.softmax(outs, dim=1)
            preds = torch.argmax(probs, dim=1)
            oh_pred  = one_hot(preds.unsqueeze(1), num_classes=outs.shape[1])
            oh_label = one_hot(lbls.long(),     num_classes=outs.shape[1])
            dice_metric(oh_pred, oh_label)

            # 3) precision/recall accum
            true = lbls.squeeze(1)
            for cls in [1, 2]:
                TP = ((preds == cls) & (true == cls)).sum().item()
                FP = ((preds == cls) & (true != cls)).sum().item()
                FN = ((preds != cls) & (true == cls)).sum().item()
                total_TP[cls] += TP
                total_FP[cls] += FP
                total_FN[cls] += FN

    # finalize
    avg_loss = total_loss / n_batches
    per_class, _ = dice_metric.aggregate()
    per_class = per_class.cpu().numpy()

    dice_metric.reset()
    dice_class1, dice_class2 = np.nanmean(per_class, axis=0)[:2]
    mean_dice = np.nanmean([dice_class1, dice_class2])

    prec1 = total_TP[1] / (total_TP[1] + total_FP[1] + 1e-6)
    rec1  = total_TP[1] / (total_TP[1] + total_FN[1] + 1e-6)
    prec2 = total_TP[2] / (total_TP[2] + total_FP[2] + 1e-6)
    rec2  = total_TP[2] / (total_TP[2] + total_FN[2] + 1e-6)

    return {
        "avg_loss":    avg_loss,
        "mean_dice":   mean_dice,
        "dice_class1": dice_class1,
        "dice_class2": dice_class2,
        "prec_class1": prec1,
        "recall_class1": rec1,
        "prec_class2": prec2,
        "recall_class2": rec2,
    }
    

import pathlib
import numpy as np
import nibabel as nib
import torch
from monai.metrics import DiceMetric

def load_nifti(path: pathlib.Path) -> np.ndarray:
    """Load a NIfTI file and return its data array."""
    return nib.load(str(path)).get_fdata().astype(np.int32)


def evaluate_predictions(
    output_dir: str,
):
    """
    Compute mean Dice over saved ensembled predictions and their ground-truth labels
    in a flat output directory containing label_*.nii.gz and prediction_*.nii.gz.

    Args:
        output_dir: directory where ensemble_inference.py saved image_*, label_*, prediction_* NIfTIs
    """
    output_dir = pathlib.Path(output_dir)
    label_files = sorted(output_dir.glob("label_*.nii.gz"))
    pred_files  = sorted(output_dir.glob("prediction_*.nii.gz"))
    assert len(label_files) == len(pred_files), \
        f"Mismatch: {len(label_files)} labels vs {len(pred_files)} predictions"

    # MONAI DiceMetric expects Tensor inputs of shape (B, C, H, W, D)
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    dice_values = []

    for lbl_path, pred_path in zip(label_files, pred_files):
        gt = load_nifti(lbl_path)
        pred = load_nifti(pred_path)
        # one-hot encode for classes 1 and 2
        gt_oh   = np.stack([(gt == 1), (gt == 2)], axis=0)[None]  # shape (1,2,H,W,D)
        pred_oh = np.stack([(pred == 1), (pred == 2)], axis=0)[None]

        # convert to Tensors
        gt_t   = torch.from_numpy(gt_oh.astype(np.float32))
        pred_t = torch.from_numpy(pred_oh.astype(np.float32))

        # compute Dice
        dice = dice_metric(pred_t, gt_t).item()
        dice_values.append(dice)

    mean_dice = float(np.mean(dice_values))
    print(f"Evaluated {len(dice_values)} cases → Mean Dice (GTVp & GTVn): {mean_dice:.4f}")
    return dice_values



