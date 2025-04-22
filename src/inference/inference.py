"""
inference.py

Single‑model and ensemble inference + evaluation scripts.
- run_inference(): slide‑window predict on test set, save NIfTIs or overlaid plots.
- evaluate(): compute loss, Dice, precision, recall for one or multiple models.
- ensemble_inference(): average softmax outputs across checkpoints, apply CC cleanup.
"""

__all__ = [
    "run_inference",
    "evaluate",
    "ensemble_inference",
]

from pathlib import Path
import glob
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple, Optional, Union, List

from monai.metrics import DiceMetric
from monai.losses import DiceCELoss
from monai.networks.utils import one_hot
from monai.inferers import sliding_window_inference

from src.utils.metrics import set_global_seed, remove_small_cc
from src.utils.file_io import save_nifti, save_predictions, load_checkpoint
from src.dataloader.dataloader import get_test_dataloader
from src.model.model import get_model

# Fix seeds for reproducibility
default_seed = 0
set_global_seed(default_seed)


def run_inference(
    data_dir: str,
    checkpoint: str,
    output_dir: str,
    batch_size: int = 1,
    model_type: str = 'nnunet',
    device: Optional[torch.device] = None,
    save_labels: bool = True,
) -> Tuple[torch.nn.Module, DataLoader]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_type, in_channels=1, out_channels=3, pretrained=False).to(device)
    model = load_checkpoint(checkpoint, model, device)
    model.eval()
    _, loader = get_test_dataloader(data_dir, batch_size=batch_size)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, desc="Inference")):
            imgs = batch['image'].to(device)
            outputs = sliding_window_inference(
                inputs=imgs,
                roi_size=(192,192,48),
                sw_batch_size=1,
                predictor=model,
                overlap=0.5,
            )
            probs = torch.softmax(outputs, dim=1)
            for b in range(imgs.size(0)):
                gid = idx * batch_size + b
                if 'label' in batch and save_labels:
                    save_predictions(imgs[b], batch['label'][b].to(device), probs[b], gid, out_path)
                else:
                    save_nifti(imgs[b], out_path/f"image_{gid}.nii.gz")
                    save_nifti(torch.argmax(probs[b], dim=0), out_path/f"pred_{gid}.nii.gz")
    print(f"Inference done; outputs in {out_path}")
    return model, loader


def evaluate(
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    dataloader: DataLoader,
    loss_criterion: DiceCELoss,
    device: Optional[torch.device] = None
) -> dict:
    """
    Evaluate a single model or list of models (ensemble) on a loader.
    Returns dict of avg_loss, per-class and mean Dice, precision, recall.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_ensemble = isinstance(model, list)
    if is_ensemble:
        for m in model:
            m.eval()
    else:
        model.eval()

    dice_metric = DiceMetric(include_background=False, reduction="none", get_not_nans=True)
    total_loss, batches = 0.0, 0
    TP, FP, FN = {1:0,2:0}, {1:0,2:0}, {1:0,2:0}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation"):
            imgs = batch['image'].to(device)
            lbls = batch['label'].to(device)

            # get outputs: average raw logits for ensemble, or single-model logits
            if is_ensemble:
                sum_logits = None
                for m in model:
                    out = sliding_window_inference(
                        inputs=imgs,
                        roi_size=(192,192,48),
                        sw_batch_size=1,
                        predictor=m,
                        overlap=0.5,
                    )  # raw logits
                    sum_logits = out if sum_logits is None else sum_logits + out
                outputs = sum_logits / len(model)
            else:
                outputs = model(imgs)

            # compute loss on logits
            loss = loss_criterion(outputs, lbls)
            total_loss += loss.item(); batches += 1

            # compute metrics on probabilities
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            oh_pred = one_hot(preds.unsqueeze(1), num_classes=outputs.shape[1])
            oh_lbl  = one_hot(lbls.long(),     num_classes=outputs.shape[1])
            dice_metric(oh_pred, oh_lbl)
            true = lbls.squeeze(1)
            for cls in [1,2]:
                TP[cls] += ((preds==cls)&(true==cls)).sum().item()
                FP[cls] += ((preds==cls)&(true!=cls)).sum().item()
                FN[cls] += ((preds!=cls)&(true==cls)).sum().item()

    avg_loss = total_loss / batches
    per_class, _ = dice_metric.aggregate(); dice_metric.reset()
    dice_vals = per_class.cpu().numpy().flatten()[:2]
    mean_dice = float(np.nanmean(dice_vals))
    prec1 = TP[1]/(TP[1]+FP[1]+1e-6); rec1 = TP[1]/(TP[1]+FN[1]+1e-6)
    prec2 = TP[2]/(TP[2]+FP[2]+1e-6); rec2 = TP[2]/(TP[2]+FN[2]+1e-6)

    return {
        'avg_loss': avg_loss,
        'mean_dice': mean_dice,
        'dice_class1': dice_vals[0],
        'dice_class2': dice_vals[1],
        'prec_class1': prec1,
        'recall_class1': rec1,
        'prec_class2': prec2,
        'recall_class2': rec2,
    }


def ensemble_inference(
    data_dir: str,
    output_dir: str,
    checkpoint_glob: str,
    model_type: str = 'nnunet',
    batch_size: int = 1,
    device: Optional[torch.device] = None,
    roi_size: Tuple[int,int,int] = (192,192,48),
    overlap: float = 0.5,
    min_cc_size: int = 300,
) -> Tuple[List[torch.nn.Module], DataLoader]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_paths = sorted(glob.glob(checkpoint_glob))
    assert ckpt_paths, f"No checkpoints match {checkpoint_glob}"

    models: List[torch.nn.Module] = []
    for ckpt in ckpt_paths:
        model = get_model(model_type, in_channels=1, out_channels=3, pretrained=False).to(device)
        model = load_checkpoint(ckpt, model, device)
        model.eval(); models.append(model)

    _, loader = get_test_dataloader(data_dir, batch_size=batch_size)
    out_path = Path(output_dir); out_path.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(loader, desc="Ensemble Inference")):
            images = batch['image'].to(device)
            labels = batch.get('label', None)
            prob_sum = None
            for m in models:
                out = sliding_window_inference(
                    inputs=images,
                    roi_size=roi_size,
                    sw_batch_size=1,
                    predictor=m,
                    overlap=overlap,
                )
                p = torch.softmax(out, dim=1)
                prob_sum = p if prob_sum is None else prob_sum + p

            avg_probs = prob_sum / len(models)
            preds = torch.argmax(avg_probs, dim=1)

            for b in range(preds.shape[0]):
                gid = idx * batch_size + b
                pred_np = remove_small_cc(preds[b].cpu().numpy(), min_voxels=min_cc_size)
                pred_t = torch.from_numpy(pred_np).unsqueeze(0)
                if labels is not None:
                    save_predictions(images[b], labels[b], avg_probs[b].cpu(), gid, out_path)
                else:
                    save_nifti(pred_t, out_path/f"pred_{gid}.nii.gz")
    print(f"Ensembled predictions saved to {out_path}")
    return models, loader


