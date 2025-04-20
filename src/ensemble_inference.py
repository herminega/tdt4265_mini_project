import pathlib
import glob
import torch
import numpy as np
from monai.inferers import sliding_window_inference
from model import get_model
from dataloader import get_mri_dataloader
from utils import load_checkpoint, remove_small_cc, save_predictions, save_nifti


def ensemble_inference(
    data_dir: str,
    output_dir: str,
    checkpoint_glob: str,
    model_type: str = 'nnunet',
    batch_size: int = 1,
    roi_size=(192,192,48),
    overlap: float = 0.5,
    min_cc_size: int = 300,
    device: torch.device = None
):
    """
    Perform ensemble inference using all checkpoints matching checkpoint_glob.
    Saves predictions using save_predictions to keep consistency with single-model inference.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load all checkpoints
    ckpt_paths = sorted(glob.glob(checkpoint_glob))
    models = []
    for ckpt in ckpt_paths:
        model = get_model(model_type, in_channels=1, out_channels=3, pretrained=False).to(device)
        model = load_checkpoint(ckpt, model, device)
        model.eval()
        models.append(model)
    print(f"Loaded {len(models)} models for ensembling.")

    # 2) Prepare test loader
    _, test_loader = get_mri_dataloader(data_dir, subset='test', batch_size=batch_size)

    # 3) Ensure output dir exists
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4) Inference and ensemble
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            images = batch['image'].to(device)
            labels = batch.get('label', None)
            if labels is not None:
                labels = labels.to(device)

            # accumulate softmax maps
            prob_sum = None
            for model in models:
                out = sliding_window_inference(
                    inputs=images,
                    roi_size=roi_size,
                    sw_batch_size=1,
                    predictor=model,
                    overlap=overlap,
                )
                probs = torch.softmax(out, dim=1).cpu().numpy()
                prob_sum = probs if prob_sum is None else prob_sum + probs

            avg_probs = prob_sum / len(models)
            pred_labels = np.argmax(avg_probs, axis=1)

            # save per-sample
            for b in range(pred_labels.shape[0]):
                gid = idx * batch_size + b
                pred = pred_labels[b]
                pred = remove_small_cc(pred, min_voxels=min_cc_size)

                if labels is not None:
                    # use save_predictions to save image, label, pred
                    save_predictions(images[b], labels[b], avg_probs[b], gid, output_dir)
                else:
                    # fallback: save NIfTI
                    pred_tensor = torch.from_numpy(pred).unsqueeze(0)
                    save_nifti(pred_tensor, str(output_dir / f"pred_{gid}.nii.gz"))
    print(f"Ensembled predictions saved to {output_dir}")


# User-specified parameters 
DATA_DIR       = "/datasets/tdt4265/mic/open/HNTS-MRG"
OUTPUT_DIR     = "results/ensemble_predictions"
CHECKPOINT_GLOB = "results/fold*/checkpoints/best.ckpt"

# Run ensemble inference
ensemble_inference(
    data_dir=DATA_DIR,
    output_dir=OUTPUT_DIR,
    checkpoint_glob=CHECKPOINT_GLOB,
    model_type='nnunet',
    batch_size=3
)
