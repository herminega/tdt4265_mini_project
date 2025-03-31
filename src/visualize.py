import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_nifti(file_path):
    """ Load a NIfTI image as a NumPy array """
    return nib.load(file_path).get_fdata()

def visualize_saved_predictions(image_path, label_path, prediction_path, slice_idx=None):
    """ Load and visualize saved predictions """
    image = load_nifti(image_path)
    label = load_nifti(label_path)
    prediction = load_nifti(prediction_path)

    # Automatically choose the middle slice if not provided
    if slice_idx is None:
        slice_idx = image.shape[2] // 2  # Middle slice in depth

    # Plot images side by side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image[:, :, slice_idx], cmap="gray")
    axes[0].set_title("MRI Scan")

    axes[1].imshow(image[:, :, slice_idx], cmap="gray")
    axes[1].imshow(label[:, :, slice_idx], cmap="jet", alpha=0.5)
    axes[1].set_title("Ground Truth")

    axes[2].imshow(image[:, :, slice_idx], cmap="gray")
    axes[2].imshow(prediction[:, :, slice_idx], cmap="jet", alpha=0.5)
    axes[2].set_title("Model Prediction")

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True, help="Path to MRI scan")
    parser.add_argument("--label_path", type=str, required=True, help="Path to ground truth label")
    parser.add_argument("--prediction_path", type=str, required=True, help="Path to model prediction")
    parser.add_argument("--slice_idx", type=int, default=None, help="Slice index to visualize")
    args = parser.parse_args()

    visualize_saved_predictions(args.image_path, args.label_path, args.prediction_path, args.slice_idx)

# How to use:
# python src/visualize.py --image_path results/predictions/image_1.nii.gz --label_path results/predictions/label_1.nii.gz --prediction_path results/predictions/prediction_1.nii.gz
