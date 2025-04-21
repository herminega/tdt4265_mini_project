"""
visualization.py

All plotting functions for diagnostics and results.
- visualize_slices(): show one axial slice with ground truth & prediction overlays.
- show_multiple_slices(): grid of several image+mask slices.
- plot_intensity_histogram(): distribution of voxel intensities.
- plot_metrics(): training/validation curves over time.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import List, Tuple, Dict
from src.utils.file_io import load_nifti


def visualize_slices(image_path, label_path, prediction_path, slice_idx=None):
    image = load_nifti(image_path)
    label = load_nifti(label_path)
    prediction = load_nifti(prediction_path)

    if slice_idx is None:
        slice_idx = image.shape[-1] // 2

    if label.ndim == 4:
        label = np.argmax(label, axis=0)
    if prediction.ndim == 4:
        prediction = np.argmax(prediction, axis=0)

    # Colors
    cmap = ListedColormap(["black", "green", "red"])
    class_labels = ["Background", "GTVp", "GTVn"]
    
    def class_distribution(arr):
        counts = np.bincount(arr.flatten().astype(int), minlength=3)
        return tuple(int(c) for c in counts)

    print("Label class distribution:", class_distribution(label))
    print("Prediction class distribution:", class_distribution(prediction))


    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, data, title in zip(
        axes,
        [image[:, :, slice_idx], label[:, :, slice_idx], prediction[:, :, slice_idx]],
        ["MRI", "Ground Truth", "Prediction"]
    ):
        ax.imshow(image[:, :, slice_idx], cmap="gray")
        if title != "MRI":
            ax.imshow(data, cmap=cmap, alpha=0.5, vmin=0, vmax=2)
        ax.set_title(title)
        ax.axis("off")

    plt.suptitle(f"Slice {slice_idx}")
    plt.tight_layout()
    plt.show()


def plot_intensity_histogram(
    image: np.ndarray,
    bins: int = 100,
    title: str = "Intensity Distribution",
) -> None:
    """
    Plot a histogram of voxel intensities for a 3D image array.
    """
    plt.figure()
    plt.hist(image.flatten(), bins=bins)
    plt.xlabel("Intensity Value")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.show()

def plot_metrics(
    history: Dict[str, Dict[str, Dict[str, float]]],
    metrics: List[Tuple[str, str, str, str]],
    figsize: Tuple[int, int] = (8, 5)
) -> None:
    """
    Plot training and validation metrics over time.

    Args:
        history: Nested dict, e.g. history['train']['loss'] = {'0':0.5, '1':0.4, ...}
        metrics: List of (stage, metric_name, label, color)
    """
    plt.figure(figsize=figsize)
    for stage, name, label, color in metrics:
        series = history[stage][name]
        steps = sorted(map(int, series.keys()))
        values = [series[str(s)] for s in steps]
        plt.plot(steps, values, label=label, color=color)

    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def show_multiple_slices(
    image: np.ndarray,
    label: np.ndarray,
    slice_indices: List[int] = None,
    title_prefix: str = "",
    cmap_label: ListedColormap = None
) -> None:
    """
    Display multiple axial slices of a 3D image alongside its segmentation mask.

    Args:
        image (np.ndarray): 3D image array shaped (H, W, D) or (C, H, W, D).
        label (np.ndarray): 3D mask array shaped (H, W, D) or (C, H, W, D).
        slice_indices (List[int], optional): Specific slice indices to plot. Defaults to five evenly spaced.
        title_prefix (str): Prefix for subplot titles.
        cmap_label (ListedColormap): Colormap for the label overlay.
    """
    # If channel-first, squeeze to (H,W,D)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if label.ndim == 4 and label.shape[0] == 1:
        label = label[0]

    D = image.shape[-1]
    if slice_indices is None:
        slice_indices = list(np.linspace(0, D - 1, 5, dtype=int))

    if cmap_label is None:
        cmap_label = ListedColormap(["black", "green", "red"])

    n = len(slice_indices)
    fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n))
    axes = np.atleast_2d(axes)

    for i, idx in enumerate(slice_indices):
        axes[i, 0].imshow(image[:, :, idx], cmap="gray")
        axes[i, 0].set_title(f"{title_prefix}Image {idx}")
        axes[i, 0].axis("off")

        im = axes[i, 1].imshow(image[:, :, idx], cmap="gray")
        axes[i, 1].imshow(label[:, :, idx], cmap=cmap_label, vmin=0, vmax=2, alpha=0.5)
        axes[i, 1].set_title(f"{title_prefix}Label {idx}")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()