import os
import random
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
import scipy.ndimage as ndi


def get_device() -> torch.device:
    """Returns CUDA device if available, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across python, numpy, and torch for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def overlay_heatmap_on_image(
    image_rgb: np.ndarray,
    anomaly_map: np.ndarray,
    alpha: float = 0.5,
    colormap_name: str = "jet",
) -> np.ndarray:
    """
    Overlays a continuous 2D anomaly heatmap on top of an RGB image using Matplotlib.

    Args:
        image_rgb: Numpy array of shape (H, W, 3), range [0, 1] or [0, 255].
        anomaly_map: 2D numpy array of shape (H, W).
        alpha: Blending weight for heatmap.
        colormap_name: Matplotlib colormap name ('jet', 'turbo', 'viridis', etc.).

    Returns:
        Blended RGB image of shape (H, W, 3) in range [0, 1].
    """
    if image_rgb.max() > 1.0:
        image_rgb = image_rgb / 255.0

    # Normalize anomaly map to [0, 1]
    norm_map = anomaly_map - anomaly_map.min()
    max_val = norm_map.max()
    if max_val > 0:
        norm_map = norm_map / max_val
    else:
        norm_map = np.zeros_like(norm_map)

    # Colorize heatmap with matplotlib
    cmap = plt.get_cmap(colormap_name.lower())
    heatmap_rgba = cmap(norm_map)
    heatmap_rgb = heatmap_rgba[:, :, :3]

    # Alpha blending
    blended = (1.0 - alpha) * image_rgb + alpha * heatmap_rgb
    return np.clip(blended, 0.0, 1.0)


def save_inspection_panel(
    image_rgb: np.ndarray,
    gt_mask: np.ndarray,
    anomaly_map: np.ndarray,
    save_path: str,
    title: str = "Defect Inspection",
    threshold: Optional[float] = None,
) -> None:
    """
    Exports a 4-panel diagnostic inspection visual:
    1. Input Image
    2. Ground Truth Mask
    3. Continuous Anomaly Heatmap
    4. Heatmap Overlay with Defect Contours
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # 1. Original RGB image
    axes[0].imshow(image_rgb)
    axes[0].set_title("Input Sample")
    axes[0].axis("off")

    # 2. Ground-truth binary mask
    axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    # 3. Continuous anomaly heatmap
    im3 = axes[2].imshow(anomaly_map, cmap="jet")
    axes[2].set_title("Predicted Anomaly Map")
    axes[2].axis("off")
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Anomaly Overlay
    overlay = overlay_heatmap_on_image(image_rgb, anomaly_map)
    axes[3].imshow(overlay)
    
    # If a threshold is given, highlight defect contours
    if threshold is not None:
        binary_pred = (anomaly_map >= threshold)
        if np.any(binary_pred):
            axes[3].contour(binary_pred, colors="red", linewidths=1.5)

    axes[3].set_title(f"Overlay {'(Defect Contour)' if threshold else ''}")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
