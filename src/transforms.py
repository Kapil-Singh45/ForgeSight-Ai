from typing import Tuple, List, Optional
import torch
from torchvision import transforms
from PIL import Image


def get_image_transforms(
    image_size: Tuple[int, int] = (256, 256),
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> transforms.Compose:
    """
    Constructs deterministic preprocessing transforms for input images.

    Industrial anomaly detection requires consistent geometry and color calibration.
    Standard ImageNet normalization and bicubic interpolation are applied.
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_mask_transforms(
    image_size: Tuple[int, int] = (256, 256),
) -> transforms.Compose:
    """
    Constructs transforms for ground-truth defect masks.
    
    Uses nearest-neighbor interpolation to preserve sharp binary defect boundaries.
    """
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])


def denormalize_image(
    tensor: torch.Tensor,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> torch.Tensor:
    """
    Reverses ImageNet normalization to return a displayable [0, 1] tensor.
    
    Args:
        tensor: Tensor of shape (C, H, W) or (B, C, H, W)
    """
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    mean_t = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)
    std_t = torch.tensor(std, device=tensor.device).view(-1, 1, 1)

    if tensor.dim() == 4:
        mean_t = mean_t.unsqueeze(0)
        std_t = std_t.unsqueeze(0)

    denorm = tensor * std_t + mean_t
    return torch.clamp(denorm, 0.0, 1.0)
