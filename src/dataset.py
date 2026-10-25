import os
from pathlib import Path
from typing import Tuple, Optional, Callable, Dict, Any, List
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from src.transforms import get_image_transforms, get_mask_transforms


class MVTecDataset(Dataset):
    """
    PyTorch Dataset for the MVTec Anomaly Detection (MVTec AD) benchmark.

    Directory structure expected:
        <root_dir>/<category>/
            train/good/*.png
            test/<defect_type>/*.png
            ground_truth/<defect_type>/*_mask.png
    """

    def __init__(
        self,
        root_dir: str,
        category: str,
        split: str = "train",
        image_size: Tuple[int, int] = (256, 256),
        transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
    ):
        """
        Args:
            root_dir: Root path containing category directories.
            category: MVTec category (e.g., 'bottle', 'carpet', 'cable').
            split: 'train' or 'test'.
            image_size: Target (height, width) tuple.
            transform: Custom image transform (if None, default is used).
            mask_transform: Custom mask transform (if None, default is used).
        """
        self.root_dir = Path(root_dir)
        self.category = category
        self.split = split
        self.image_size = image_size

        self.category_dir = self.root_dir / self.category
        if not self.category_dir.exists():
            raise FileNotFoundError(
                f"Category directory not found: '{self.category_dir}'. "
                f"Please ensure MVTec AD '{category}' is downloaded and extracted."
            )

        self.transform = transform or get_image_transforms(image_size=image_size)
        self.mask_transform = mask_transform or get_mask_transforms(image_size=image_size)

        self.samples = self._load_samples()

    def _load_samples(self) -> List[Dict[str, Any]]:
        """Scans the directory structure and pairs images with labels and ground-truth masks."""
        split_dir = self.category_dir / self.split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        samples = []
        defect_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])

        for defect_dir in defect_dirs:
            defect_type = defect_dir.name
            is_anomaly = 0 if defect_type == "good" else 1

            for img_path in sorted(defect_dir.glob("*.png")):
                # Determine ground-truth mask path
                if is_anomaly == 0:
                    mask_path = None
                else:
                    mask_path = (
                        self.category_dir
                        / "ground_truth"
                        / defect_type
                        / f"{img_path.stem}_mask.png"
                    )
                    if not mask_path.exists():
                        # Try alternative naming without _mask suffix if present
                        alt_mask_path = self.category_dir / "ground_truth" / defect_type / img_path.name
                        if alt_mask_path.exists():
                            mask_path = alt_mask_path
                        else:
                            raise FileNotFoundError(f"Expected mask not found for {img_path}: {mask_path}")

                samples.append({
                    "image_path": str(img_path),
                    "mask_path": str(mask_path) if mask_path else None,
                    "label": is_anomaly,
                    "defect_type": defect_type,
                    "filename": img_path.name,
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Load RGB image
        with Image.open(sample["image_path"]) as img:
            image = img.convert("RGB")
            image_tensor = self.transform(image)

        # Load or generate binary ground-truth mask
        if sample["mask_path"] is not None:
            with Image.open(sample["mask_path"]) as mask_img:
                mask = mask_img.convert("L")
                mask_tensor = self.mask_transform(mask)
                # Binarize mask: any pixel > 0 is defective (1.0)
                mask_tensor = (mask_tensor > 0.5).float()
        else:
            # Normal sample: ground truth mask is all zeros
            mask_tensor = torch.zeros((1, self.image_size[0], self.image_size[1]), dtype=torch.float32)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "label": sample["label"],
            "defect_type": sample["defect_type"],
            "image_path": sample["image_path"],
            "filename": sample["filename"],
        }


def build_dataloaders(
    root_dir: str,
    category: str,
    image_size: Tuple[int, int] = (256, 256),
    batch_size: int = 16,
    num_workers: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Constructs train and test DataLoaders with automatic OS-aware worker management.
    """
    if num_workers is None:
        # Windows multiprocessing with PyTorch can have spawn issues; default to 0 on Windows
        num_workers = 0 if os.name == "nt" else 2

    train_dataset = MVTecDataset(
        root_dir=root_dir,
        category=category,
        split="train",
        image_size=image_size,
    )

    test_dataset = MVTecDataset(
        root_dir=root_dir,
        category=category,
        split="test",
        image_size=image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # Unsupervised feature extraction doesn't strictly need shuffle
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return train_loader, test_loader
