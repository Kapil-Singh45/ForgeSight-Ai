from typing import Optional, Tuple, Union
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from src.model import PatchFeatureExtractor


class PatchCoreDetector:
    """
    PatchCore Anomaly Detector.

    Builds a memory bank of normal patch representations extracted from defect-free
    industrial images, and scores test samples based on their nearest-neighbor distances.
    """

    def __init__(
        self,
        feature_extractor: PatchFeatureExtractor,
        coreset_sampling_ratio: float = 0.1,
        n_neighbors: int = 9,
        gaussian_blur_sigma: float = 4.0,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            feature_extractor: Feature extractor module (frozen CNN).
            coreset_sampling_ratio: Fraction of patch embeddings to keep in the memory bank.
            n_neighbors: Number of nearest neighbors to aggregate for scoring.
            gaussian_blur_sigma: Standard deviation for Gaussian smoothing of anomaly maps.
            device: torch.device ('cuda' or 'cpu').
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = feature_extractor.to(self.device)
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.n_neighbors = n_neighbors
        self.gaussian_blur_sigma = gaussian_blur_sigma

        # Memory bank holding normal patch vectors: Shape (N_subsampled, Feature_dim)
        self.memory_bank: Optional[np.ndarray] = None
        self.nn_index: Optional[NearestNeighbors] = None
        self.patch_grid_shape: Optional[Tuple[int, int]] = None

    def fit(self, train_loader) -> None:
        """
        Extracts patch embeddings from normal training images and creates the memory bank.
        """
        self.feature_extractor.eval()
        all_features = []

        print(f"[*] Extracting normal feature representations on device: {self.device}")
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Fitting memory bank"):
                images = batch["image"].to(self.device)
                # Output shape: (B, N_patches, Feature_dim)
                features = self.feature_extractor(images)
                
                # Cache patch spatial grid resolution (e.g. 32x32 for 256x256 image with stride 8)
                if self.patch_grid_shape is None:
                    n_patches = features.shape[1]
                    grid_size = int(np.sqrt(n_patches))
                    self.patch_grid_shape = (grid_size, grid_size)

                # Flatten across batch and patch dimensions
                features_flat = features.reshape(-1, features.shape[-1]).cpu().numpy()
                all_features.append(features_flat)

        raw_memory = np.concatenate(all_features, axis=0)
        total_patches = raw_memory.shape[0]
        print(f"[*] Collected {total_patches:,} patch vectors of dimension {raw_memory.shape[1]}.")

        # Subsample to keep memory bank compact and fast
        if 0.0 < self.coreset_sampling_ratio < 1.0:
            target_size = max(1, int(total_patches * self.coreset_sampling_ratio))
            print(f"[*] Subsampling memory bank ({self.coreset_sampling_ratio * 100:.1f}%) -> {target_size:,} patches...")
            indices = np.random.RandomState(42).choice(total_patches, size=target_size, replace=False)
            self.memory_bank = raw_memory[indices]
        else:
            self.memory_bank = raw_memory

        print(f"[*] Indexing memory bank with Scikit-learn NearestNeighbors (metric=euclidean)...")
        self.nn_index = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            algorithm="auto",
            metric="euclidean",
            n_jobs=-1,
        )
        self.nn_index.fit(self.memory_bank)
        print("[*] Memory bank fitting complete.")

    def predict_image(
        self,
        image_tensor: torch.Tensor,
        target_size: Tuple[int, int] = (256, 256),
    ) -> Tuple[float, np.ndarray]:
        """
        Scores a single input image and generates its 2D anomaly heatmap.

        Args:
            image_tensor: Input tensor of shape (1, C, H, W) or (C, H, W).
            target_size: (Height, Width) of the original image to upscale the heatmap to.

        Returns:
            image_score: Continuous scalar anomaly score for the image.
            anomaly_map: 2D numpy array of shape (H, W) with pixel anomaly values.
        """
        if self.nn_index is None:
            raise RuntimeError("Detector must be fitted with fit() before running inference.")

        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)
        self.feature_extractor.eval()

        with torch.no_grad():
            features = self.feature_extractor(image_tensor)  # (1, N_patches, Dim)
            features_np = features.squeeze(0).cpu().numpy()  # (N_patches, Dim)

        # Compute nearest neighbor distance for each patch
        # distances shape: (N_patches, n_neighbors)
        distances, _ = self.nn_index.kneighbors(features_np, n_neighbors=1)
        patch_scores = distances[:, 0]  # Minimum distance to normal memory bank

        # Image-level anomaly score is the maximum patch distance
        image_score = float(np.max(patch_scores))

        # Reshape patch scores back into 2D spatial grid
        grid_h, grid_w = self.patch_grid_shape
        patch_map = patch_scores.reshape(grid_h, grid_w)

        # Upscale patch map to original resolution via bilinear interpolation
        patch_tensor = torch.from_numpy(patch_map).unsqueeze(0).unsqueeze(0).float()
        upscaled_map = F.interpolate(
            patch_tensor, size=target_size, mode="bilinear", align_corners=False
        ).squeeze().numpy()

        # Apply Gaussian smoothing filter
        if self.gaussian_blur_sigma > 0:
            smoothed_map = gaussian_filter(upscaled_map, sigma=self.gaussian_blur_sigma)
        else:
            smoothed_map = upscaled_map

        return image_score, smoothed_map

    def save(self, filepath: Union[str, Path]) -> None:
        """Saves memory bank and config to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "memory_bank": self.memory_bank,
            "patch_grid_shape": self.patch_grid_shape,
            "coreset_sampling_ratio": self.coreset_sampling_ratio,
            "n_neighbors": self.n_neighbors,
            "gaussian_blur_sigma": self.gaussian_blur_sigma,
        }
        torch.save(checkpoint, filepath)
        print(f"[*] Saved detector checkpoint to {filepath}")

    def load(self, filepath: Union[str, Path]) -> None:
        """Loads memory bank from disk and initializes the search index."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found at {filepath}")

        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
        self.memory_bank = checkpoint["memory_bank"]
        self.patch_grid_shape = checkpoint["patch_grid_shape"]
        self.coreset_sampling_ratio = checkpoint["coreset_sampling_ratio"]
        self.n_neighbors = checkpoint["n_neighbors"]
        self.gaussian_blur_sigma = checkpoint["gaussian_blur_sigma"]

        self.nn_index = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            algorithm="auto",
            metric="euclidean",
            n_jobs=-1,
        )
        self.nn_index.fit(self.memory_bank)
        print(f"[*] Loaded detector checkpoint from {filepath}")
