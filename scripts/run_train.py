import argparse
import sys
import time
from pathlib import Path

# Add project root to python path for imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_config, set_seed, get_device
from src.dataset import build_dataloaders
from src.model import PatchFeatureExtractor
from src.anomaly_detector import PatchCoreDetector
from data.download_mvtec import setup_mvtec_category


def main():
    parser = argparse.ArgumentParser(description="Train / Fit PatchCore Anomaly Detector.")
    parser.add_argument("--config", type=str, default="configs/mvtec_config.yaml", help="Path to config file")
    parser.add_argument("--category", type=str, default=None, help="Override category in config")
    parser.add_argument("--auto-download", action="store_true", help="Download dataset if not found")
    args = parser.parse_args()

    # 1. Load configuration
    cfg = load_config(args.config)
    category = args.category or cfg["dataset"]["category"]
    data_dir = cfg["dataset"]["data_dir"]
    image_size = tuple(cfg["dataset"]["image_size"])
    batch_size = cfg["system"]["batch_size"]
    num_workers = cfg["system"]["num_workers"]
    output_dir = Path(cfg["system"]["output_dir"])

    # 2. Set seed for deterministic execution
    set_seed(cfg["system"]["seed"])
    device = get_device()
    print(f"\n==========================================")
    print(f"  ForgeSight AI - Training / Feature Fitting")
    print(f"  Category: {category} | Hardware: {device}")
    print(f"==========================================\n")

    # 3. Check / Prepare Dataset
    category_path = Path(data_dir) / category
    if not category_path.exists():
        if args.auto_download:
            print(f"[*] Category '{category}' not found locally. Auto-downloading...")
            setup_mvtec_category(category=category, data_dir=data_dir)
        else:
            print(f"[!] Category '{category}' not found at {category_path}.")
            print(f"[!] Run 'python data/download_mvtec.py --category {category}' or pass '--auto-download'.")
            sys.exit(1)

    # 4. Construct DataLoaders
    train_loader, _ = build_dataloaders(
        root_dir=data_dir,
        category=category,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    print(f"[*] Loaded {len(train_loader.dataset)} normal training samples.")

    # 5. Initialize Feature Extractor Backbone
    model_cfg = cfg["model"]
    feature_extractor = PatchFeatureExtractor(
        backbone_name=model_cfg["backbone"],
        layers=model_cfg["layers"],
        pretrained=model_cfg["pretrained"],
    )

    # 6. Initialize Anomaly Detector
    det_cfg = cfg["detector"]
    detector = PatchCoreDetector(
        feature_extractor=feature_extractor,
        coreset_sampling_ratio=det_cfg["coreset_sampling_ratio"],
        n_neighbors=det_cfg["n_neighbors"],
        gaussian_blur_sigma=det_cfg["gaussian_blur_sigma"],
        device=device,
    )

    # 7. Fit Memory Bank
    start_time = time.time()
    detector.fit(train_loader)
    elapsed = time.time() - start_time
    print(f"[+] Memory bank fitted in {elapsed:.2f} seconds.")

    # 8. Save Checkpoint
    checkpoint_path = output_dir / "checkpoints" / f"{category}_patchcore.pt"
    detector.save(checkpoint_path)
    print(f"[+] Model checkpoint ready at: {checkpoint_path}\n")


if __name__ == "__main__":
    main()
