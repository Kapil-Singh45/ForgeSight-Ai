import os
import sys
import tarfile
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from tqdm import tqdm


VALID_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper"
]


def verify_dataset(category_dir: Path) -> bool:
    """Verifies that train, test, and ground_truth folders exist and are populated."""
    train_good = category_dir / "train" / "good"
    test_dir = category_dir / "test"

    if not (train_good.exists() and test_dir.exists()):
        return False

    n_train = len(list(train_good.glob("*.png")))
    n_test = len(list(test_dir.rglob("*.png")))
    print(f"[+] Verified '{category_dir.name}': {n_train} train images, {n_test} test images.")
    return n_train > 0 and n_test > 0


def _download_via_hf_hub(category: str, data_dir: Path) -> bool:
    """Downloads category using huggingface_hub snapshot_download."""
    try:
        from huggingface_hub import snapshot_download
        print(f"[*] Downloading '{category}' from Hugging Face (foersben/mvtec-ad) via huggingface_hub...")
        snapshot_download(
            repo_id="foersben/mvtec-ad",
            repo_type="dataset",
            allow_patterns=f"{category}/*",
            local_dir=str(data_dir),
        )
        return True
    except Exception as e:
        print(f"[!] huggingface_hub download attempt encountered: {e}. Falling back to direct API...")
        return False


def _download_single_file(url: str, dest_path: Path) -> None:
    """Downloads a single file from url to dest_path."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(resp.content)


def _download_via_hf_api(category: str, data_dir: Path) -> None:
    """Fetches category tree from Hugging Face API and downloads all files in parallel."""
    print(f"[*] Querying Hugging Face dataset tree for '{category}'...")
    tree_url = f"https://huggingface.co/api/datasets/foersben/mvtec-ad/tree/main/{category}?recursive=true"
    resp = requests.get(tree_url, timeout=30)
    resp.raise_for_status()

    file_items = [f["path"] for f in resp.json() if f.get("type") == "file"]
    print(f"[*] Found {len(file_items)} files. Downloading in parallel...")

    base_raw_url = "https://huggingface.co/datasets/foersben/mvtec-ad/resolve/main"

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_file = {}
        for rel_path in file_items:
            file_url = f"{base_raw_url}/{rel_path}"
            dest_file = data_dir / rel_path
            future = executor.submit(_download_single_file, file_url, dest_file)
            future_to_file[future] = rel_path

        for future in tqdm(as_completed(future_to_file), total=len(future_to_file), desc=f"Downloading {category}"):
            future.result()


def setup_mvtec_category(category: str, data_dir: str = "./data/mvtec") -> Path:
    """Ensures the specified category is downloaded and verified."""
    if category not in VALID_CATEGORIES:
        raise ValueError(f"Unknown category '{category}'. Available: {VALID_CATEGORIES}")

    dest_dir = Path(data_dir)
    target_dir = dest_dir / category

    if target_dir.exists() and verify_dataset(target_dir):
        print(f"[!] Dataset category '{category}' is already verified at {target_dir}")
        return target_dir

    dest_dir.mkdir(parents=True, exist_ok=True)

    # 1. First attempt: huggingface_hub snapshot_download
    success = _download_via_hf_hub(category, dest_dir)

    # 2. Second attempt: parallel multi-threaded HTTP download
    if not success or not (target_dir.exists() and verify_dataset(target_dir)):
        _download_via_hf_api(category, dest_dir)

    # 3. Final verification
    if not verify_dataset(target_dir):
        raise RuntimeError(f"Verification failed for extracted dataset at {target_dir}")

    print(f"[+] Dataset '{category}' is fully ready at: {target_dir}")
    return target_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and extract MVTec AD category.")
    parser.add_argument("--category", type=str, default="bottle", help="Category name (e.g. bottle, carpet)")
    parser.add_argument("--data_dir", type=str, default="./data/mvtec", help="Directory where data is stored")
    args = parser.parse_args()

    setup_mvtec_category(args.category, args.data_dir)
