import io
import sys
from pathlib import Path
import requests
import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import PatchFeatureExtractor
from src.anomaly_detector import PatchCoreDetector
from src.transforms import get_image_transforms


def create_test_suite():
    suite_dir = Path("data/test_suite")
    suite_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Loading PatchCore model to evaluate test suite...")
    feature_extractor = PatchFeatureExtractor(backbone_name="resnet18", pretrained=True)
    detector = PatchCoreDetector(feature_extractor=feature_extractor)
    detector.load("outputs/checkpoints/bottle_patchcore.pt")
    transform = get_image_transforms((256, 256))
    threshold = 1.9485

    base_url = "https://huggingface.co/datasets/foersben/mvtec-ad/resolve/main/bottle/test"

    # 1. Normal Test Bottles (PASS)
    normal_indices = [0, 1, 2, 3, 4]
    print("\n[*] Fetching genuine Normal test bottles...")
    for idx in normal_indices:
        url = f"{base_url}/good/{idx:03d}.png"
        img = Image.open(io.BytesIO(requests.get(url).content)).convert("RGB")
        filename = f"PASS_Normal_Bottle_{idx+1:02d}.png"
        filepath = suite_dir / filename
        img.save(filepath)

        score, _ = detector.predict_image(transform(img))
        verdict = "PASS" if score < threshold else "REJECT"
        print(f"  [+] {filename:<30} | Score: {score:.4f} | Verdict: {verdict}")

    # 2. Genuine Defective Bottles (REJECT)
    defects = [
        ("broken_large", 0, "REJECT_Broken_Large_01.png"),
        ("broken_large", 1, "REJECT_Broken_Large_02.png"),
        ("broken_small", 0, "REJECT_Broken_Small_01.png"),
        ("broken_small", 1, "REJECT_Broken_Small_02.png"),
        ("contamination", 0, "REJECT_Contamination_01.png"),
        ("contamination", 1, "REJECT_Contamination_02.png"),
    ]

    print("\n[*] Fetching genuine Defective test bottles...")
    for defect_type, idx, filename in defects:
        url = f"{base_url}/{defect_type}/{idx:03d}.png"
        img = Image.open(io.BytesIO(requests.get(url).content)).convert("RGB")
        filepath = suite_dir / filename
        img.save(filepath)

        score, _ = detector.predict_image(transform(img))
        verdict = "PASS" if score < threshold else "REJECT"
        print(f"  [+] {filename:<30} | Score: {score:.4f} | Verdict: {verdict}")

    # 3. Synthetic Novel Defects (Injected onto Normal Bottle 0)
    print("\n[*] Generating Synthetic Flaws on Normal Bottle...")
    normal_base = Image.open(suite_dir / "PASS_Normal_Bottle_01.png").copy()

    # Synthetic Flaw A: Surface Scratch / Crack
    scratch_img = normal_base.copy()
    draw_scratch = ImageDraw.Draw(scratch_img)
    # Draw dark jagged scratch near bottle body
    scratch_coords = [(450, 480), (470, 520), (490, 570), (510, 610), (525, 650)]
    draw_scratch.line(scratch_coords, fill=(30, 20, 20), width=6)
    scratch_filename = "REJECT_Synthetic_Scratch.png"
    scratch_img.save(suite_dir / scratch_filename)
    score_scratch, _ = detector.predict_image(transform(scratch_img))
    print(f"  [+] {scratch_filename:<30} | Score: {score_scratch:.4f} | Verdict: {'REJECT' if score_scratch >= threshold else 'PASS'}")

    # Synthetic Flaw B: Chemical Discoloration / Stain
    stain_img = normal_base.copy()
    draw_stain = ImageDraw.Draw(stain_img)
    # Draw circular contaminant
    draw_stain.ellipse([420, 400, 490, 470], fill=(60, 20, 10))
    stain_filename = "REJECT_Synthetic_Stain.png"
    stain_img.save(suite_dir / stain_filename)
    score_stain, _ = detector.predict_image(transform(stain_img))
    print(f"  [+] {stain_filename:<30} | Score: {score_stain:.4f} | Verdict: {'REJECT' if score_stain >= threshold else 'PASS'}")

    print(f"\n[+] Complete Test Suite generated at: {suite_dir.resolve()}")


if __name__ == "__main__":
    create_test_suite()
