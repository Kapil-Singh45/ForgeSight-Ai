import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

# Add project root to python path for imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_config, set_seed, get_device, save_inspection_panel
from src.transforms import denormalize_image
from src.dataset import build_dataloaders
from src.model import PatchFeatureExtractor
from src.anomaly_detector import PatchCoreDetector
from src.metrics import compute_image_metrics, compute_pixel_auroc, compute_au_pro


def main():
    parser = argparse.ArgumentParser(description="Evaluate PatchCore Anomaly Detector.")
    parser.add_argument("--config", type=str, default="configs/mvtec_config.yaml", help="Path to config file")
    parser.add_argument("--category", type=str, default=None, help="Override category in config")
    parser.add_argument("--num_vis", type=int, default=10, help="Number of visual panels to export")
    args = parser.parse_args()

    # 1. Load config
    cfg = load_config(args.config)
    category = args.category or cfg["dataset"]["category"]
    data_dir = cfg["dataset"]["data_dir"]
    image_size = tuple(cfg["dataset"]["image_size"])
    batch_size = cfg["system"]["batch_size"]
    num_workers = cfg["system"]["num_workers"]
    output_dir = Path(cfg["system"]["output_dir"])

    set_seed(cfg["system"]["seed"])
    device = get_device()

    print(f"\n==========================================")
    print(f"  ForgeSight AI - Test Set Evaluation")
    print(f"  Category: {category} | Hardware: {device}")
    print(f"==========================================\n")

    # 2. Check checkpoint
    checkpoint_path = output_dir / "checkpoints" / f"{category}_patchcore.pt"
    if not checkpoint_path.exists():
        print(f"[!] Checkpoint not found at: {checkpoint_path}")
        print(f"[!] Please run training first: python scripts/run_train.py --category {category}")
        sys.exit(1)

    # 3. Load Model and Detector
    model_cfg = cfg["model"]
    feature_extractor = PatchFeatureExtractor(
        backbone_name=model_cfg["backbone"],
        layers=model_cfg["layers"],
        pretrained=model_cfg["pretrained"],
    )
    detector = PatchCoreDetector(
        feature_extractor=feature_extractor,
        device=device,
    )
    detector.load(checkpoint_path)

    # 4. Construct Test DataLoader
    _, test_loader = build_dataloaders(
        root_dir=data_dir,
        category=category,
        image_size=image_size,
        batch_size=1,  # Evaluation sample-by-sample for heatmap generation & localization
        num_workers=num_workers,
    )
    print(f"[*] Loaded {len(test_loader.dataset)} test samples (normal + defective).")

    # 5. Run Inference over Test Set
    gt_labels = []
    pred_image_scores = []
    gt_masks = []
    anomaly_maps = []
    test_samples_for_vis = []

    print("[*] Running inference and generating defect anomaly maps...")
    start_time = time.time()

    for idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
        image_tensor = batch["image"]
        mask_tensor = batch["mask"]
        label = batch["label"].item()
        defect_type = batch["defect_type"][0]
        filename = batch["filename"][0]

        # Generate anomaly score and 2D heatmap
        image_score, anomaly_map = detector.predict_image(image_tensor, target_size=image_size)

        gt_labels.append(label)
        pred_image_scores.append(image_score)

        gt_mask_np = mask_tensor.squeeze().cpu().numpy()
        gt_masks.append(gt_mask_np)
        anomaly_maps.append(anomaly_map)

        if idx < args.num_vis:
            # Denormalize image for clear visualization
            vis_img = denormalize_image(image_tensor).squeeze().permute(1, 2, 0).cpu().numpy()
            test_samples_for_vis.append({
                "image": vis_img,
                "gt_mask": gt_mask_np,
                "anomaly_map": anomaly_map,
                "label": label,
                "defect_type": defect_type,
                "filename": filename,
                "score": image_score,
            })

    inference_duration = time.time() - start_time
    fps = len(test_loader.dataset) / inference_duration
    print(f"[+] Inference complete in {inference_duration:.2f}s ({fps:.1f} FPS).")

    # 6. Compute Benchmark Metrics
    print("\n[*] Computing Image-level classification metrics...")
    img_metrics = compute_image_metrics(
        y_true=np.array(gt_labels),
        y_scores=np.array(pred_image_scores),
    )

    print("[*] Computing Pixel-level ROC-AUC localization metric...")
    pix_auroc = compute_pixel_auroc(gt_masks=gt_masks, anomaly_maps=anomaly_maps)

    print("[*] Computing Per-Region Overlap (AU-PRO) metric...")
    pro_max_fpr = cfg["evaluation"]["pro_max_fpr"]
    au_pro = compute_au_pro(gt_masks=gt_masks, anomaly_maps=anomaly_maps, max_fpr=pro_max_fpr)

    # 7. Print Results Table
    print("\n=======================================================")
    print(f"       BENCHMARK RESULTS: MVTec AD [{category.upper()}]")
    print("=======================================================")
    print(f"  Image-level ROC-AUC  : {img_metrics['image_auroc'] * 100:.2f}%")
    print(f"  Image-level F1-Score : {img_metrics['f1_score'] * 100:.2f}%")
    print(f"  Pixel-level ROC-AUC  : {pix_auroc * 100:.2f}%")
    print(f"  Per-Region Overlap   : {au_pro * 100:.2f}% (PRO @ FPR <= {pro_max_fpr*100:.0f}%)")
    print(f"  Optimal Threshold    : {img_metrics['optimal_threshold']:.4f}")
    print("=======================================================\n")

    # 8. Export Visual Inspection Panels
    heatmaps_dir = output_dir / "heatmaps" / category
    print(f"[*] Exporting {len(test_samples_for_vis)} inspection heatmaps to {heatmaps_dir}...")
    for item in test_samples_for_vis:
        status = "DEFECT" if item["label"] == 1 else "NORMAL"
        save_name = f"{item['defect_type']}_{item['filename']}"
        save_path = heatmaps_dir / save_name
        title = f"{category.upper()} - {status} ({item['defect_type']}) | Anomaly Score: {item['score']:.3f}"
        save_inspection_panel(
            image_rgb=item["image"],
            gt_mask=item["gt_mask"],
            anomaly_map=item["anomaly_map"],
            save_path=str(save_path),
            title=title,
            threshold=img_metrics["optimal_threshold"],
        )

    # 9. Save JSON Summary Report
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / f"{category}_evaluation.json"

    report_data = {
        "category": category,
        "dataset": cfg["dataset"]["name"],
        "num_test_samples": len(gt_labels),
        "num_defective_samples": int(sum(gt_labels)),
        "num_normal_samples": int(len(gt_labels) - sum(gt_labels)),
        "image_auroc": img_metrics["image_auroc"],
        "image_f1_score": img_metrics["f1_score"],
        "pixel_auroc": pix_auroc,
        "au_pro": au_pro,
        "optimal_threshold": img_metrics["optimal_threshold"],
        "pro_max_fpr": pro_max_fpr,
        "fps": fps,
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4)
    print(f"[+] Saved evaluation report to: {report_file}\n")


if __name__ == "__main__":
    main()
