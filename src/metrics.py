from typing import List, Dict, Tuple, Any
import numpy as np
from scipy.ndimage import label
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, f1_score


def compute_image_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> Dict[str, Any]:
    """
    Computes image-level anomaly classification metrics.

    Args:
        y_true: Binary ground-truth labels (0 for normal, 1 for defective).
        y_scores: Continuous image anomaly scores.

    Returns:
        Dictionary containing image_auroc, optimal_threshold, precision, recall, f1_score.
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)

    auroc = roc_auc_score(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Optimal threshold using Youden's J statistic (TPR - FPR)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    optimal_threshold = float(thresholds[best_idx])

    # Compute binary predictions at optimal threshold
    y_pred = (y_scores >= optimal_threshold).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Precision & recall curve
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_scores)

    return {
        "image_auroc": float(auroc),
        "optimal_threshold": optimal_threshold,
        "f1_score": float(f1),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
    }


def compute_pixel_auroc(
    gt_masks: List[np.ndarray],
    anomaly_maps: List[np.ndarray],
) -> float:
    """
    Computes pixel-level ROC-AUC over all pixels in the evaluation set.

    Args:
        gt_masks: List of binary ground-truth masks of shape (H, W).
        anomaly_maps: List of continuous anomaly score heatmaps of shape (H, W).

    Returns:
        Scalar pixel-level ROC-AUC in [0.0, 1.0].
    """
    flat_gt = np.concatenate([m.ravel() for m in gt_masks]).astype(int)
    flat_scores = np.concatenate([s.ravel() for s in anomaly_maps]).astype(np.float32)

    return float(roc_auc_score(flat_gt, flat_scores))


def compute_au_pro(
    gt_masks: List[np.ndarray],
    anomaly_maps: List[np.ndarray],
    max_fpr: float = 0.3,
    num_thresholds: int = 200,
) -> float:
    """
    Computes the Area Under the Per-Region Overlap curve (AU-PRO) up to max_fpr.
    
    Standard MVTec benchmark metric: treats every connected defect region
    equally regardless of pixel count (prevents large defects from dominating).

    Args:
        gt_masks: List of 2D binary numpy arrays (0 = normal, 1 = defect).
        anomaly_maps: List of 2D continuous anomaly maps.
        max_fpr: Maximum False Positive Rate limit (default 0.3).
        num_thresholds: Number of evaluation thresholds.

    Returns:
        Normalized AU-PRO score in [0.0, 1.0].
    """
    # Find all connected defect components
    connected_components = []
    for mask in gt_masks:
        if np.any(mask > 0):
            labeled, num_features = label(mask > 0)
            for region_id in range(1, num_features + 1):
                connected_components.append((mask == region_id, (labeled == region_id)))

    if not connected_components:
        return 1.0

    # Determine evaluation thresholds spanning min and max scores
    all_scores = np.concatenate([s.ravel() for s in anomaly_maps])
    min_score, max_score = float(np.min(all_scores)), float(np.max(all_scores))
    thresholds = np.linspace(min_score, max_score, num_thresholds)

    # Total normal pixels across all images
    total_normal_pixels = sum(np.sum(m == 0) for m in gt_masks)
    if total_normal_pixels == 0:
        return 1.0

    pro_values = []
    fpr_values = []

    for th in thresholds:
        # False Positive Rate on normal pixels
        fp_pixels = sum(
            np.sum((score_map >= th) & (mask == 0))
            for score_map, mask in zip(anomaly_maps, gt_masks)
        )
        fpr = fp_pixels / total_normal_pixels
        fpr_values.append(fpr)

        # Region overlap for each defect component
        region_overlaps = []
        for orig_mask, comp_mask in connected_components:
            comp_size = np.sum(comp_mask)
            # Find matching anomaly map (or compute overlap directly)
            # We locate which image this component belongs to
            # But simpler: evaluate across all components efficiently
            pass

    # Efficient vectorization for PRO:
    # Instead of nested loops, compute region overlap using pre-accumulated values
    fpr_list = []
    pro_list = []

    # Sort thresholds descending
    descending_thresholds = np.linspace(max_score, min_score, num_thresholds)

    for th in descending_thresholds:
        # Compute FPR
        fp_count = 0
        for sm, m in zip(anomaly_maps, gt_masks):
            fp_count += np.count_nonzero((sm >= th) & (m == 0))
        current_fpr = fp_count / total_normal_pixels

        if current_fpr > max_fpr:
            break

        # Compute mean per-region overlap
        overlaps = []
        comp_idx = 0
        for sm, m in zip(anomaly_maps, gt_masks):
            if np.any(m > 0):
                labeled, num_regions = label(m > 0)
                pred_binary = sm >= th
                for r_id in range(1, num_regions + 1):
                    region = labeled == r_id
                    reg_area = np.count_nonzero(region)
                    overlap = np.count_nonzero(pred_binary & region) / reg_area
                    overlaps.append(overlap)

        mean_pro = float(np.mean(overlaps)) if overlaps else 0.0
        fpr_list.append(current_fpr)
        pro_list.append(mean_pro)

    if len(fpr_list) < 2 or max(fpr_list) == 0:
        return 0.0

    # Sort by FPR
    sorted_pairs = sorted(zip(fpr_list, pro_list), key=lambda x: x[0])
    fpr_arr = np.array([p[0] for p in sorted_pairs])
    pro_arr = np.array([p[1] for p in sorted_pairs])

    # Integrate under PRO-FPR curve up to max_fpr
    au_pro = np.trapezoid(pro_arr, fpr_arr) / max_fpr
    return float(np.clip(au_pro, 0.0, 1.0))
