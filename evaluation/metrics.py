"""
Comprehensive Evaluation Metrics for Liver Histopathology Segmentation.
Calculates:
1. Per-class Dice Similarity Coefficient (DSC) & mean Dice
2. Per-class IoU (Jaccard) & mean IoU
3. Inter-Observer Reliability (Cohen's Kappa & Dice agreement between 2 Pathologists)
4. Statistical Hypothesis Testing (Wilcoxon Signed-Rank & Paired t-test)
"""

import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score


def compute_dice_score(pred_mask: np.ndarray, target_mask: np.ndarray, eps: float = 1e-6) -> float:
    """Computes binary Dice coefficient between two binary masks."""
    intersection = np.sum((pred_mask == 1) & (target_mask == 1))
    total = np.sum(pred_mask == 1) + np.sum(target_mask == 1)
    if total == 0:
        return 1.0  # Both empty
    return (2.0 * intersection + eps) / (total + eps)


def compute_iou_score(pred_mask: np.ndarray, target_mask: np.ndarray, eps: float = 1e-6) -> float:
    """Computes binary IoU (Jaccard) between two binary masks."""
    intersection = np.sum((pred_mask == 1) & (target_mask == 1))
    union = np.sum((pred_mask == 1) | (target_mask == 1))
    if union == 0:
        return 1.0  # Both empty
    return (intersection + eps) / (union + eps)


def evaluate_multiclass_segmentation(
    pred_masks: np.ndarray,
    target_masks: np.ndarray,
    class_names: list = ["necrosis", "normal", "steatosis"]
):
    """
    Evaluates multi-class segmentation predictions against ground truth.
    Args:
        pred_masks (np.ndarray): Predicted class map (N, H, W) with values 0, 1, 2, ...
        target_masks (np.ndarray): Ground truth class map (N, H, W).
    """
    results = {}
    num_classes = len(class_names)
    dices_per_class = {c: [] for c in class_names}
    ious_per_class = {c: [] for c in class_names}

    N = pred_masks.shape[0]
    for i in range(N):
        p_map = pred_masks[i]
        t_map = target_masks[i]

        for cls_idx, cls_name in enumerate(class_names):
            p_bin = (p_map == cls_idx).astype(np.uint8)
            t_bin = (t_map == cls_idx).astype(np.uint8)

            d = compute_dice_score(p_bin, t_bin)
            j = compute_iou_score(p_bin, t_bin)

            dices_per_class[cls_name].append(d)
            ious_per_class[cls_name].append(j)

    # Average metrics per class
    print("\n" + "=" * 60)
    print("SEGMENTATION PERFORMANCE EVALUATION")
    print("=" * 60)

    summary_dice = []
    summary_iou = []

    for cls_name in class_names:
        m_dice = float(np.mean(dices_per_class[cls_name]))
        std_dice = float(np.std(dices_per_class[cls_name]))
        m_iou = float(np.mean(ious_per_class[cls_name]))
        std_iou = float(np.std(ious_per_class[cls_name]))

        results[f"Dice_{cls_name}"] = (m_dice, std_dice)
        results[f"IoU_{cls_name}"] = (m_iou, std_iou)

        summary_dice.append(m_dice)
        summary_iou.append(m_iou)

        print(f"Class: {cls_name:12s} | Dice: {m_dice:.4f} ± {std_dice:.4f} | IoU: {m_iou:.4f} ± {std_iou:.4f}")

    results["mDice"] = float(np.mean(summary_dice))
    results["mIoU"] = float(np.mean(summary_iou))

    print("-" * 60)
    print(f"Mean Dice (mDice): {results['mDice']:.4f}")
    print(f"Mean IoU (mIoU)  : {results['mIoU']:.4f}")
    print("=" * 60)

    return results


def evaluate_inter_pathologist_agreement(
    pathologist_1_masks: np.ndarray,
    pathologist_2_masks: np.ndarray,
    class_names: list = ["necrosis", "normal", "steatosis"]
):
    """
    Computes inter-observer agreement between two pathologists to demonstrate ground-truth reliability.
    Reports Cohen's Kappa and Pairwise Dice score.
    """
    flat_p1 = pathologist_1_masks.flatten()
    flat_p2 = pathologist_2_masks.flatten()

    # Subsample if pixel count is massive for Kappa computation
    if len(flat_p1) > 1_000_000:
        idx = np.random.choice(len(flat_p1), size=1_000_000, replace=False)
        flat_p1 = flat_p1[idx]
        flat_p2 = flat_p2[idx]

    kappa = cohen_kappa_score(flat_p1, flat_p2)

    pairwise_dices = {}
    for cls_idx, cls_name in enumerate(class_names):
        d = compute_dice_score((flat_p1 == cls_idx).astype(np.uint8), (flat_p2 == cls_idx).astype(np.uint8))
        pairwise_dices[cls_name] = d

    print("\n" + "=" * 60)
    print("INTER-OBSERVER AGREEMENT (PATHOLOGIST 1 vs PATHOLOGIST 2)")
    print("=" * 60)
    print(f"Cohen's Kappa (κ): {kappa:.4f}  (>0.80 = Almost Perfect, >0.60 = Substantial)")
    for cls_name, d_val in pairwise_dices.items():
        print(f"Pairwise Dice [{cls_name}]: {d_val:.4f}")
    print("=" * 60)

    return {"kappa": kappa, "pairwise_dices": pairwise_dices}


def perform_statistical_tests(model_a_scores: list, model_b_scores: list, metric_name: str = "Dice"):
    """
    Conducts Wilcoxon signed-rank test and paired t-test between two model variants (e.g. Baseline vs Proposed).
    Essential for Scopus Q2-Q3 paper credibility.
    """
    a = np.array(model_a_scores)
    b = np.array(model_b_scores)
    diff = b - a

    # Paired t-test (parametric)
    t_stat, p_val_t = stats.ttest_rel(b, a)

    # Wilcoxon signed-rank test (non-parametric, robust to non-normal distributions)
    try:
        w_stat, p_val_w = stats.wilcoxon(b, a)
    except Exception:
        w_stat, p_val_w = float('nan'), float('nan')

    print(f"\n[STATISTICAL TEST: {metric_name}]")
    print(f"Mean Difference (Proposed - Baseline): {np.mean(diff):+.4f}")
    print(f"Paired t-test: t={t_stat:.3f}, p-value={p_val_t:.4e} {'(Significant p<0.05)' if p_val_t < 0.05 else '(Not Significant)'}")
    print(f"Wilcoxon test: W={w_stat:.3f}, p-value={p_val_w:.4e} {'(Significant p<0.05)' if p_val_w < 0.05 else '(Not Significant)'}")

    return {
        "mean_diff": float(np.mean(diff)),
        "t_stat": float(t_stat),
        "p_val_t": float(p_val_t),
        "p_val_w": float(p_val_w)
    }
