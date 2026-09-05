"""
Module tiện ích đánh giá — dùng CHUNG cho 08, 09, 11.

Định nghĩa (chuẩn trong lĩnh vực xác thực sinh trắc học):
- Positive (label=1) = "Match"     -> query genuine so với reference đúng chủ
- Negative (label=0) = "Non-match" -> query forgery so với reference

- FAR (False Acceptance Rate): tỷ lệ chữ ký GIẢ bị chấp nhận NHẦM thành match.
  Chỉ số ngân hàng quan tâm nhất (chấp nhận nhầm gây thiệt hại tài chính).
      FAR = FP / (FP + TN)

- FRR (False Rejection Rate): tỷ lệ chữ ký THẬT bị từ chối NHẦM.
      FRR = FN / (FN + TP)
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def compute_far_frr(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    frr = np.mean(y_pred[pos_mask] == 0) if pos_mask.sum() > 0 else float("nan")
    far = np.mean(y_pred[neg_mask] == 1) if neg_mask.sum() > 0 else float("nan")
    return far, frr


def find_threshold_at_target_far(y_true, y_scores, target_far=0.10):
    """Chọn threshold sao cho FAR trên validation KHÔNG VƯỢT QUÁ target_far
    (ưu tiên đúng nhu cầu ngân hàng, chấp nhận FRR cao hơn để đổi lấy FAR thấp)."""
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    neg_scores = np.sort(y_scores[y_true == 0])[::-1]
    n_neg = len(neg_scores)
    if n_neg == 0:
        return float(np.median(y_scores)), float("nan")
    idx = int(np.floor(target_far * n_neg))
    idx = min(max(idx, 0), n_neg - 1)
    threshold = neg_scores[idx]
    achieved_far = (idx + 1) / n_neg
    return float(threshold), float(achieved_far)


def find_threshold_by_eer(y_true, y_scores):
    """Tìm threshold tại điểm Equal Error Rate (FAR ≈ FRR) trên VALIDATION."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer_threshold = thresholds[eer_idx]
    eer_value = (fpr[eer_idx] + fnr[eer_idx]) / 2
    return eer_threshold, eer_value


def evaluate_predictions(y_true, y_pred, y_scores, pair_type=None, writer_id=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracy = np.mean(y_true == y_pred)
    far, frr = compute_far_frr(y_true, y_pred)
    auc = roc_auc_score(y_true, y_scores) if len(set(y_true)) > 1 else float("nan")

    result = {"accuracy": accuracy, "FAR": far, "FRR": frr, "ROC_AUC": auc, "n_pairs": len(y_true)}

    if pair_type is not None:
        pair_type = np.asarray(pair_type)
        for ptype in ["negative_skilled", "negative_random"]:
            mask = pair_type == ptype
            if mask.sum() > 0:
                result[f"FAR_{ptype}"] = np.mean(y_pred[mask] == 1)

    per_writer_df = None
    if writer_id is not None:
        df = pd.DataFrame({
            "writer_id": writer_id, "y_true": y_true, "y_pred": y_pred,
            "correct": (y_true == y_pred).astype(int),
        })
        per_writer_df = df.groupby("writer_id").agg(
            n_pairs=("correct", "size"), accuracy=("correct", "mean")
        ).reset_index()

    return result, per_writer_df