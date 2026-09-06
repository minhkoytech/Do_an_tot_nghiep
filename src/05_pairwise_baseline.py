"""
05_pairwise_baseline.py
------------------------
Pairwise Classical ML Baseline (RF/SVM) - dung gop y so 4 cua GVHD.

Bai toan: (Reference, Query) -> |feature_ref - feature_query| -> RF/SVM -> Match/Non-match
Day la CUNG mot formulation voi Siamese Network (nhan 1 cap anh, quyet
dinh match/non-match), khac voi RF phan loai anh don le truoc day
(Image -> Genuine/Forgery). Nho vay, so sanh Pairwise RF/SVM vs Siamese
sau nay moi hop ly va cong bang.

Bo dac trung su dung (11/17, da loai dac trung du thua):
    glcm_correlation, hu_1, hu_2, hu_3, hu_4, aspect_ratio,
    pixel_density, glcm_homogeneity, glcm_energy, num_crossings,
    baseline_ratio

    Loai bo glcm_contrast, glcm_dissimilarity (trung voi glcm_homogeneity
    ve mat cong thuc khi GLCM nhi phan: contrast = dissimilarity, va
    homogeneity = 1 - 0.5*contrast), glcm_ASM (trung voi glcm_energy vi
    energy = sqrt(ASM)), va hu_5/hu_6/hu_7 (effect size yeu nhat o
    04_stats_tests.py Phan B).

Ky luat train/val/test (dung gop y so 2 cua GVHD):
    - Fit StandardScaler CHI tren train.
    - Tune hyperparameter (RF, SVM) tren train, CHON model/threshold
      dua tren hieu nang o VAL (khong dung test).
    - Threshold quyet dinh (dua tren EER) duoc CHON va CO DINH tren val.
    - Test set CHI duoc danh gia DUY NHAT MOT LAN, sau khi model va
      threshold da duoc co dinh.

Chi so danh gia (dung gop y so 1 cua GVHD):
    - Accuracy, Precision, Recall, F1, ROC-AUC (threshold-independent)
    - FAR (False Acceptance Rate): ty le cap gia (label=0) bi chap nhan
      nham la match. QUAN TRONG NHAT trong ngan hang.
    - FRR (False Rejection Rate): ty le cap that (label=1) bi tu choi nham.
    - EER (Equal Error Rate): diem ma FAR ~ FRR.
    - Bao cao RIENG FAR cho tung loai gia mao (skilled vs random forgery).
    - Bao cao ket qua theo TUNG WRITER (khong chi tong the) de xem mo
      hinh co on dinh giua cac nguoi ky hay khong.

Output:
    model_artifacts/pairwise_rf.joblib
    model_artifacts/pairwise_svm.joblib
    model_artifacts/pairwise_scaler.joblib
    model_artifacts/pairwise_threshold.json
    results/tables/pairwise_baseline_test_summary.csv
    results/tables/pairwise_baseline_per_writer.csv
    results/tables/pairwise_baseline_per_pairtype.csv
    results/figures/pairwise_baseline_roc_curve.png

Cach dung (khong can tham so, path da khop san):
    cd src
    python 05_pairwise_baseline.py
"""

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_curve, roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
)

# ---------------------------------------------------------------------------
# Duong dan mac dinh khop voi cau truc project DO_AN_TOT_NGHIEP
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FEATURES = PROJECT_ROOT / "data" / "processed" / "features.csv"
DEFAULT_PAIRS_DIR = PROJECT_ROOT / "data" / "processed" / "pairs"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "model_artifacts"
DEFAULT_TABLES_DIR = PROJECT_ROOT / "results" / "tables"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

# Bo dac trung da chon sau khi loai du thua (xem 04_stats_tests.py Phan B)
SELECTED_FEATURES = [
    "glcm_correlation", "hu_1", "hu_2", "hu_3", "hu_4",
    "aspect_ratio", "pixel_density", "glcm_homogeneity",
    "glcm_energy", "num_crossings", "baseline_ratio",
]

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Chuan bi du lieu: tinh delta feature vector cho tung pair
# ---------------------------------------------------------------------------
def compute_pair_deltas(pairs_df: pd.DataFrame, features_df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Join features_df vao pairs_df, tinh delta = |feat_ref - feat_query|."""
    feat_indexed = features_df.set_index("path")[feature_cols]
    merged = pairs_df.merge(
        feat_indexed.add_suffix("_ref"), left_on="reference_path", right_index=True, how="inner"
    )
    merged = merged.merge(
        feat_indexed.add_suffix("_query"), left_on="query_path", right_index=True, how="inner"
    )
    for col in feature_cols:
        merged[f"delta_{col}"] = (merged[f"{col}_ref"] - merged[f"{col}_query"]).abs()
    delta_cols = [f"delta_{c}" for c in feature_cols]
    keep_cols = ["reference_path", "query_path", "label", "pair_type", "writer_id_ref"] + delta_cols
    return merged[keep_cols], delta_cols


# ---------------------------------------------------------------------------
# EER: tim threshold tren VAL, sau do ap dung CO DINH len TEST
# ---------------------------------------------------------------------------
def find_eer_threshold(y_true_val: np.ndarray, scores_val: np.ndarray):
    """
    Dung sklearn.roc_curve: fpr chinh la FAR (ty le am tinh - cap gia -
    bi chap nhan nham), tpr = 1 - FRR. EER la diem ma FAR ~ FRR, tuc
    fpr + tpr ~ 1. Tra ve (threshold, eer_value_tren_val).
    """
    fpr, tpr, thresholds = roc_curve(y_true_val, scores_val)
    frr = 1 - tpr
    diff = np.abs(fpr - frr)
    idx = np.argmin(diff)
    eer_threshold = thresholds[idx]
    eer_value = (fpr[idx] + frr[idx]) / 2
    return eer_threshold, eer_value


def compute_far_frr(y_true: np.ndarray, y_pred: np.ndarray):
    """
    FAR = FP / (FP + TN)  : ty le cap GIA (label=0) bi chap nhan nham la match
    FRR = FN / (FN + TP)  : ty le cap THAT (label=1) bi tu choi nham
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    frr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    return far, frr


# ---------------------------------------------------------------------------
# Huan luyen + chon hyperparameter tot nhat dua tren EER cua VAL
# ---------------------------------------------------------------------------
def train_and_select_best(model_name, param_grid, X_train, y_train, X_val, y_val):
    """
    Huan luyen tung cau hinh hyperparameter tren TRAIN, danh gia EER tren
    VAL, chon cau hinh co EER thap nhat. Khong dung test set o buoc nay.
    """
    best_model, best_eer, best_params = None, np.inf, None

    for params in param_grid:
        if model_name == "rf":
            model = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1, **params)
        elif model_name == "svm":
            model = SVC(random_state=RANDOM_SEED, **params)
        else:
            raise ValueError(model_name)

        model.fit(X_train, y_train)
        scores_val = model.decision_function(X_val) if model_name == "svm" else model.predict_proba(X_val)[:, 1]
        _, eer_value = find_eer_threshold(y_val, scores_val)

        print(f"  [{model_name}] params={params} -> EER(val)={eer_value:.4f}")
        if eer_value < best_eer:
            best_model, best_eer, best_params = model, eer_value, params

    print(f"  >>> Best {model_name}: params={best_params}, EER(val)={best_eer:.4f}")
    return best_model, best_params, best_eer


def get_scores(model, model_name, X):
    return model.decision_function(X) if model_name == "svm" else model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Danh gia chi tiet tren TEST (chi 1 lan, threshold da co dinh tu VAL)
# ---------------------------------------------------------------------------
def evaluate_on_test(model, model_name, threshold, test_df, delta_cols, imputer, scaler):
    X_test = scaler.transform(imputer.transform(test_df[delta_cols].values))
    y_test = test_df["label"].values
    scores_test = get_scores(model, model_name, X_test)
    y_pred_test = (scores_test >= threshold).astype(int)

    far, frr = compute_far_frr(y_test, y_pred_test)
    overall = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred_test),
        "precision": precision_score(y_test, y_pred_test, zero_division=0),
        "recall": recall_score(y_test, y_pred_test, zero_division=0),
        "f1": f1_score(y_test, y_pred_test, zero_division=0),
        "roc_auc": roc_auc_score(y_test, scores_test),
        "FAR": far,
        "FRR": frr,
        "threshold_from_val": threshold,
    }

    # FAR rieng cho tung loai gia mao - quan trong voi ngan hang
    per_pairtype = []
    for pair_type in ["skilled_forgery", "random_forgery"]:
        subset = test_df["pair_type"] == pair_type
        if subset.sum() == 0:
            continue
        y_true_sub = test_df.loc[subset, "label"].values
        y_pred_sub = y_pred_test[subset.values]
        false_accept_rate = np.mean(y_pred_sub == 1)  # tat ca deu la label=0 nen sai la accept nham
        per_pairtype.append({"model": model_name, "pair_type": pair_type, "n": int(subset.sum()), "FAR": false_accept_rate})
    # FRR cho genuine_genuine
    subset = test_df["pair_type"] == "genuine_genuine"
    if subset.sum() > 0:
        y_pred_sub = y_pred_test[subset.values]
        false_reject_rate = np.mean(y_pred_sub == 0)
        per_pairtype.append({"model": model_name, "pair_type": "genuine_genuine", "n": int(subset.sum()), "FRR": false_reject_rate})

    # Ket qua theo tung writer - de xem mo hinh co on dinh giua cac nguoi ky
    per_writer = []
    eval_df = test_df.copy()
    eval_df["y_pred"] = y_pred_test
    eval_df["correct"] = (eval_df["y_pred"] == eval_df["label"]).astype(int)
    for writer_id, group in eval_df.groupby("writer_id_ref"):
        writer_far, writer_frr = compute_far_frr(group["label"].values, group["y_pred"].values)
        per_writer.append(
            {
                "model": model_name,
                "writer_id": writer_id,
                "n_pairs": len(group),
                "accuracy": group["correct"].mean(),
                "FAR": writer_far,
                "FRR": writer_frr,
            }
        )

    return overall, pd.DataFrame(per_pairtype), pd.DataFrame(per_writer), (y_test, scores_test)


def plot_roc_curves(results_scores: dict, out_path: Path):
    plt.figure(figsize=(6, 6))
    for model_name, (y_test, scores_test) in results_scores.items():
        fpr, tpr, _ = roc_curve(y_test, scores_test)
        auc = roc_auc_score(y_test, scores_test)
        plt.plot(fpr, tpr, label=f"{model_name.upper()} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("FAR (False Acceptance Rate)")
    plt.ylabel("1 - FRR (True Accept Rate)")
    plt.title("ROC Curve - Pairwise Classical ML Baseline (Test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Da luu bieu do ROC: {out_path}")


def parse_writer_split(writer_split_path, split_name):
    with open(writer_split_path, "r") as f:
        for line in f:
            if line.strip().startswith(split_name):
                bracket_content = line.split(":", 1)[1]
                return [int(x) for x in re.findall(r"\d+", bracket_content)]
    raise ValueError(f"Khong tim thay split '{split_name}'")


def main():
    parser = argparse.ArgumentParser(description="Pairwise Classical ML baseline (RF/SVM)")
    parser.add_argument("--features", default=str(DEFAULT_FEATURES))
    parser.add_argument("--pairs_dir", default=str(DEFAULT_PAIRS_DIR))
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--tables_dir", default=str(DEFAULT_TABLES_DIR))
    parser.add_argument("--figures_dir", default=str(DEFAULT_FIGURES_DIR))
    args = parser.parse_args()

    features_df = pd.read_csv(args.features)
    pairs_train = pd.read_csv(Path(args.pairs_dir) / "pairs_train.csv")
    pairs_val = pd.read_csv(Path(args.pairs_dir) / "pairs_val.csv")
    pairs_test = pd.read_csv(Path(args.pairs_dir) / "pairs_test.csv")

    train_df, delta_cols = compute_pair_deltas(pairs_train, features_df, SELECTED_FEATURES)
    val_df, _ = compute_pair_deltas(pairs_val, features_df, SELECTED_FEATURES)
    test_df, _ = compute_pair_deltas(pairs_test, features_df, SELECTED_FEATURES)

    print(f"Train pairs: {len(train_df)} | Val pairs: {len(val_df)} | Test pairs: {len(test_df)}")
    print(f"So dac trung su dung: {len(delta_cols)} -> {delta_cols}")

    # Fit imputer + scaler CHI tren train (mot vai delta feature co the la
    # NaN, vd baseline_ratio khi anh gan nhu trong / chia cho 0)
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(train_df[delta_cols].values)
    X_val_imputed = imputer.transform(val_df[delta_cols].values)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_imputed)
    X_val = scaler.transform(X_val_imputed)
    y_train = train_df["label"].values
    y_val = val_df["label"].values

    # -----------------------------------------------------------------
    # Huan luyen + chon hyperparameter tren TRAIN, chon model tren VAL
    # -----------------------------------------------------------------
    print("\n=== Huan luyen Random Forest (chon hyperparameter tren VAL) ===")
    rf_grid = [
        {"n_estimators": 200, "max_depth": None},
        {"n_estimators": 200, "max_depth": 10},
        {"n_estimators": 400, "max_depth": 20},
    ]
    best_rf, best_rf_params, _ = train_and_select_best("rf", rf_grid, X_train, y_train, X_val, y_val)

    print("\n=== Huan luyen SVM (chon hyperparameter tren VAL) ===")
    svm_grid = [
        {"C": 1, "kernel": "rbf", "gamma": "scale"},
        {"C": 10, "kernel": "rbf", "gamma": "scale"},
        {"C": 1, "kernel": "linear"},
    ]
    best_svm, best_svm_params, _ = train_and_select_best("svm", svm_grid, X_train, y_train, X_val, y_val)

    # -----------------------------------------------------------------
    # Chon threshold (EER) TREN VAL, CO DINH truoc khi dung test
    # -----------------------------------------------------------------
    rf_scores_val = get_scores(best_rf, "rf", X_val)
    svm_scores_val = get_scores(best_svm, "svm", X_val)
    rf_threshold, rf_eer_val = find_eer_threshold(y_val, rf_scores_val)
    svm_threshold, svm_eer_val = find_eer_threshold(y_val, svm_scores_val)
    print(f"\nThreshold RF (tu val, EER={rf_eer_val:.4f}): {rf_threshold:.4f}")
    print(f"Threshold SVM (tu val, EER={svm_eer_val:.4f}): {svm_threshold:.4f}")

    # -----------------------------------------------------------------
    # Danh gia TREN TEST - CHI MOT LAN, threshold da co dinh
    # -----------------------------------------------------------------
    print("\n=== Danh gia tren TEST (mot lan duy nhat) ===")
    rf_overall, rf_pairtype, rf_writer, rf_scores_test = evaluate_on_test(
        best_rf, "rf", rf_threshold, test_df, delta_cols, imputer, scaler
    )
    svm_overall, svm_pairtype, svm_writer, svm_scores_test = evaluate_on_test(
        best_svm, "svm", svm_threshold, test_df, delta_cols, imputer, scaler
    )

    summary_df = pd.DataFrame([rf_overall, svm_overall])
    tables_dir = Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_path = tables_dir / "pairwise_baseline_test_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n{summary_df.to_string(index=False)}")
    print(f"\nDa luu: {summary_path}")

    pairtype_df = pd.concat([rf_pairtype, svm_pairtype], ignore_index=True)
    pairtype_path = tables_dir / "pairwise_baseline_per_pairtype.csv"
    pairtype_df.to_csv(pairtype_path, index=False)
    print(f"\n=== FAR/FRR theo loai gia mao ===\n{pairtype_df.to_string(index=False)}")
    print(f"Da luu: {pairtype_path}")

    writer_df = pd.concat([rf_writer, svm_writer], ignore_index=True)
    writer_path = tables_dir / "pairwise_baseline_per_writer.csv"
    writer_df.to_csv(writer_path, index=False)
    print(f"\n=== Do on dinh theo writer (model=rf) ===")
    print(writer_df[writer_df["model"] == "rf"].describe())
    print(f"Da luu: {writer_path}")

    # ROC curve
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_roc_curves({"rf": rf_scores_test, "svm": svm_scores_test}, figures_dir / "pairwise_baseline_roc_curve.png")

    # -----------------------------------------------------------------
    # Luu model artifacts de dung lai sau nay (vd so sanh voi Siamese)
    # -----------------------------------------------------------------
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_rf, model_dir / "pairwise_rf.joblib")
    joblib.dump(best_svm, model_dir / "pairwise_svm.joblib")
    joblib.dump(imputer, model_dir / "pairwise_imputer.joblib")
    joblib.dump(scaler, model_dir / "pairwise_scaler.joblib")
    with open(model_dir / "pairwise_threshold.json", "w") as f:
        json.dump(
            {
                "selected_features": SELECTED_FEATURES,
                "rf_params": best_rf_params,
                "rf_threshold": float(rf_threshold),
                "svm_params": best_svm_params,
                "svm_threshold": float(svm_threshold),
            },
            f,
            indent=2,
        )
    print(f"\nDa luu model artifacts vao: {model_dir}")


if __name__ == "__main__":
    main()