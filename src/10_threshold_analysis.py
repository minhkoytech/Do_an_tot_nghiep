"""
10_threshold_analysis.py
--------------------------
So sanh 4 model (RF, SVM, Siamese ensemble, Combined) TAI CUNG MOT MUC
FAR MUC TIEU co dinh (vd 5%, 10%, 15%, 20%) - thay vi so sanh tai
threshold EER rieng cua tung model nhu cac buoc truoc.

TAI SAO CAN PHAN TICH NAY (tra loi cau hoi "sao chon duoc deep learning
neu FAR cua no cao hon RF?"):
    ROC-AUC do kha nang phan biet TREN TOAN BO duong cong ROC, nhung
    threshold EER (can bang FAR~FRR) chi la MOT DIEM CU THE tren duong
    cong do. Ngan hang thuc te KHONG van hanh theo EER - ho dat ra MOT
    MUC FAR TOI DA CHAP NHAN DUOC (vd "khong duoc qua 10% chu ky gia
    lot qua"), roi xem FRR o muc do la bao nhieu. Day moi la cach so
    sanh dung voi quyet dinh thuc te.

    Neu Siamese co ROC-AUC cao hon RF, VE MAT LY THUYET, tai BAT KY
    muc FAR co dinh nao, Siamese CO THE co FRR thap hon RF - du
    threshold EER mac dinh cua Siamese dang cho FAR cao hon RF. Script
    nay kiem tra TRUC TIEP gia thuyet do bang du lieu that, thay vi chi
    suy luan ly thuyet.

QUAN TRONG VE PHUONG PHAP LUAN: threshold cho tung muc FAR muc tieu
duoc CHON TREN VAL (khong phai test), giong het nguyen tac EER truoc
do - tranh data leakage. Test set van chi danh gia 1 lan cho moi cau
hinh threshold.

YEU CAU: da chay xong 05, 06, 08 truoc do.

Cach dung (khong can tham so, path da khop san):
    cd src
    python 10_threshold_analysis.py

Output:
    results/tables/threshold_analysis_target_far.csv
    results/figures/threshold_analysis_frr_vs_far.png
"""

import argparse
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURES = PROJECT_ROOT / "data" / "processed" / "features.csv"
DEFAULT_PAIRS_DIR = PROJECT_ROOT / "data" / "processed" / "pairs"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "model_artifacts"
DEFAULT_TABLES_DIR = PROJECT_ROOT / "results" / "tables"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

TARGET_FAR_LEVELS = [0.05, 0.10, 0.15, 0.20]  # cac muc FAR "ngan hang co the dat ra"


def load_module(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, SRC_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


siamese_mod = load_module("06_siamese_network.py", "siamese_mod_thresh")
combined_mod = load_module("08_combined_model.py", "combined_mod_thresh")
SELECTED_FEATURES = siamese_mod.SELECTED_FEATURES


def find_threshold_at_target_far(y_true_val, scores_val, target_far):
    """
    Tim threshold TREN VAL sao cho FAR dat GAN NHAT voi target_far (khong
    vuot qua, uu tien an toan hon cho ngan hang - tuc FAR thuc te <= target).
    """
    fpr, tpr, thresholds = roc_curve(y_true_val, scores_val)  # fpr = FAR
    # Loc cac diem co FAR <= target, chon diem co FAR GAN target nhat (FRR thap nhat trong so do)
    valid_idx = np.where(fpr <= target_far)[0]
    if len(valid_idx) == 0:
        # Khong co threshold nao dat FAR <= target (hiem), lay diem FAR nho nhat co the
        idx = np.argmin(fpr)
    else:
        idx = valid_idx[np.argmax(fpr[valid_idx])]  # FAR gan target nhat tu duoi len
    return thresholds[idx], fpr[idx], 1 - tpr[idx]  # threshold, FAR dat duoc, FRR tuong ung


def compute_far_frr_accuracy(y_true, y_pred):
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    far = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    frr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    accuracy = (tp + tn) / len(y_true)
    return far, frr, accuracy


def main():
    parser = argparse.ArgumentParser(description="So sanh model tai cung muc FAR muc tieu")
    parser.add_argument("--features", default=str(DEFAULT_FEATURES))
    parser.add_argument("--pairs_dir", default=str(DEFAULT_PAIRS_DIR))
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--tables_dir", default=str(DEFAULT_TABLES_DIR))
    parser.add_argument("--figures_dir", default=str(DEFAULT_FIGURES_DIR))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = Path(args.model_dir)
    features_df = pd.read_csv(args.features)
    pairs_dir = Path(args.pairs_dir)
    pairs_val = pd.read_csv(pairs_dir / "pairs_val.csv")
    pairs_test = pd.read_csv(pairs_dir / "pairs_test.csv")

    # ----- Chuan bi dac trung classical (dung chung cho RF/SVM/Combined) -----
    val_classical, classical_cols = combined_mod.compute_classical_deltas(pairs_val, features_df, SELECTED_FEATURES)
    test_classical, _ = combined_mod.compute_classical_deltas(pairs_test, features_df, SELECTED_FEATURES)
    y_val = val_classical["label"].values
    y_test = test_classical["label"].values

    # ----- Load RF/SVM -----
    rf = joblib.load(model_dir / "pairwise_rf.joblib")
    svm = joblib.load(model_dir / "pairwise_svm.joblib")
    pw_scaler = joblib.load(model_dir / "pairwise_scaler.joblib")
    pw_imputer = joblib.load(model_dir / "pairwise_imputer.joblib")
    X_val_classical = pw_scaler.transform(pw_imputer.transform(val_classical[classical_cols].values))
    X_test_classical = pw_scaler.transform(pw_imputer.transform(test_classical[classical_cols].values))
    rf_scores_val = rf.predict_proba(X_val_classical)[:, 1]
    rf_scores_test = rf.predict_proba(X_test_classical)[:, 1]
    svm_scores_val = svm.decision_function(X_val_classical)
    svm_scores_test = svm.decision_function(X_test_classical)

    # ----- Load Siamese ensemble -----
    siamese_models, embedding_dim = combined_mod.load_siamese_ensemble(model_dir, device)
    print(f"Da load {len(siamese_models)} model Siamese.")

    def siamese_scores(pairs_df):
        _, scores = siamese_mod.compute_ensemble_scores(
            siamese_models,
            torch.utils.data.DataLoader(siamese_mod.SignaturePairDataset(pairs_df, augment=False), batch_size=32, shuffle=False),
            device,
        )
        return scores

    siamese_scores_val = siamese_scores(pairs_val)
    siamese_scores_test = siamese_scores(pairs_test)

    # ----- Load Combined model -----
    combined_rf = joblib.load(model_dir / "combined_rf.joblib")
    combined_scaler = joblib.load(model_dir / "combined_scaler.joblib")
    combined_imputer = joblib.load(model_dir / "combined_imputer.joblib")

    print("Dang tinh embedding delta cho model Combined (VAL)...")
    val_emb_deltas = combined_mod.compute_embedding_deltas(pairs_val, siamese_models, device)
    print("Dang tinh embedding delta cho model Combined (TEST)...")
    test_emb_deltas = combined_mod.compute_embedding_deltas(pairs_test, siamese_models, device)

    emb_cols = [f"emb_delta_{i}" for i in range(embedding_dim)]
    val_combined_df = val_classical.copy()
    test_combined_df = test_classical.copy()
    for i, col in enumerate(emb_cols):
        val_combined_df[col] = val_emb_deltas[:, i]
        test_combined_df[col] = test_emb_deltas[:, i]
    combined_feature_cols = classical_cols + emb_cols

    X_val_combined = combined_scaler.transform(combined_imputer.transform(val_combined_df[combined_feature_cols].values))
    X_test_combined = combined_scaler.transform(combined_imputer.transform(test_combined_df[combined_feature_cols].values))
    combined_scores_val = combined_rf.predict_proba(X_val_combined)[:, 1]
    combined_scores_test = combined_rf.predict_proba(X_test_combined)[:, 1]

    # ----- So sanh tai tung muc FAR muc tieu -----
    models_scores = {
        "rf": (rf_scores_val, rf_scores_test),
        "svm": (svm_scores_val, svm_scores_test),
        "siamese": (siamese_scores_val, siamese_scores_test),
        "combined": (combined_scores_val, combined_scores_test),
    }

    results = []
    for target_far in TARGET_FAR_LEVELS:
        print(f"\n=== Muc FAR muc tieu: {target_far:.0%} ===")
        for model_name, (scores_val, scores_test) in models_scores.items():
            threshold, achieved_far_val, _ = find_threshold_at_target_far(y_val, scores_val, target_far)
            y_pred_test = (scores_test >= threshold).astype(int)
            far_test, frr_test, acc_test = compute_far_frr_accuracy(y_test, y_pred_test)
            results.append({
                "target_FAR": target_far, "model": model_name,
                "FAR_val_dat_duoc": achieved_far_val, "threshold": threshold,
                "FAR_test": far_test, "FRR_test": frr_test, "accuracy_test": acc_test,
            })
            print(f"  {model_name:10s} -> FAR(test)={far_test:.4f} | FRR(test)={frr_test:.4f} | acc={acc_test:.4f}")

    results_df = pd.DataFrame(results)
    tables_dir = Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_path = tables_dir / "threshold_analysis_target_far.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nDa luu: {results_path}")

    # ----- Bieu do: FRR vs target FAR, tung model 1 duong -----
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    for model_name in models_scores.keys():
        sub = results_df[results_df["model"] == model_name].sort_values("target_FAR")
        plt.plot(sub["target_FAR"], sub["FRR_test"], marker="o", label=model_name.upper())
    plt.xlabel("Muc FAR muc tieu (ngan hang dat ra)")
    plt.ylabel("FRR tren test (o muc FAR do)")
    plt.title("So sanh FRR cua 4 model tai CUNG muc FAR muc tieu")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig_path = figures_dir / "threshold_analysis_frr_vs_far.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Da luu bieu do: {fig_path}")

    print("\n=== KET LUAN ===")
    print("Model nao co FRR THAP NHAT tai CUNG muc FAR la model TOT HON")
    print("cho ngan hang O MUC RUI RO DO - day la cach so sanh cong bang,")
    print("khac voi so sanh tai threshold EER rieng cua tung model.")


if __name__ == "__main__":
    main()