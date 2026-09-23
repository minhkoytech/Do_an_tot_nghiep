"""
recompute_combined_threshold.py
---------------------------------
Tinh lai threshold cho model Combined theo MUC FAR MUC TIEU (vd 10%) thay
vi threshold EER mac dinh - giup Combined "khong tin de" nhu hien tai.

VI SAO CAN SCRIPT NAY:
    App demo dang dung threshold EER (can bang FAR~FRR) cho Combined,
    duoc luu trong model_artifacts/combined_threshold.json tu luc chay
    08_combined_model.py. Nhung 10_threshold_analysis.py da chung minh:
    Combined thang ro nhat o CAC MUC FAR THAP (5-10%) - tuc la neu chon
    threshold nghiem ngat hon (FAR muc tieu thap hon EER), Combined se
    "kho tin" hon, giam FAR, van giu FRR tot hon RF/SVM/Siamese o cung
    muc do.

Script nay tinh lai threshold tren VAL (dung nguyen tac cu, khong dung
test) cho MOT MUC FAR MUC TIEU cu the, ghi de vao combined_threshold.json
(sao luu ban cu truoc khi ghi de).

Cach dung:
    cd src
    python recompute_combined_threshold.py --target_far 0.10

Sau khi chay xong, KHOI DONG LAI app.py / backend de no doc threshold moi.
"""

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
MODEL_DIR = PROJECT_ROOT / "model_artifacts"
PAIRS_DIR = PROJECT_ROOT / "data" / "processed" / "pairs"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.csv"


def load_module(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, SRC_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_threshold_at_target_far(y_true_val, scores_val, target_far):
    """Tim threshold tren VAL sao cho FAR gan nhat voi target_far (khong vuot qua)."""
    fpr, tpr, thresholds = roc_curve(y_true_val, scores_val)
    valid_idx = np.where(fpr <= target_far)[0]
    if len(valid_idx) == 0:
        idx = np.argmin(fpr)
    else:
        idx = valid_idx[np.argmax(fpr[valid_idx])]
    return thresholds[idx], fpr[idx], 1 - tpr[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_far", type=float, default=0.10,
                         help="Muc FAR muc tieu cho Combined (mac dinh 0.10 = 10%%)")
    args = parser.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    siamese_mod = load_module("06_siamese_network.py", "siamese_mod_recompute")
    combined_mod = load_module("08_combined_model.py", "combined_mod_recompute")
    SELECTED_FEATURES = siamese_mod.SELECTED_FEATURES

    print("Dang load model Siamese ensemble va Combined...")
    siamese_models, embedding_dim = combined_mod.load_siamese_ensemble(MODEL_DIR, device)
    combined_rf = joblib.load(MODEL_DIR / "combined_rf.joblib")
    combined_scaler = joblib.load(MODEL_DIR / "combined_scaler.joblib")
    combined_imputer = joblib.load(MODEL_DIR / "combined_imputer.joblib")

    features_df = pd.read_csv(FEATURES_PATH)
    pairs_val = pd.read_csv(PAIRS_DIR / "pairs_val.csv")

    print(f"Dang tinh dac trung ket hop tren VAL ({len(pairs_val)} pairs)...")
    val_classical, classical_cols = combined_mod.compute_classical_deltas(pairs_val, features_df, SELECTED_FEATURES)
    val_emb_deltas = combined_mod.compute_embedding_deltas(pairs_val, siamese_models, device)

    emb_cols = [f"emb_delta_{i}" for i in range(embedding_dim)]
    for i, col in enumerate(emb_cols):
        val_classical[col] = val_emb_deltas[:, i]
    combined_feature_cols = classical_cols + emb_cols

    X_val = combined_scaler.transform(combined_imputer.transform(val_classical[combined_feature_cols].values))
    y_val = val_classical["label"].values
    scores_val = combined_rf.predict_proba(X_val)[:, 1]

    old_threshold_path = MODEL_DIR / "combined_threshold.json"
    with open(old_threshold_path) as f:
        old_config = json.load(f)
    old_threshold = old_config["threshold"]

    new_threshold, achieved_far, achieved_frr = find_threshold_at_target_far(y_val, scores_val, args.target_far)

    print(f"\nThreshold CU (EER):                {old_threshold:.4f}")
    print(f"Threshold MOI (FAR muc tieu={args.target_far:.0%}): {new_threshold:.4f}")
    print(f"  -> FAR dat duoc tren VAL: {achieved_far:.4f}")
    print(f"  -> FRR dat duoc tren VAL: {achieved_frr:.4f}")

    # Sao luu ban cu truoc khi ghi de
    backup_path = MODEL_DIR / "combined_threshold_backup_eer.json"
    if not backup_path.exists():
        shutil.copy(old_threshold_path, backup_path)
        print(f"\nDa sao luu threshold EER cu vao: {backup_path}")

    old_config["threshold"] = float(new_threshold)
    old_config["threshold_method"] = f"target_far_{args.target_far}"
    old_config["threshold_eer_backup"] = float(old_threshold)
    with open(old_threshold_path, "w") as f:
        json.dump(old_config, f, indent=2)

    print(f"\nDa cap nhat: {old_threshold_path}")
    print("KHOI DONG LAI app.py / backend de ap dung threshold moi.")
    print(f"\nDe quay lai threshold EER cu: copy {backup_path.name} de len combined_threshold.json")


if __name__ == "__main__":
    main()