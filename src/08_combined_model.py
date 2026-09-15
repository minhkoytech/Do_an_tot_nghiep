"""
08_combined_model.py
---------------------
Model "Ket hop" - dong thu 3 trong bang phuong phap cua de cuong (Buoc 5):

    Phuong phap       | Du lieu dau vao                  | Mo hinh su dung
    Ket hop            Dac trung thu cong + Embedding CNN   Mo hinh ket hop (stacking/voting)

Phuong phap: STACKING. Ghep [11 dac trung delta thu cong da chon loc
(xem 05_pairwise_baseline.py)] VOI [vector delta embedding tu Siamese
Network da train (06_siamese_network.py, trung binh ca ensemble)]
thanh MOT vector dac trung duy nhat, roi train mot RF meta-classifier
tren vector ket hop nay.

Y tuong: dac trung thu cong (Hu Moments, GLCM...) nam bat cac dac diem
HINH HOC/THONG KE tuong minh, de giai thich. Embedding CNN nam bat cac
hoa tiet/mo hinh phuc tap hon ma dac trung thu cong khong the bieu dien
duoc. Ket hop ca hai co the tan dung diem manh cua moi huong.

Ky luat train/val/test: GIONG HET 05_pairwise_baseline.py - fit tren
train, chon hyperparameter + threshold bang EER tren val, test CHI
danh gia DUY NHAT 1 LAN.

YEU CAU: phai chay xong 03_features.py VA 06_siamese_network.py truoc
(can co san features.csv va model_artifacts/siamese_model_*.pt).

Cach dung (khong can tham so, path da khop san):
    cd src
    python 08_combined_model.py

Output:
    model_artifacts/combined_rf.joblib
    model_artifacts/combined_scaler.joblib
    model_artifacts/combined_imputer.joblib
    model_artifacts/combined_threshold.json
    results/tables/combined_test_summary.csv
    results/tables/combined_per_pairtype.csv
    results/tables/combined_per_writer.csv
    results/tables/final_model_comparison_with_combined.csv  (ca 4 model)
    results/figures/final_model_comparison_with_combined_roc.png
"""

import argparse
import importlib.util
import json
import re
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_curve, roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURES = PROJECT_ROOT / "data" / "processed" / "features.csv"
DEFAULT_PAIRS_DIR = PROJECT_ROOT / "data" / "processed" / "pairs"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "model_artifacts"
DEFAULT_TABLES_DIR = PROJECT_ROOT / "results" / "tables"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
RANDOM_SEED = 42


def load_module(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, SRC_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


siamese_mod = load_module("06_siamese_network.py", "siamese_mod_combined")
SELECTED_FEATURES = siamese_mod.SELECTED_FEATURES


# ---------------------------------------------------------------------------
# Dac trung thu cong (delta) - tai su dung dung logic tu 05/06
# ---------------------------------------------------------------------------
def compute_classical_deltas(pairs_df, features_df, feature_cols):
    feat_indexed = features_df.set_index("path")[feature_cols]
    merged = pairs_df.merge(feat_indexed.add_suffix("_ref"), left_on="reference_path", right_index=True, how="inner")
    merged = merged.merge(feat_indexed.add_suffix("_query"), left_on="query_path", right_index=True, how="inner")
    for col in feature_cols:
        merged[f"delta_{col}"] = (merged[f"{col}_ref"] - merged[f"{col}_query"]).abs()
    delta_cols = [f"delta_{c}" for c in feature_cols]
    return merged, delta_cols


# ---------------------------------------------------------------------------
# Embedding CNN (delta) - dung ensemble Siamese da train
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_embedding_for_image(models, image_path, device):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)
    embeddings = [model.embedding_net(tensor).cpu().numpy().flatten() for model in models]
    return np.mean(embeddings, axis=0)  # trung binh embedding qua ca ensemble


def compute_embedding_deltas(pairs_df, models, device):
    """Tinh |embedding_ref - embedding_query| cho tung cap, cache embedding theo duong dan anh de khong tinh lai."""
    unique_paths = pd.unique(pd.concat([pairs_df["reference_path"], pairs_df["query_path"]]))
    print(f"Tinh embedding cho {len(unique_paths)} anh (dung {len(models)} model trong ensemble)...")
    embedding_cache = {}
    for i, path in enumerate(unique_paths):
        embedding_cache[path] = compute_embedding_for_image(models, path, device)
        if (i + 1) % 200 == 0:
            print(f"  ... {i+1}/{len(unique_paths)}")

    deltas = np.array([
        np.abs(embedding_cache[row.reference_path] - embedding_cache[row.query_path])
        for row in pairs_df.itertuples()
    ])
    return deltas


def load_siamese_ensemble(model_dir: Path, device):
    with open(model_dir / "siamese_threshold.json") as f:
        config = json.load(f)
    embedding_dim = config.get("embedding_dim", 64)

    models = []
    model_files = sorted(model_dir.glob("siamese_model_*.pt"))
    if not model_files:
        legacy = model_dir / "siamese_best.pt"
        if legacy.exists():
            model_files = [legacy]
    for path in model_files:
        model = siamese_mod.SiameseNetwork(embedding_dim=embedding_dim, backbone_type="custom").to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.eval()
        models.append(model)
    return models, embedding_dim


# ---------------------------------------------------------------------------
# EER / FAR / FRR - giong het cac script truoc, giu nhat quan
# ---------------------------------------------------------------------------
def find_eer_threshold(y_true, scores):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    frr = 1 - tpr
    idx = np.argmin(np.abs(fpr - frr))
    return thresholds[idx], (fpr[idx] + frr[idx]) / 2


def compute_far_frr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    frr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    return far, frr


def evaluate_on_test(y_test, scores_test, threshold, test_merged_df, model_name):
    y_pred = (scores_test >= threshold).astype(int)
    far, frr = compute_far_frr(y_test, y_pred)
    overall = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, scores_test),
        "FAR": far, "FRR": frr, "threshold": threshold,
    }

    eval_df = test_merged_df.copy()
    eval_df["y_pred"] = y_pred
    per_pairtype = []
    for pt in ["skilled_forgery", "random_forgery"]:
        mask = eval_df["pair_type"] == pt
        if mask.sum() == 0:
            continue
        per_pairtype.append({"model": model_name, "pair_type": pt, "n": int(mask.sum()),
                              "FAR": float(np.mean(eval_df.loc[mask, "y_pred"] == 1))})
    mask = eval_df["pair_type"] == "genuine_genuine"
    if mask.sum() > 0:
        per_pairtype.append({"model": model_name, "pair_type": "genuine_genuine", "n": int(mask.sum()),
                              "FRR": float(np.mean(eval_df.loc[mask, "y_pred"] == 0))})

    eval_df["correct"] = (eval_df["y_pred"] == eval_df["label"]).astype(int)
    per_writer = []
    for writer_id, group in eval_df.groupby("writer_id_ref"):
        w_far, w_frr = compute_far_frr(group["label"].values, group["y_pred"].values)
        per_writer.append({"model": model_name, "writer_id": writer_id, "n_pairs": len(group),
                            "accuracy": group["correct"].mean(), "FAR": w_far, "FRR": w_frr})

    return overall, pd.DataFrame(per_pairtype), pd.DataFrame(per_writer)


def main():
    parser = argparse.ArgumentParser(description="Model Ket hop: dac trung thu cong + CNN embedding (stacking)")
    parser.add_argument("--features", default=str(DEFAULT_FEATURES))
    parser.add_argument("--pairs_dir", default=str(DEFAULT_PAIRS_DIR))
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--tables_dir", default=str(DEFAULT_TABLES_DIR))
    parser.add_argument("--figures_dir", default=str(DEFAULT_FIGURES_DIR))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dang su dung device: {device}")

    features_df = pd.read_csv(args.features)
    pairs_dir = Path(args.pairs_dir)
    pairs_train = pd.read_csv(pairs_dir / "pairs_train.csv")
    pairs_val = pd.read_csv(pairs_dir / "pairs_val.csv")
    pairs_test = pd.read_csv(pairs_dir / "pairs_test.csv")

    model_dir = Path(args.model_dir)
    siamese_models, embedding_dim = load_siamese_ensemble(model_dir, device)
    print(f"Da load {len(siamese_models)} model Siamese (embedding_dim={embedding_dim}).")

    # ----- Dac trung thu cong (delta) -----
    train_merged, classical_cols = compute_classical_deltas(pairs_train, features_df, SELECTED_FEATURES)
    val_merged, _ = compute_classical_deltas(pairs_val, features_df, SELECTED_FEATURES)
    test_merged, _ = compute_classical_deltas(pairs_test, features_df, SELECTED_FEATURES)

    # ----- Embedding CNN (delta) -----
    print("\n=== Tinh embedding delta cho TRAIN ===")
    train_emb_deltas = compute_embedding_deltas(pairs_train, siamese_models, device)
    print("\n=== Tinh embedding delta cho VAL ===")
    val_emb_deltas = compute_embedding_deltas(pairs_val, siamese_models, device)
    print("\n=== Tinh embedding delta cho TEST ===")
    test_emb_deltas = compute_embedding_deltas(pairs_test, siamese_models, device)

    emb_cols = [f"emb_delta_{i}" for i in range(embedding_dim)]
    for i, col in enumerate(emb_cols):
        train_merged[col] = train_emb_deltas[:, i]
        val_merged[col] = val_emb_deltas[:, i]
        test_merged[col] = test_emb_deltas[:, i]

    combined_cols = classical_cols + emb_cols
    print(f"\nTong so dac trung ket hop: {len(combined_cols)} "
          f"({len(classical_cols)} thu cong + {len(emb_cols)} embedding CNN)")

    # ----- Impute + Scale (fit CHI tren train) -----
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(train_merged[combined_cols].values)
    X_val = imputer.transform(val_merged[combined_cols].values)
    X_test = imputer.transform(test_merged[combined_cols].values)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    y_train = train_merged["label"].values
    y_val = val_merged["label"].values
    y_test = test_merged["label"].values

    # ----- Chon hyperparameter RF tren VAL (giong 05_pairwise_baseline.py) -----
    print("\n=== Huan luyen RF meta-classifier (chon hyperparameter tren VAL) ===")
    param_grid = [
        {"n_estimators": 200, "max_depth": None},
        {"n_estimators": 200, "max_depth": 10},
        {"n_estimators": 400, "max_depth": 20},
    ]
    best_model, best_eer, best_params = None, np.inf, None
    for params in param_grid:
        model = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1, **params)
        model.fit(X_train, y_train)
        scores_val = model.predict_proba(X_val)[:, 1]
        _, eer_value = find_eer_threshold(y_val, scores_val)
        print(f"  params={params} -> EER(val)={eer_value:.4f}")
        if eer_value < best_eer:
            best_model, best_eer, best_params = model, eer_value, params
    print(f"  >>> Best: params={best_params}, EER(val)={best_eer:.4f}")

    scores_val = best_model.predict_proba(X_val)[:, 1]
    threshold, eer_val = find_eer_threshold(y_val, scores_val)
    print(f"\nThreshold (tu val, EER={eer_val:.4f}): {threshold:.4f}")

    # ----- Danh gia TREN TEST - CHI MOT LAN -----
    print("\n=== Danh gia model Ket hop tren TEST (mot lan duy nhat) ===")
    scores_test = best_model.predict_proba(X_test)[:, 1]
    overall, per_pairtype, per_writer = evaluate_on_test(y_test, scores_test, threshold, test_merged, "combined")

    tables_dir = Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([overall]).to_csv(tables_dir / "combined_test_summary.csv", index=False)
    per_pairtype.to_csv(tables_dir / "combined_per_pairtype.csv", index=False)
    per_writer.to_csv(tables_dir / "combined_per_writer.csv", index=False)

    print(pd.DataFrame([overall]).to_string(index=False))
    print(f"\n=== FAR/FRR theo loai gia mao (Combined) ===\n{per_pairtype.to_string(index=False)}")

    # ----- Luu model artifacts -----
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_dir / "combined_rf.joblib")
    joblib.dump(scaler, model_dir / "combined_scaler.joblib")
    joblib.dump(imputer, model_dir / "combined_imputer.joblib")
    with open(model_dir / "combined_threshold.json", "w") as f:
        json.dump({"threshold": float(threshold), "rf_params": best_params,
                    "classical_features": classical_cols, "embedding_dim": embedding_dim}, f, indent=2)

    # ----- So sanh voi 3 model truoc (RF/SVM/Siamese) neu co san -----
    prior_comparison_path = tables_dir / "final_model_comparison.csv"
    combined_scores_dict = {"combined": (y_test, scores_test)}
    all_rows = [overall]
    if prior_comparison_path.exists():
        prior_df = pd.read_csv(prior_comparison_path)
        all_rows = list(prior_df.to_dict("records")) + [overall]
        print(f"\nDa tim thay ket qua 3 model truoc do ({prior_comparison_path}), gop chung vao bang so sanh.")

    final_df = pd.DataFrame(all_rows)
    final_path = tables_dir / "final_model_comparison_with_combined.csv"
    final_df.to_csv(final_path, index=False)
    print(f"\n=== BANG SO SANH CUOI CUNG (4 MODEL) ===\n{final_df.to_string(index=False)}")
    print(f"Da luu: {final_path}")

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6.5, 6.5))
    fpr, tpr, _ = roc_curve(y_test, scores_test)
    plt.plot(fpr, tpr, label=f"COMBINED (AUC={roc_auc_score(y_test, scores_test):.3f})", linewidth=2)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("FAR (False Acceptance Rate)")
    plt.ylabel("1 - FRR (True Accept Rate)")
    plt.title("ROC - Model Ket hop (Classical features + CNN embedding)")
    plt.legend()
    plt.tight_layout()
    fig_path = figures_dir / "combined_model_roc.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Da luu bieu do: {fig_path}")

    print(f"\nHoan tat. Model Ket hop da luu tai: {model_dir / 'combined_rf.joblib'}")


if __name__ == "__main__":
    main()