"""
11_optimize_accuracy.py
------------------------
Tim MODEL + NGUONG cho ACCURACY CAO NHAT (uu tien quyet dinh Khop/Khong
khop dung nhieu nhat co the), thay vi nguong EER (can bang FAR~FRR) hay
nguong theo FAR muc tieu.

KY LUAT CHONG RO RI (giu nguyen nhu toan bo do an):
    - Nguong cho tung model duoc quet va chon TREN TAP VALIDATION.
    - Model tot nhat cung duoc chon theo accuracy TREN VALIDATION.
    - Tap test chi duoc danh gia MOT LAN de bao cao, khong dung de chon.

Ket qua duoc ghi vao model_artifacts/decision_config.json. App (app.py)
tu dong doc file nay va dung dung model + nguong do, khong can sua code.

Cach dung:
    cd src
    python 11_optimize_accuracy.py

Sau do KHOI DONG LAI app de ap dung.
"""

import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

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


siamese_mod = load_module("06_siamese_network.py", "siamese_mod_opt")
combined_mod = load_module("08_combined_model.py", "combined_mod_opt")
SELECTED_FEATURES = siamese_mod.SELECTED_FEATURES


def best_threshold_by_accuracy(y_true, scores):
    """
    Quet toan bo nguong kha di (moi diem giua 2 score lien tiep) va chon
    nguong cho ACCURACY cao nhat. Khac voi EER (can bang 2 loai loi) -
    o day chi quan tam tong so quyet dinh DUNG nhieu nhat.
    """
    order = np.argsort(scores)
    s_sorted = scores[order]
    y_sorted = y_true[order]

    n = len(y_true)
    n_pos = int(y_true.sum())

    # Voi nguong t: du doan 1 neu score >= t. Quet tu thap den cao.
    # tp(k) = so nhan 1 trong phan duoi index k tro len
    tp = n_pos - np.concatenate([[0], np.cumsum(y_sorted)])[:-1] if n else np.array([])
    # Tinh lai ro rang, don gian va an toan hon:
    best_acc, best_thr = -1.0, None
    candidates = np.unique(s_sorted)
    # Them mot nguong thap hon tat ca (du doan tat ca la 1)
    candidates = np.concatenate([[candidates[0] - 1e-6], candidates])
    for t in candidates:
        pred = (scores >= t).astype(int)
        acc = (pred == y_true).mean()
        if acc > best_acc:
            best_acc, best_thr = acc, float(t)
    return best_thr, best_acc


def metrics_at(y_true, scores, threshold):
    pred = (scores >= threshold).astype(int)
    tp = int(((y_true == 1) & (pred == 1)).sum())
    tn = int(((y_true == 0) & (pred == 0)).sum())
    fp = int(((y_true == 0) & (pred == 1)).sum())
    fn = int(((y_true == 1) & (pred == 0)).sum())
    return {
        "accuracy": (tp + tn) / len(y_true),
        "FAR": fp / (fp + tn) if (fp + tn) else np.nan,
        "FRR": fn / (fn + tp) if (fn + tp) else np.nan,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features_df = pd.read_csv(FEATURES_PATH)
    pairs_val = pd.read_csv(PAIRS_DIR / "pairs_val.csv")
    pairs_test = pd.read_csv(PAIRS_DIR / "pairs_test.csv")

    val_classical, classical_cols = combined_mod.compute_classical_deltas(pairs_val, features_df, SELECTED_FEATURES)
    test_classical, _ = combined_mod.compute_classical_deltas(pairs_test, features_df, SELECTED_FEATURES)
    y_val = val_classical["label"].values
    y_test = test_classical["label"].values

    scores = {}  # model -> (scores_val, scores_test)

    # ----- RF / SVM -----
    rf = joblib.load(MODEL_DIR / "pairwise_rf.joblib")
    svm = joblib.load(MODEL_DIR / "pairwise_svm.joblib")
    pw_scaler = joblib.load(MODEL_DIR / "pairwise_scaler.joblib")
    pw_imputer = joblib.load(MODEL_DIR / "pairwise_imputer.joblib")
    Xv = pw_scaler.transform(pw_imputer.transform(val_classical[classical_cols].values))
    Xt = pw_scaler.transform(pw_imputer.transform(test_classical[classical_cols].values))
    scores["rf"] = (rf.predict_proba(Xv)[:, 1], rf.predict_proba(Xt)[:, 1])
    scores["svm"] = (svm.decision_function(Xv), svm.decision_function(Xt))

    # ----- Siamese ensemble -----
    siamese_models, embedding_dim = combined_mod.load_siamese_ensemble(MODEL_DIR, device)
    from torch.utils.data import DataLoader

    def siamese_scores(pairs_df):
        loader = DataLoader(siamese_mod.SignaturePairDataset(pairs_df, augment=False), batch_size=32, shuffle=False)
        _, sc = siamese_mod.compute_ensemble_scores(siamese_models, loader, device)
        return sc

    print(f"Da load {len(siamese_models)} model Siamese. Dang tinh diem...")
    scores["siamese"] = (siamese_scores(pairs_val), siamese_scores(pairs_test))

    # ----- Combined (neu co) -----
    if (MODEL_DIR / "combined_rf.joblib").exists():
        combined_rf = joblib.load(MODEL_DIR / "combined_rf.joblib")
        c_scaler = joblib.load(MODEL_DIR / "combined_scaler.joblib")
        c_imputer = joblib.load(MODEL_DIR / "combined_imputer.joblib")
        print("Dang tinh embedding cho Combined (VAL)...")
        ev = combined_mod.compute_embedding_deltas(pairs_val, siamese_models, device)
        print("Dang tinh embedding cho Combined (TEST)...")
        et = combined_mod.compute_embedding_deltas(pairs_test, siamese_models, device)
        emb_cols = [f"emb_delta_{i}" for i in range(embedding_dim)]
        vdf, tdf = val_classical.copy(), test_classical.copy()
        for i, c in enumerate(emb_cols):
            vdf[c], tdf[c] = ev[:, i], et[:, i]
        cols = classical_cols + emb_cols
        Xcv = c_scaler.transform(c_imputer.transform(vdf[cols].values))
        Xct = c_scaler.transform(c_imputer.transform(tdf[cols].values))
        scores["combined"] = (combined_rf.predict_proba(Xcv)[:, 1], combined_rf.predict_proba(Xct)[:, 1])

    # ----- Chon nguong toi da accuracy tren VAL, danh gia tren TEST -----
    rows = []
    for name, (sv, stt) in scores.items():
        thr, val_acc = best_threshold_by_accuracy(y_val, sv)
        m = metrics_at(y_test, stt, thr)
        rows.append({
            "model": name, "threshold": thr,
            "accuracy_val": val_acc,
            "accuracy_test": m["accuracy"], "FAR_test": m["FAR"], "FRR_test": m["FRR"],
        })

    df = pd.DataFrame(rows).sort_values("accuracy_val", ascending=False).reset_index(drop=True)
    print("\n=== NGUONG TOI UU ACCURACY (chon tren VAL, bao cao tren TEST) ===")
    print(df.round(4).to_string(index=False))

    best = df.iloc[0]
    config = {
        "model": best["model"],
        "threshold": float(best["threshold"]),
        "selection": "max_accuracy_on_validation",
        "accuracy_val": float(best["accuracy_val"]),
        "accuracy_test": float(best["accuracy_test"]),
        "FAR_test": float(best["FAR_test"]),
        "FRR_test": float(best["FRR_test"]),
    }
    out = MODEL_DIR / "decision_config.json"
    with open(out, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n>>> Model duoc chon: {best['model'].upper()} | nguong = {best['threshold']:.4f}")
    print(f">>> Accuracy tren test: {best['accuracy_test']:.4f} "
          f"(FAR={best['FAR_test']:.4f}, FRR={best['FRR_test']:.4f})")
    print(f"Da ghi: {out}")
    print("KHOI DONG LAI app de ap dung.")

    out_table = PROJECT_ROOT / "results" / "tables" / "accuracy_optimized_thresholds.csv"
    out_table.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_table, index=False)
    print(f"Da luu bang: {out_table}")


if __name__ == "__main__":
    main()