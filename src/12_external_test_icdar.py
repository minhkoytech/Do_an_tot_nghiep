"""
12_external_test_icdar.py
--------------------------
External test: danh gia model DA TRAIN TREN CEDAR len bo du lieu KHAC
(ICDAR SigComp 2011) de do KHA NANG TONG QUAT HOA - muc 5.1 trong de
cuong, va la viec GVHD xep uu tien thap ("neu co thoi gian").

QUAN TRONG VE PHUONG PHAP LUAN:
    - KHONG train lai, KHONG chon lai nguong tren ICDAR. Dung nguyen
      model + nguong da co dinh tu CEDAR. Neu chinh nguong theo ICDAR
      thi khong con la "external test" nua.
    - Ket qua thuong THAP HON tren CEDAR do khac biet nguon du lieu
      (domain shift): khac nguoi viet, khac do phan giai quet (400 dpi
      so voi 300 dpi), khac ngon ngu. Day la ket qua BINH THUONG va can
      duoc bao cao trung thuc.

CACH DUNG - 2 BUOC:

  Buoc 1 (BAT BUOC lam truoc): do cau truc thu muc
      python 12_external_test_icdar.py --data_dir "D:/duong/dan/SigComp2011" --inspect
  Script se in ra cay thu muc, so luong anh, vai ten file mau. Gui ket
  qua nay de xac dinh dung tham so cho buoc 2.

  Buoc 2: chay danh gia
      python 12_external_test_icdar.py --data_dir "..." \
          --genuine_keywords genuine reference \
          --forged_keywords forg questioned \
          --writer_regex "(\\d{3})\\d+"

Tham so:
  --genuine_keywords : tu khoa trong DUONG DAN cho biet anh la chu ky THAT
  --forged_keywords  : tu khoa cho biet anh la chu ky GIA
  --writer_regex     : regex lay ID nguoi ky tu TEN FILE (nhom 1)
"""

import argparse
import importlib.util
import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
MODEL_DIR = PROJECT_ROOT / "model_artifacts"
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def load_module(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, SRC_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# BUOC 1: do cau truc thu muc
# ---------------------------------------------------------------------------
def inspect_dir(data_dir: Path):
    print(f"Dang do cau truc: {data_dir}\n")
    by_folder = defaultdict(list)
    for p in data_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            by_folder[p.parent].append(p.name)

    if not by_folder:
        print("KHONG tim thay anh nao. Kiem tra lai --data_dir.")
        return

    print(f"{'Thu muc (tuong doi)':<60}{'So anh':>8}   Vai ten file mau")
    print("-" * 120)
    for folder in sorted(by_folder):
        names = sorted(by_folder[folder])
        rel = folder.relative_to(data_dir)
        print(f"{str(rel):<60}{len(names):>8}   {', '.join(names[:3])}")

    total = sum(len(v) for v in by_folder.values())
    print("-" * 120)
    print(f"Tong so anh: {total}")
    print("\nGUI KET QUA NAY DE XAC DINH THAM SO cho buoc 2 "
          "(--genuine_keywords, --forged_keywords, --writer_regex).")


# ---------------------------------------------------------------------------
# BUOC 2: xay manifest + pairs + danh gia
# ---------------------------------------------------------------------------
def build_manifest(data_dir: Path, genuine_kw, forged_kw, writer_regex,
                    writer_from="folder", forged_suffix="_forg"):
    """
    writer_from="folder" (mac dinh): lay writer_id tu TEN THU MUC cha, va
      xac dinh nhan bang hau to cua thu muc (vd "049" = that, "049_forg" =
      gia). Phu hop voi bo Kaggle sign_data (ICDAR 2011 Dutch) vi ten FILE
      o bo nay khong nhat quan: "01_049.png", "049_01.PNG",
      "0119001_01.png" - ID nam o vi tri khac nhau nen regex tren ten file
      khong dang tin.
    writer_from="filename": lay writer_id tu ten file bang writer_regex va
      xac dinh nhan bang tu khoa trong duong dan (cho cac bo khac).
    """
    rows = []
    pattern = re.compile(writer_regex)
    skipped_label, skipped_writer = 0, 0

    if writer_from == "folder":
        for p in data_dir.rglob("*"):
            if not (p.is_file() and p.suffix.lower() in IMG_EXT):
                continue
            folder = p.parent.name
            if folder.lower().endswith(forged_suffix.lower()):
                label = "forged"
                writer_id = folder[: -len(forged_suffix)]
            else:
                label = "genuine"
                writer_id = folder
            rows.append({"path": str(p), "writer_id": writer_id, "label": label})

        df = pd.DataFrame(rows)
        print(f"Da nhan dien: {len(df)} anh "
              f"({(df['label'] == 'genuine').sum()} that, {(df['label'] == 'forged').sum()} gia), "
              f"{df['writer_id'].nunique() if len(df) else 0} nguoi ky "
              f"(writer_id lay tu ten thu muc)")
        return df

    for p in data_dir.rglob("*"):
        if not (p.is_file() and p.suffix.lower() in IMG_EXT):
            continue
        path_low = str(p).lower()

        # Nhan: uu tien kiem tra "gia" truoc vi mot so bo dat ten thu muc
        # dang "Offline Forgeries" chua ca tu "offline" lan "forg"
        if any(k.lower() in path_low for k in forged_kw):
            label = "forged"
        elif any(k.lower() in path_low for k in genuine_kw):
            label = "genuine"
        else:
            skipped_label += 1
            continue

        m = pattern.search(p.stem)
        if not m:
            skipped_writer += 1
            continue
        writer_id = m.group(1)
        rows.append({"path": str(p), "writer_id": writer_id, "label": label})

    df = pd.DataFrame(rows)
    print(f"Da nhan dien: {len(df)} anh "
          f"({(df['label'] == 'genuine').sum()} that, {(df['label'] == 'forged').sum()} gia), "
          f"{df['writer_id'].nunique() if len(df) else 0} nguoi ky")
    if skipped_label:
        print(f"  Bo qua {skipped_label} anh (khong khop tu khoa that/gia)")
    if skipped_writer:
        print(f"  Bo qua {skipped_writer} anh (khong lay duoc writer_id tu ten file)")
    return df


def build_pairs(manifest, max_pos=40, max_skilled=40, n_random=20, seed=42):
    rng = np.random.RandomState(seed)
    pairs = []
    writers = sorted(manifest["writer_id"].unique())

    for w in writers:
        gen = manifest[(manifest.writer_id == w) & (manifest.label == "genuine")]["path"].tolist()
        forg = manifest[(manifest.writer_id == w) & (manifest.label == "forged")]["path"].tolist()
        if len(gen) < 2:
            continue

        pos = list(combinations(gen, 2))
        rng.shuffle(pos)
        for a, b in pos[:max_pos]:
            pairs.append({"reference_path": a, "query_path": b, "label": 1,
                          "pair_type": "genuine_genuine", "writer_id_ref": w})

        if forg:
            neg = [(a, b) for a in gen for b in forg]
            rng.shuffle(neg)
            for a, b in neg[:max_skilled]:
                pairs.append({"reference_path": a, "query_path": b, "label": 0,
                              "pair_type": "skilled_forgery", "writer_id_ref": w})

        others = manifest[(manifest.writer_id != w) & (manifest.label == "genuine")]["path"].tolist()
        for _ in range(min(n_random, len(others))):
            pairs.append({"reference_path": gen[rng.randint(len(gen))],
                          "query_path": others[rng.randint(len(others))], "label": 0,
                          "pair_type": "random_forgery", "writer_id_ref": w})

    return pd.DataFrame(pairs)


def metrics(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {
        "accuracy": (tp + tn) / len(y_true),
        "FAR": fp / (fp + tn) if (fp + tn) else np.nan,
        "FRR": fn / (fn + tp) if (fn + tp) else np.nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--inspect", action="store_true", help="Chi do cau truc thu muc, khong danh gia")
    ap.add_argument("--genuine_keywords", nargs="+", default=["genuine", "reference"])
    ap.add_argument("--forged_keywords", nargs="+", default=["forg", "questioned"])
    ap.add_argument("--writer_regex", default=r"(\d{3})")
    ap.add_argument("--writer_from", choices=["folder", "filename"], default="folder",
                    help="Lay writer_id tu ten THU MUC (mac dinh, dung cho bo Kaggle sign_data) "
                         "hay tu TEN FILE bang --writer_regex")
    ap.add_argument("--forged_suffix", default="_forg",
                    help="Hau to thu muc danh dau chu ky gia (dung khi --writer_from folder)")
    ap.add_argument("--max_pos", type=int, default=40)
    ap.add_argument("--max_skilled", type=int, default=40)
    ap.add_argument("--n_random", type=int, default=20)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Khong tim thay: {data_dir}")

    if args.inspect:
        inspect_dir(data_dir)
        return

    preprocess_mod = load_module("01_preprocessing.py", "pre_ext")
    features_mod = load_module("03_features.py", "feat_ext")
    siamese_mod = load_module("06_siamese_network.py", "siam_ext")
    combined_mod = load_module("08_combined_model.py", "comb_ext")
    SELECTED = siamese_mod.SELECTED_FEATURES

    manifest = build_manifest(data_dir, args.genuine_keywords, args.forged_keywords,
                               args.writer_regex, args.writer_from, args.forged_suffix)
    if len(manifest) < 10:
        print("Qua it anh nhan dien duoc. Chay lai voi --inspect de kiem tra tham so.")
        return

    pairs = build_pairs(manifest, args.max_pos, args.max_skilled, args.n_random)
    print(f"\nDa tao {len(pairs)} cap:")
    print(pairs["pair_type"].value_counts().to_string())

    # Tien xu ly + trich dac trung (cache theo duong dan anh)
    out_dir = PROJECT_ROOT / "data" / "processed" / "icdar_external"
    out_dir.mkdir(parents=True, exist_ok=True)
    import cv2
    cache = {}
    uniq = pd.unique(pd.concat([pairs.reference_path, pairs.query_path]))
    print(f"\nDang tien xu ly + trich dac trung {len(uniq)} anh...")
    for i, src in enumerate(uniq):
        dst = out_dir / f"img_{i}.png"
        cv2.imwrite(str(dst), preprocess_mod.preprocess_image(src))
        cache[src] = str(dst)
        if (i + 1) % 200 == 0:
            print(f"  ... {i + 1}/{len(uniq)}")

    feats = {src: features_mod.extract_all_features(dst) for src, dst in cache.items()}

    X_classical = np.array([
        [abs(feats[r.reference_path][f] - feats[r.query_path][f]) for f in SELECTED]
        for r in pairs.itertuples()
    ])
    y = pairs["label"].values

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    siamese_models, emb_dim = combined_mod.load_siamese_ensemble(MODEL_DIR, device)

    # Nguong CO DINH tu CEDAR
    cfg_path = MODEL_DIR / "decision_config.json"
    cfg = json.load(open(cfg_path)) if cfg_path.exists() else None

    results = []

    # --- RF / SVM ---
    rf = joblib.load(MODEL_DIR / "pairwise_rf.joblib")
    svm = joblib.load(MODEL_DIR / "pairwise_svm.joblib")
    sc = joblib.load(MODEL_DIR / "pairwise_scaler.joblib")
    im = joblib.load(MODEL_DIR / "pairwise_imputer.joblib")
    thr_pw = json.load(open(MODEL_DIR / "pairwise_threshold.json"))
    Xp = sc.transform(im.transform(X_classical))
    for name, model, key in [("rf", rf, "rf_threshold"), ("svm", svm, "svm_threshold")]:
        s = model.predict_proba(Xp)[:, 1] if name == "rf" else model.decision_function(Xp)
        t = cfg["threshold"] if (cfg and cfg["model"] == name) else thr_pw[key]
        results.append({"model": name, "threshold": t, **metrics(y, (s >= t).astype(int))})

    # --- Siamese ---
    from torch.utils.data import DataLoader
    pairs_cached = pairs.copy()
    pairs_cached["reference_path"] = pairs_cached["reference_path"].map(cache)
    pairs_cached["query_path"] = pairs_cached["query_path"].map(cache)
    loader = DataLoader(siamese_mod.SignaturePairDataset(pairs_cached, augment=False), batch_size=32, shuffle=False)
    _, s_siam = siamese_mod.compute_ensemble_scores(siamese_models, loader, device)
    t = cfg["threshold"] if (cfg and cfg["model"] == "siamese") else json.load(open(MODEL_DIR / "siamese_threshold.json"))["threshold"]
    results.append({"model": "siamese", "threshold": t, **metrics(y, (s_siam >= t).astype(int))})

    # --- Combined ---
    if (MODEL_DIR / "combined_rf.joblib").exists():
        crf = joblib.load(MODEL_DIR / "combined_rf.joblib")
        csc = joblib.load(MODEL_DIR / "combined_scaler.joblib")
        cim = joblib.load(MODEL_DIR / "combined_imputer.joblib")
        emb_d = combined_mod.compute_embedding_deltas(pairs_cached, siamese_models, device)
        Xc = csc.transform(cim.transform(np.hstack([X_classical, emb_d])))
        s_c = crf.predict_proba(Xc)[:, 1]
        t = cfg["threshold"] if (cfg and cfg["model"] == "combined") else json.load(open(MODEL_DIR / "combined_threshold.json"))["threshold"]
        results.append({"model": "combined", "threshold": t, **metrics(y, (s_c >= t).astype(int))})

    df = pd.DataFrame(results)
    print("\n=== KET QUA EXTERNAL TEST TREN ICDAR SigComp 2011 ===")
    print("(model va nguong GIU NGUYEN tu CEDAR, khong train/chinh lai)")
    print(df.round(4).to_string(index=False))

    out = PROJECT_ROOT / "results" / "tables" / "external_test_icdar.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nDa luu: {out}")


if __name__ == "__main__":
    main()