"""
07_cross_validation.py
-----------------------
Kiem tra do on dinh cua ket qua qua NHIEU CACH CHIA NGUOI KY (theo gop y
GVHD): lap lai toan bo quy trinh tren 5 cach chia signer-disjoint 35/10/10
voi 5 random seed khac nhau, roi bao cao mean +- SD cua cac chi so chinh.

NGUYEN TAC (dung gop y GVHD):
    - GIU NGUYEN kien truc va tham so da chon o lan chia chinh:
        RF 200 cay do sau 10, SVM RBF C=10, Siamese Triplet Loss voi margin
        rieng cho gia co ky nang, ensemble 3 mang, mo hinh ket hop RF 400
        cay do sau 20, 11 dac trung da chon. KHONG do tim lai tham so.
    - Moi lan chia chi chon nguong tren validation cua chinh lan chia do.
    - Muc dich chi la kiem tra ket luan hien tai co on dinh hay khong.

AN TOAN: moi seed ghi vao THU MUC RIENG, khong ghi de mo hinh va ket qua
chinh dang dung cho bao cao va ung dung web:
    data/processed/cv/seed_<s>/pairs/
    model_artifacts/cv/seed_<s>/
    results/cv/seed_<s>/tables/ va figures/

Co the chay lai neu bi ngat giua chung: seed nao da xong se duoc bo qua.

Cach dung:
    cd src
    python 07_cross_validation.py
    python 07_cross_validation.py --num_models 1     # nhanh hon, bao cao ro la dung 1 mang

Ket qua tong hop:
    results/tables/cv_all_seeds.csv          (tung seed, tung mo hinh)
    results/tables/cv_summary.csv            (mean, SD tung chi so)
    results/tables/cv_target_far_summary.csv (FRR tai FAR muc tieu, mean, SD)
    results/tables/cv_summary_tables.txt     (bang san sang dua vao bao cao)
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
NAMES = {"rf": "Random Forest", "svm": "SVM", "siamese": "Siamese Network", "combined": "Mô hình kết hợp"}
ORDER = ["rf", "svm", "siamese", "combined"]


def run(script, args, label):
    cmd = [sys.executable, str(SRC / script)] + [str(a) for a in args]
    print(f"\n>>> {label}\n    {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=str(SRC))
    if r.returncode != 0:
        raise RuntimeError(f"{script} that bai (ma loi {r.returncode})")


def seed_dirs(seed):
    return {
        "pairs": ROOT / "data" / "processed" / "cv" / f"seed_{seed}" / "pairs",
        "models": ROOT / "model_artifacts" / "cv" / f"seed_{seed}",
        "tables": ROOT / "results" / "cv" / f"seed_{seed}" / "tables",
        "figures": ROOT / "results" / "cv" / f"seed_{seed}" / "figures",
    }


def run_seed(seed, args):
    d = seed_dirs(seed)
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    common = ["--pairs_dir", d["pairs"], "--model_dir", d["models"],
              "--tables_dir", d["tables"], "--figures_dir", d["figures"]]

    run("01b_generate_pairs.py", [
        "--seed", seed, "--output_dir", d["pairs"],
        "--val_writers", args.val_writers, "--test_writers", args.test_writers,
        "--max_pos_pairs_per_writer", args.max_pos,
        "--max_skilled_neg_pairs_per_writer", args.max_skilled,
        "--num_random_neg_pairs_per_writer", args.n_random,
    ], f"[seed {seed}] Chia nguoi ky va tao cap")
    run("05_pairwise_baseline.py", common + ["--fixed_params"], f"[seed {seed}] RF, SVM (tham so co dinh)")
    extra = ["--epochs", args.epochs, "--patience", args.epochs, "--batch_size", 4] if args.epochs else []
    run("06_siamese_network.py", common + ["--num_models", args.num_models] + extra,
        f"[seed {seed}] Siamese ({args.num_models} mang)")
    run("08_combined_model.py", common + ["--fixed_params"], f"[seed {seed}] Mo hinh ket hop (tham so co dinh)")
    run("10_threshold_analysis.py", common, f"[seed {seed}] EER va FRR tai FAR muc tieu")


def collect(seed):
    t = seed_dirs(seed)["tables"]
    comp = pd.read_csv(t / "final_model_comparison_with_combined.csv")[["model", "accuracy", "roc_auc", "FAR", "FRR"]]
    comp = comp.rename(columns={"FAR": "FAR_eer", "FRR": "FRR_eer"})
    eer = pd.read_csv(t / "eer_test.csv")[["model", "EER_test"]]
    df = comp.merge(eer, on="model")

    pt_frames = [pd.read_csv(t / f) for f in
                 ["pairwise_baseline_per_pairtype.csv", "siamese_per_pairtype.csv", "combined_per_pairtype.csv"]
                 if (t / f).exists()]
    if pt_frames:
        pt = pd.concat(pt_frames)
        for ptype, col in [("skilled_forgery", "FAR_skilled"), ("random_forgery", "FAR_random")]:
            sub = pt[pt["pair_type"] == ptype][["model", "FAR"]].rename(columns={"FAR": col})
            df = df.merge(sub, on="model", how="left")

    df["seed"] = seed
    far = pd.read_csv(t / "threshold_analysis_target_far.csv")
    far["seed"] = seed
    return df, far


def fmt(m, s, pct=True):
    if pct:
        return f"{m*100:.1f} ± {s*100:.1f}".replace(".", ",")
    return f"{m:.3f} ± {s:.3f}".replace(".", ",")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    ap.add_argument("--num_models", type=int, default=3,
                    help="So mang Siamese trong ensemble. 3 = giong cau hinh chinh.")
    ap.add_argument("--max_pos", type=int, default=150)
    ap.add_argument("--max_skilled", type=int, default=150)
    ap.add_argument("--n_random", type=int, default=75)
    ap.add_argument("--rerun", action="store_true", help="Chay lai ca nhung seed da xong")
    ap.add_argument("--val_writers", type=int, default=10)
    ap.add_argument("--test_writers", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=None, help="Chi dung de chay thu nhanh")
    args = ap.parse_args()

    for i, seed in enumerate(args.seeds, 1):
        done = (seed_dirs(seed)["tables"] / "eer_test.csv").exists()
        print(f"\n{'#' * 64}\n# LAN CHIA {i}/{len(args.seeds)} - seed {seed}" + ("  (da xong, bo qua)" if done and not args.rerun else "") + f"\n{'#' * 64}")
        if done and not args.rerun:
            continue
        run_seed(seed, args)

    rows, fars = zip(*[collect(s) for s in args.seeds])
    all_df = pd.concat(rows, ignore_index=True)
    far_df = pd.concat(fars, ignore_index=True)

    out = ROOT / "results" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out / "cv_all_seeds.csv", index=False)

    metrics = [c for c in ["accuracy", "roc_auc", "EER_test", "FAR_eer", "FRR_eer", "FAR_skilled", "FAR_random"] if c in all_df]
    summary = all_df.groupby("model")[metrics].agg(["mean", "std"])
    summary.to_csv(out / "cv_summary.csv")

    far_sum = far_df.groupby(["target_FAR", "model"])[["FAR_test", "FRR_test"]].agg(["mean", "std"])
    far_sum.to_csv(out / "cv_target_far_summary.csv")

    # ---------- Bang san sang dua vao bao cao ----------
    n = len(args.seeds)
    lines = [f"Bang: Ket qua trung binh ± do lech chuan qua {n} cach chia nguoi ky"]
    lines.append("| Mô hình | Accuracy (%) | ROC-AUC | EER (%) | FAR giả có kỹ năng (%) | FAR giả ngẫu nhiên (%) |")
    lines.append("|---|---|---|---|---|---|")
    for m in ORDER:
        if m not in summary.index:
            continue
        r = summary.loc[m]
        cells = [NAMES[m],
                 fmt(r[("accuracy", "mean")], r[("accuracy", "std")]),
                 fmt(r[("roc_auc", "mean")], r[("roc_auc", "std")], pct=False),
                 fmt(r[("EER_test", "mean")], r[("EER_test", "std")])]
        for c in ["FAR_skilled", "FAR_random"]:
            cells.append(fmt(r[(c, "mean")], r[(c, "std")]) if (c, "mean") in r.index else "–")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append(f"\nBang: FRR (%) tai FAR muc tieu, nguong chon tren validation, trung binh ± do lech chuan qua {n} cach chia")
    lines.append("| FAR mục tiêu | " + " | ".join(NAMES[m] for m in ORDER) + " |")
    lines.append("|---|" + "---|" * len(ORDER))
    for tf in sorted(far_df["target_FAR"].unique()):
        cells = [f"{tf*100:.0f}%"]
        for m in ORDER:
            key = (tf, m)
            if key in far_sum.index:
                r = far_sum.loc[key]
                cells.append(fmt(r[("FRR_test", "mean")], r[("FRR_test", "std")]))
            else:
                cells.append("–")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append(f"\nBang: FAR thuc te (%) tren tap kiem thu tai FAR muc tieu, trung binh ± do lech chuan")
    lines.append("| FAR mục tiêu | " + " | ".join(NAMES[m] for m in ORDER) + " |")
    lines.append("|---|" + "---|" * len(ORDER))
    for tf in sorted(far_df["target_FAR"].unique()):
        cells = [f"{tf*100:.0f}%"]
        for m in ORDER:
            key = (tf, m)
            cells.append(fmt(far_sum.loc[key][("FAR_test", "mean")], far_sum.loc[key][("FAR_test", "std")]) if key in far_sum.index else "–")
        lines.append("| " + " | ".join(cells) + " |")

    # Mo hinh nao co FRR thap nhat o tung lan chia (kiem tra ket luan co on dinh khong)
    lines.append("\nSo lan chia ma moi mo hinh co FRR thap nhat tai tung FAR muc tieu:")
    for tf in sorted(far_df["target_FAR"].unique()):
        sub = far_df[far_df["target_FAR"] == tf]
        counts, ties = {}, 0
        for _seed, g in sub.groupby("seed"):
            best = g["FRR_test"].min()
            w = g[g["FRR_test"] <= best + 1e-9]["model"].tolist()
            if len(w) > 1:
                ties += 1
            for m in w:
                counts[m] = counts.get(m, 0) + 1
        txt = ", ".join(f"{NAMES.get(m, m)} {counts[m]}/{n}" for m in ORDER if m in counts)
        lines.append(f"  FAR {tf*100:.0f}%: {txt}" + (f"  (co {ties} lan chia hoa nhau)" if ties else ""))

    text = "\n".join(lines)
    print("\n" + "=" * 64 + "\n" + text)
    (out / "cv_summary_tables.txt").write_text(text, encoding="utf-8")
    print(f"\nDa luu: {out / 'cv_summary_tables.txt'}")


if __name__ == "__main__":
    main()