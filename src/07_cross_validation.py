"""
07_cross_validation.py
------------------------
Tu dong chay lai toan bo pipeline (01b_generate_pairs -> 05_pairwise_baseline
-> 06_siamese_network) qua NHIEU CACH CHIA WRITER KHAC NHAU (nhieu seed),
roi tong hop ket qua TRUNG BINH +- DO LECH CHUAN.

VI SAO CAN SCRIPT NAY:
    CEDAR chi co 55 writer (10 writer o tap test). Ket qua danh gia tu
    MOT lan chia duy nhat co the dao dong dang ke chi vi "may/rui" - tap
    test lan do gom nhung writer de hay kho hon trung binh. Chay qua
    nhieu cach chia (seed) khac nhau roi lay trung binh +- std la cach
    danh gia dung dan va thuyet phuc hon nhieu so voi bao cao 1 con so
    tu 1 lan chay - va la thuc hanh chuan trong nghien cuu (k-fold /
    repeated random split evaluation).

    QUAN TRONG: muc dich la de BIET ket qua co on dinh hay khong, KHONG
    PHAI de "chon lan chay dep nhat roi bao cao rieng lan do". Ket qua
    dung de bao cao trong do an la TRUNG BINH +- STD cua ca ca N seed.

Script nay dung subprocess voi sys.executable (chinh python dang chay
script nay) de goi lai 01b/05/06 - dam bao LUON dung dung venv dang
kich hoat, tranh loi "nham interpreter" (vd dung python he thong thay
vi venv) da tung xay ra.

Mac dinh chay 3 seed (1, 2, 3), MOI SEED CHI TRAIN 1 MODEL SIAMESE
(khong ensemble 3 model nhu binh thuong) - de tong thoi gian chay
TUONG DUONG voi 1 lan chay ensemble-3-model truoc day (3 seed x 1
model = 3 model tong cong, cung tai nguyen tinh toan). Neu muon chinh
xac hon (nhung lau hon), tang --num_models_per_seed.

Cach dung (khong can tham so, path da khop san, chay tu thu muc src):
    cd src
    python 07_cross_validation.py

CANH BAO THOI GIAN: tong thoi gian xap xi = (thoi gian 1 lan chay day
du 01b+05+06) x so luong seed. Neu may cham, giam --seeds xuong con 2
gia tri, hoac giam --siamese_epochs / --max_pos_pairs_per_writer.

Output:
    results/tables/final_model_comparison_seed{N}.csv  (ket qua rieng tung seed)
    results/tables/cross_validation_all_seeds.csv       (gop tat ca seed)
    results/tables/cross_validation_summary.csv         (mean +- std moi model)
    results/figures/cross_validation_auc_comparison.png (bieu do so sanh)
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
DEFAULT_TABLES_DIR = PROJECT_ROOT / "results" / "tables"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def run_step(script_name: str, args_list: list, description: str):
    """
    Goi lai mot script khac (01b/05/06) bang CHINH python dang chay
    script nay (sys.executable) - dam bao dung dung venv, khong bao gio
    bi nham interpreter du nguoi dung dang kich hoat venv nao.
    """
    cmd = [sys.executable, str(SRC_DIR / script_name)] + args_list
    print(f"\n>>> {description}")
    print(f"    Lenh: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(SRC_DIR))
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} that bai (exit code {result.returncode}). Dung lai tai day.")


def main():
    parser = argparse.ArgumentParser(description="Danh gia qua nhieu writer-split khac nhau (cross-validation)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3],
                         help="Danh sach seed de chia writer khac nhau. Mac dinh: 1 2 3")
    parser.add_argument("--val_writers", type=int, default=10)
    parser.add_argument("--test_writers", type=int, default=10)
    parser.add_argument("--num_models_per_seed", type=int, default=1,
                         help="So model Siamese train MOI seed (mac dinh 1, khong ensemble, "
                              "de giu tong thoi gian hop ly khi nhan voi so seed)")
    parser.add_argument("--max_pos_pairs_per_writer", type=int, default=150)
    parser.add_argument("--max_skilled_neg_pairs_per_writer", type=int, default=150)
    parser.add_argument("--num_random_neg_pairs_per_writer", type=int, default=75)
    parser.add_argument("--siamese_epochs", type=int, default=60)
    parser.add_argument("--siamese_patience", type=int, default=15)
    parser.add_argument("--loss", choices=["contrastive", "triplet"], default="triplet")
    args = parser.parse_args()

    tables_dir = Path(DEFAULT_TABLES_DIR)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(DEFAULT_FIGURES_DIR)
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for seed in args.seeds:
        print(f"\n{'#'*70}\n# SEED {seed} ({args.seeds.index(seed)+1}/{len(args.seeds)})\n{'#'*70}")

        run_step("01b_generate_pairs.py", [
            "--seed", str(seed),
            "--val_writers", str(args.val_writers),
            "--test_writers", str(args.test_writers),
            "--max_pos_pairs_per_writer", str(args.max_pos_pairs_per_writer),
            "--max_skilled_neg_pairs_per_writer", str(args.max_skilled_neg_pairs_per_writer),
            "--num_random_neg_pairs_per_writer", str(args.num_random_neg_pairs_per_writer),
        ], f"[Seed {seed}] Sinh writer split + pairs")

        run_step("05_pairwise_baseline.py", [], f"[Seed {seed}] Train Pairwise RF/SVM baseline")

        run_step("06_siamese_network.py", [
            "--num_models", str(args.num_models_per_seed),
            "--epochs", str(args.siamese_epochs),
            "--patience", str(args.siamese_patience),
            "--loss", args.loss,
        ], f"[Seed {seed}] Train Siamese Network")

        seed_comparison_path = tables_dir / "final_model_comparison.csv"
        seed_df = pd.read_csv(seed_comparison_path)
        seed_df["seed"] = seed
        all_results.append(seed_df)

        archived_path = tables_dir / f"final_model_comparison_seed{seed}.csv"
        shutil.copy(seed_comparison_path, archived_path)
        print(f"\n>>> Da luu ket qua rieng cua seed {seed}: {archived_path}")

    combined_df = pd.concat(all_results, ignore_index=True)
    combined_path = tables_dir / "cross_validation_all_seeds.csv"
    combined_df.to_csv(combined_path, index=False)

    metrics = ["accuracy", "roc_auc", "FAR", "FRR"]
    summary = combined_df.groupby("model")[metrics].agg(["mean", "std"])
    summary_path = tables_dir / "cross_validation_summary.csv"
    summary.to_csv(summary_path)

    print(f"\n{'='*70}\nTONG HOP KET QUA QUA {len(args.seeds)} WRITER-SPLIT KHAC NHAU\n{'='*70}")
    print(summary)
    print(f"\nDa luu bang chi tiet: {combined_path}")
    print(f"Da luu bang tong hop: {summary_path}")

    fig, ax = plt.subplots(figsize=(7, 5))
    models = list(combined_df["model"].unique())
    means = [combined_df[combined_df["model"] == m]["roc_auc"].mean() for m in models]
    stds = [combined_df[combined_df["model"] == m]["roc_auc"].std() for m in models]
    ax.bar(models, means, yerr=stds, capsize=8)
    ax.set_ylabel("ROC-AUC")
    ax.set_title(f"ROC-AUC trung binh +/- std qua {len(args.seeds)} writer-split khac nhau")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig_path = figures_dir / "cross_validation_auc_comparison.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Da luu bieu do: {fig_path}")

    print(f"\nHoan tat cross-validation qua {len(args.seeds)} seed.")


if __name__ == "__main__":
    main()