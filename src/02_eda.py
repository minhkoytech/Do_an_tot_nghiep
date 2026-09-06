"""
02_eda.py
---------
Phan tich kham pha du lieu (EDA) tren TAP TRAIN (khong dung val/test o
buoc nay de tranh nhin truoc du lieu se dung danh gia sau nay).

Thuc hien 2 muc phan tich, khop voi Buoc 2 trong de cuong:

  A. MUC ANH DON LE (image-level)
     So sanh phan bo aspect ratio & mat do pixel den giua 2 nhom:
     chu ky that (genuine) vs chu ky gia (forged), tren toan bo anh
     thuoc cac writer trong tap train.

  B. MUC CAP (pair-level) - khop chinh xac voi "ba nhom" trong de cuong:
     that / gia ngau nhien / gia co ky nang
     Dung truc tiep pair_type da sinh o buoc 01b:
       - genuine_genuine  -> "that"          (2 chu ky that, cung writer)
       - random_forgery   -> "gia ngau nhien" (chu ky that writer khac)
       - skilled_forgery  -> "gia co ky nang" (chu ky gia, cung writer)
     Voi moi cap, tinh do lech (delta) aspect ratio va mat do pixel den
     giua reference va query, roi so sanh phan bo giua 3 nhom bang
     boxplot/violin plot. Ky vong: delta cua "that" nho nhat, "gia co
     ky nang" trung binh (vi ke gia mao co bat chuoc), "gia ngau nhien"
     lon nhat (vi net chu hoan toan khac nguoi).

Outlier detection: dung phuong phap IQR (Q1 - 1.5*IQR, Q3 + 1.5*IQR)
cho tung nhom, ket qua outlier duoc luu rieng de kiem tra thu cong
(vi du: chu ky bi cat sai, anh loi khi scan,...).

Output:
    results/tables/eda_image_level_summary.csv
    results/tables/eda_pair_level_summary.csv
    results/tables/eda_image_level_outliers.csv
    results/tables/eda_pair_level_outliers.csv
    results/figures/eda_image_level_boxplot.png
    results/figures/eda_image_level_violin.png
    results/figures/eda_pair_level_boxplot.png
    results/figures/eda_pair_level_violin.png

Cach dung (khong can tham so, path da khop san):
    cd src
    python 02_eda.py
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # khong can hien thi man hinh, chi luu file
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Duong dan mac dinh khop voi cau truc project DO_AN_TOT_NGHIEP
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "manifest.csv"
DEFAULT_PAIRS_DIR = PROJECT_ROOT / "data" / "processed" / "pairs"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
DEFAULT_TABLES_DIR = PROJECT_ROOT / "results" / "tables"

# Ten hien thi de bao cao khop voi de cuong (khac voi ten cot pair_type ky thuat)
PAIR_TYPE_DISPLAY_NAME = {
    "genuine_genuine": "That",
    "random_forgery": "Gia ngau nhien",
    "skilled_forgery": "Gia co ky nang",
}


# ---------------------------------------------------------------------------
# Trich xuat 2 dac trung hinh hoc co ban tu anh da tien xu ly
# ---------------------------------------------------------------------------
def compute_basic_image_features(image_path: str):
    """
    Tinh 2 dac trung don gian tu mot anh chu ky DA TIEN XU LY (nhi phan,
    net chu = 255, nen = 0):
      - aspect_ratio: ty le rong/cao cua bounding box net chu
      - pixel_density: ty le pixel den (net chu) tren tong so pixel canvas

    Day KHONG PHAI la dac trung dung de huan luyen model (viec do thuoc
    Buoc 3 - 03_features.py voi Hu Moments, GLCM...). Day chi la 2 chi
    so don gian, de tinh, du de lam EDA ban dau.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.nan, np.nan

    ink_pixel_count = int(np.sum(img > 127))
    total_pixels = img.size
    pixel_density = ink_pixel_count / total_pixels if total_pixels > 0 else np.nan

    coords = cv2.findNonZero(img)
    if coords is None:
        aspect_ratio = np.nan
    else:
        _, _, w, h = cv2.boundingRect(coords)
        aspect_ratio = w / h if h > 0 else np.nan

    return aspect_ratio, pixel_density


def parse_writer_split(writer_split_path: str, split_name: str):
    """
    Doc file writer_split.txt (tao boi 01b_generate_pairs.py) va tra ve
    danh sach writer_id thuoc split_name ('train', 'val', hoac 'test').

    Dinh dang moi dong: "train (35 writers): [1, 2, 5, 9, ...]"
    """
    with open(writer_split_path, "r") as f:
        for line in f:
            if line.strip().startswith(split_name):
                bracket_content = line.split(":", 1)[1]
                writer_ids = [int(x) for x in re.findall(r"\d+", bracket_content)]
                return writer_ids
    raise ValueError(f"Khong tim thay split '{split_name}' trong {writer_split_path}")


# ---------------------------------------------------------------------------
# A. EDA muc anh don le
# ---------------------------------------------------------------------------
def run_image_level_eda(manifest_df: pd.DataFrame, train_writer_ids, figures_dir: Path, tables_dir: Path):
    print("\n=== [A] EDA MUC ANH DON LE (train writers only) ===")
    train_df = manifest_df[manifest_df["writer_id"].isin(train_writer_ids)].copy()

    aspect_ratios, pixel_densities = [], []
    for path in tqdm(train_df["path"], desc="Tinh dac trung anh"):
        ar, pd_ = compute_basic_image_features(path)
        aspect_ratios.append(ar)
        pixel_densities.append(pd_)
    train_df["aspect_ratio"] = aspect_ratios
    train_df["pixel_density"] = pixel_densities
    train_df["label_vn"] = train_df["label"].map({"genuine": "That", "forged": "Gia"})

    # Bang thong ke mo ta
    summary = train_df.groupby("label_vn")[["aspect_ratio", "pixel_density"]].describe()
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_path = tables_dir / "eda_image_level_summary.csv"
    summary.to_csv(summary_path)
    print(f"Da luu bang thong ke: {summary_path}")
    print(summary)

    # Phat hien outlier bang IQR, theo tung nhom (that / gia) rieng
    outliers = detect_outliers_iqr(train_df, group_col="label_vn", feature_cols=["aspect_ratio", "pixel_density"])
    outliers_path = tables_dir / "eda_image_level_outliers.csv"
    outliers.to_csv(outliers_path, index=False)
    print(f"So luong anh la outlier: {len(outliers)} / {len(train_df)}  (da luu: {outliers_path})")

    # Bieu do
    figures_dir.mkdir(parents=True, exist_ok=True)
    _plot_boxplot_and_violin(
        train_df, x_col="label_vn", feature_cols=["aspect_ratio", "pixel_density"],
        title_prefix="Muc anh don le", out_prefix=figures_dir / "eda_image_level",
    )
    return train_df


# ---------------------------------------------------------------------------
# B. EDA muc cap (pair-level) - khop voi "ba nhom" trong de cuong
# ---------------------------------------------------------------------------
def run_pair_level_eda(pairs_train_path: str, image_features: dict, figures_dir: Path, tables_dir: Path):
    print("\n=== [B] EDA MUC CAP (pair-level): That / Gia ngau nhien / Gia co ky nang ===")
    pairs_df = pd.read_csv(pairs_train_path)

    delta_aspect, delta_density = [], []
    for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="Tinh delta features"):
        ref_ar, ref_pd = image_features.get(row["reference_path"], (np.nan, np.nan))
        qry_ar, qry_pd = image_features.get(row["query_path"], (np.nan, np.nan))
        delta_aspect.append(abs(ref_ar - qry_ar))
        delta_density.append(abs(ref_pd - qry_pd))

    pairs_df["delta_aspect_ratio"] = delta_aspect
    pairs_df["delta_pixel_density"] = delta_density
    pairs_df["group_vn"] = pairs_df["pair_type"].map(PAIR_TYPE_DISPLAY_NAME)

    summary = pairs_df.groupby("group_vn")[["delta_aspect_ratio", "delta_pixel_density"]].describe()
    tables_dir.mkdir(parents=True, exist_ok=True)
    summary_path = tables_dir / "eda_pair_level_summary.csv"
    summary.to_csv(summary_path)
    print(f"Da luu bang thong ke: {summary_path}")
    print(summary)

    outliers = detect_outliers_iqr(
        pairs_df, group_col="group_vn", feature_cols=["delta_aspect_ratio", "delta_pixel_density"]
    )
    outliers_path = tables_dir / "eda_pair_level_outliers.csv"
    outliers.to_csv(outliers_path, index=False)
    print(f"So luong cap la outlier: {len(outliers)} / {len(pairs_df)}  (da luu: {outliers_path})")

    figures_dir.mkdir(parents=True, exist_ok=True)
    _plot_boxplot_and_violin(
        pairs_df, x_col="group_vn", feature_cols=["delta_aspect_ratio", "delta_pixel_density"],
        title_prefix="Muc cap (pair-level)", out_prefix=figures_dir / "eda_pair_level",
        order=["That", "Gia co ky nang", "Gia ngau nhien"],
    )
    return pairs_df


# ---------------------------------------------------------------------------
# Ham dung chung: outlier detection (IQR) va ve bieu do
# ---------------------------------------------------------------------------
def detect_outliers_iqr(df: pd.DataFrame, group_col: str, feature_cols: list) -> pd.DataFrame:
    """
    Phat hien outlier bang phuong phap IQR, tinh RIENG cho tung nhom
    trong group_col (vi cac nhom co phan bo khac nhau ro rang, khong
    nen dung chung mot nguong cho toan bo du lieu).

    Mot dong duoc coi la outlier neu CO IT NHAT MOT trong cac feature_cols
    nam ngoai khoang [Q1 - 1.5*IQR, Q3 + 1.5*IQR] cua nhom no thuoc ve.
    """
    outlier_rows = []
    for group_name, group_df in df.groupby(group_col):
        mask = pd.Series(False, index=group_df.index)
        for col in feature_cols:
            q1, q3 = group_df[col].quantile(0.25), group_df[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask |= (group_df[col] < lower) | (group_df[col] > upper)
        outlier_rows.append(group_df[mask])
    return pd.concat(outlier_rows) if outlier_rows else df.iloc[0:0]


def _plot_boxplot_and_violin(df, x_col, feature_cols, title_prefix, out_prefix, order=None):
    """Ve boxplot va violin plot cho tung feature, luu ra file PNG."""
    sns.set_style("whitegrid")

    for plot_kind, plot_func in [("boxplot", sns.boxplot), ("violin", sns.violinplot)]:
        fig, axes = plt.subplots(1, len(feature_cols), figsize=(6 * len(feature_cols), 5))
        if len(feature_cols) == 1:
            axes = [axes]
        for ax, feature in zip(axes, feature_cols):
            plot_func(data=df, x=x_col, y=feature, order=order, ax=ax)
            ax.set_title(f"{title_prefix}: {feature}")
            ax.set_xlabel("")
        fig.tight_layout()
        out_path = f"{out_prefix}_{plot_kind}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Da luu bieu do: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="EDA cho du lieu chu ky (chi tren tap train)")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--pairs_dir", default=str(DEFAULT_PAIRS_DIR))
    parser.add_argument("--figures_dir", default=str(DEFAULT_FIGURES_DIR))
    parser.add_argument("--tables_dir", default=str(DEFAULT_TABLES_DIR))
    args = parser.parse_args()

    manifest_df = pd.read_csv(args.manifest)
    writer_split_path = Path(args.pairs_dir) / "writer_split.txt"
    train_writer_ids = parse_writer_split(writer_split_path, "train")
    print(f"So writer trong tap train: {len(train_writer_ids)}")

    figures_dir = Path(args.figures_dir)
    tables_dir = Path(args.tables_dir)

    # A. EDA muc anh don le -> tan dung luon ket qua nay lam cache cho phan B,
    # tranh phai tinh lai feature cho cung mot anh 2 lan.
    train_image_df = run_image_level_eda(manifest_df, train_writer_ids, figures_dir, tables_dir)
    image_features = {
        row["path"]: (row["aspect_ratio"], row["pixel_density"])
        for _, row in train_image_df.iterrows()
    }

    # B. EDA muc cap - dung file pairs_train.csv da sinh o buoc 01b
    pairs_train_path = Path(args.pairs_dir) / "pairs_train.csv"
    run_pair_level_eda(str(pairs_train_path), image_features, figures_dir, tables_dir)

    print("\nHoan tat EDA. Xem ket qua trong results/figures/ va results/tables/")


if __name__ == "__main__":
    main()