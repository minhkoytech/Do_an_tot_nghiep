import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu

# ---------------------------------------------------------------------------
# Duong dan mac dinh khop voi cau truc project DO_AN_TOT_NGHIEP
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FEATURES = PROJECT_ROOT / "data" / "processed" / "features.csv"
DEFAULT_PAIRS_DIR = PROJECT_ROOT / "data" / "processed" / "pairs"
DEFAULT_TABLES_DIR = PROJECT_ROOT / "results" / "tables"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

META_COLS = ["path", "writer_id", "sample_id", "label"]
ALPHA = 0.05  # nguong y nghia thong ke (sau khi hieu chinh FDR)
TOP_N_PLOT = 6  # so dac trung co y nghia nhat duoc ve boxplot minh hoa


def parse_writer_split(writer_split_path: str, split_name: str):
    """Doc file writer_split.txt (tao boi 01b_generate_pairs.py)."""
    with open(writer_split_path, "r") as f:
        for line in f:
            if line.strip().startswith(split_name):
                bracket_content = line.split(":", 1)[1]
                return [int(x) for x in re.findall(r"\d+", bracket_content)]
    raise ValueError(f"Khong tim thay split '{split_name}' trong {writer_split_path}")


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """
    Hieu chinh Benjamini-Hochberg (kiem soat False Discovery Rate),
    cai dat thu cong bang NumPy (khong can them thu vien statsmodels).
    Tra ve mang p-value da hieu chinh (adjusted p-value / q-value),
    cung thu tu voi mang dau vao.
    """
    n = len(pvalues)
    order = np.argsort(pvalues)
    ranked_p = pvalues[order]
    ranks = np.arange(1, n + 1)

    adjusted = ranked_p * n / ranks
    # Dam bao tinh don dieu: adjusted p-value khong duoc giam dan khi rank tang
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    result = np.empty(n)
    result[order] = adjusted
    return result


def rank_biserial_effect_size(u_stat: float, n1: int, n2: int) -> float:
    """
    Xap xi effect size (rank-biserial correlation) tu thong ke U cua
    Mann-Whitney: r = 1 - 2U / (n1 * n2)
    Gia tri trong khoang [-1, 1], |r| cang lon nghia la 2 nhom cang
    tach biet ro rang theo rank.
    """
    return 1 - (2 * u_stat) / (n1 * n2)


def run_mannwhitney_tests(features_df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Chay Mann-Whitney U test cho tung dac trung, so sanh nhom 'genuine'
    va 'forged'. Tra ve DataFrame ket qua (chua hieu chinh multiple
    testing - buoc do thuc hien o main()).
    """
    genuine_df = features_df[features_df["label"] == "genuine"]
    forged_df = features_df[features_df["label"] == "forged"]

    results = []
    for feature in feature_cols:
        g_values = genuine_df[feature].dropna()
        f_values = forged_df[feature].dropna()
        if len(g_values) < 2 or len(f_values) < 2:
            continue

        u_stat, p_value = mannwhitneyu(g_values, f_values, alternative="two-sided")
        effect_size = rank_biserial_effect_size(u_stat, len(g_values), len(f_values))

        results.append(
            {
                "feature": feature,
                "n_genuine": len(g_values),
                "n_forged": len(f_values),
                "median_genuine": g_values.median(),
                "median_forged": f_values.median(),
                "u_statistic": u_stat,
                "p_value": p_value,
                "effect_size_r": effect_size,
            }
        )

    return pd.DataFrame(results)


def plot_top_features(features_df: pd.DataFrame, top_features: list, out_path: Path):
    """Ve boxplot cho cac dac trung co y nghia thong ke nhat, de minh hoa truc quan."""
    sns.set_style("whitegrid")
    n = len(top_features)
    if n == 0:
        print("Khong co dac trung nao dat nguong y nghia de ve bieu do.")
        return

    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, feature in zip(axes, top_features):
        sns.boxplot(data=features_df, x="label", y=feature, ax=ax, order=["genuine", "forged"])
        ax.set_title(feature)
        ax.set_xlabel("")
    for ax in axes[len(top_features):]:
        ax.axis("off")

    fig.suptitle("Top dac trung co y nghia thong ke nhat (Mann-Whitney U, sau hieu chinh FDR)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Da luu bieu do: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Kiem dinh Mann-Whitney U cho dac trung chu ky")
    parser.add_argument("--features", default=str(DEFAULT_FEATURES))
    parser.add_argument("--pairs_dir", default=str(DEFAULT_PAIRS_DIR))
    parser.add_argument("--tables_dir", default=str(DEFAULT_TABLES_DIR))
    parser.add_argument("--figures_dir", default=str(DEFAULT_FIGURES_DIR))
    parser.add_argument("--alpha", type=float, default=ALPHA)
    args = parser.parse_args()

    features_df = pd.read_csv(args.features)

    # Chi giu lai anh thuoc writer trong tap train (chong data leakage)
    writer_split_path = Path(args.pairs_dir) / "writer_split.txt"
    train_writer_ids = parse_writer_split(writer_split_path, "train")
    train_df = features_df[features_df["writer_id"].isin(train_writer_ids)].copy()
    print(f"So anh trong tap train dung de kiem dinh: {len(train_df)} "
          f"({train_df['label'].value_counts().to_dict()})")

    feature_cols = [c for c in features_df.columns if c not in META_COLS]
    results_df = run_mannwhitney_tests(train_df, feature_cols)

    # Hieu chinh multiple testing (Benjamini-Hochberg FDR)
    results_df["p_value_adjusted"] = benjamini_hochberg(results_df["p_value"].values)
    results_df["significant"] = results_df["p_value_adjusted"] < args.alpha
    results_df = results_df.sort_values("p_value_adjusted").reset_index(drop=True)

    tables_dir = Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    results_path = tables_dir / "stats_tests_results.csv"
    results_df.to_csv(results_path, index=False)

    print(f"\nDa luu bang ket qua: {results_path}")
    print(f"\nSo dac trung co y nghia thong ke (p_adjusted < {args.alpha}): "
          f"{results_df['significant'].sum()} / {len(results_df)}")
    print("\n=== Ket qua (sap xep theo p-value da hieu chinh) ===")
    display_cols = ["feature", "median_genuine", "median_forged", "p_value", "p_value_adjusted", "effect_size_r", "significant"]
    print(results_df[display_cols].to_string(index=False))

    # Ve bieu do minh hoa cho top dac trung co y nghia nhat (uu tien theo |effect_size|)
    significant_df = results_df[results_df["significant"]].copy()
    significant_df["abs_effect"] = significant_df["effect_size_r"].abs()
    top_features = significant_df.sort_values("abs_effect", ascending=False)["feature"].head(TOP_N_PLOT).tolist()

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_top_features(train_df, top_features, figures_dir / "stats_tests_top_features_boxplot.png")

    print(f"\nDac trung nen uu tien dua vao model (sap xep theo effect size): {top_features}")


if __name__ == "__main__":
    main()