import pandas as pd
from pathlib import Path
from scipy.stats import mannwhitneyu

PROC_DIR = Path("data/processed")
RESULTS_DIR = Path("results/tables")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROC_DIR / "features.csv")

    feature_cols = [c for c in df.columns if c not in ("filename", "writer_id", "label")]
    genuine = df[df["label"] == "genuine"]
    forged = df[df["label"] == "skilled_forgery"]

    results = []
    for col in feature_cols:
        stat, p = mannwhitneyu(genuine[col], forged[col], alternative="two-sided")
        results.append({"feature": col, "U_stat": stat, "p_value": p, "significant_0.05": p < 0.05})

    result_df = pd.DataFrame(results).sort_values("p_value")
    result_df.to_csv(RESULTS_DIR / "mannwhitney_overview.csv", index=False)

    print("=== TỔNG QUAN (chỉ để tham khảo, không dùng trực tiếp cho CV) ===")
    print(result_df.to_string(index=False))
    n_sig = result_df["significant_0.05"].sum()
    print(f"\n{n_sig}/{len(result_df)} đặc trưng có p < 0.05 trên toàn bộ dữ liệu.")


if __name__ == "__main__":
    main()