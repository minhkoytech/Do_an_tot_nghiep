import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROC_DIR = Path("data/processed")
FIG_DIR = Path("results/figures")


def pixel_density(img):
    return (img > 0).sum() / img.size


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(PROC_DIR / "labels.csv")

    densities = []
    for _, row in labels.iterrows():
        img = cv2.imread(str(PROC_DIR / row["filename"]), cv2.IMREAD_GRAYSCALE)
        densities.append(pixel_density(img))
    labels["pixel_density"] = densities

    # Thống kê mô tả theo nhóm
    print(labels.groupby("label")["pixel_density"].describe())

    # Boxplot so sánh genuine vs skilled_forgery
    plt.figure(figsize=(6, 5))
    sns.boxplot(data=labels, x="label", y="pixel_density")
    plt.title("Mật độ pixel đen: Genuine vs Skilled Forgery")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pixel_density_boxplot.png", dpi=150)
    plt.close()

    # Violin plot chi tiết hơn
    plt.figure(figsize=(6, 5))
    sns.violinplot(data=labels, x="label", y="pixel_density")
    plt.title("Phân bố mật độ pixel đen theo nhóm")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pixel_density_violin.png", dpi=150)
    plt.close()

    # Phát hiện outlier bằng IQR
    for label_name, group in labels.groupby("label"):
        q1, q3 = group["pixel_density"].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = group[(group["pixel_density"] < low) | (group["pixel_density"] > high)]
        print(f"[{label_name}] {len(outliers)} outlier(s) trên tổng {len(group)}")

    labels.to_csv(PROC_DIR / "labels_with_eda.csv", index=False)
    print(f"Biểu đồ lưu tại {FIG_DIR}/, dữ liệu EDA lưu tại {PROC_DIR}/labels_with_eda.csv")


if __name__ == "__main__":
    main()