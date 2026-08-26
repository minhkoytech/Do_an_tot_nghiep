"""
Chẩn đoán bước 3 — kiểm tra TRỰC TIẾP trên ảnh đã lưu (data/processed/*.png)
xem còn artifact độ đậm mực hay không, đo bằng CHÍNH cách một CNN đơn giản
có thể "nhìn thấy" (trung bình/độ lệch chuẩn pixel vùng nét chữ).

Đây là bước kiểm tra CUỐI CÙNG, bắt buộc, trước khi chạy 09_siamese_network.py
(để tránh lặp lại việc tốn ~20 tiếng train rồi phát hiện rò rỉ sau).

Nếu bảng kết quả không còn AUC > 0.75 nào -> an toàn để chạy 09.
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from sklearn.metrics import roc_auc_score

PROC_DIR = Path("data/processed")


def ink_stats_from_saved_image(filename):
    img = cv2.imread(str(PROC_DIR / filename), cv2.IMREAD_GRAYSCALE)
    mask = img < 255
    if mask.sum() == 0:
        return 0.0, 0.0
    ink = img[mask].astype(np.float64)
    return ink.mean(), ink.std()


def main():
    labels = pd.read_csv(PROC_DIR / "labels.csv")

    print("Đang tính thống kê độ đậm mực từ ảnh đã lưu (có thể mất 1-2 phút)...")
    stats = []
    for _, row in labels.iterrows():
        mean, std = ink_stats_from_saved_image(row["filename"])
        stats.append({"filename": row["filename"], "writer_id": row["writer_id"],
                       "label": row["label"], "ink_mean": mean, "ink_std": std})
    stats_df = pd.DataFrame(stats)

    y = (stats_df["label"] == "genuine").astype(int)
    for col in ["ink_mean", "ink_std"]:
        auc = roc_auc_score(y, stats_df[col])
        auc = max(auc, 1 - auc)
        print(f"[Giá trị thô] {col}: AUC = {auc:.4f}")

    diff_rows = []
    for wid in stats_df["writer_id"].unique():
        sub = stats_df[stats_df["writer_id"] == wid]
        genuine = sub[sub["label"] == "genuine"]
        forged = sub[sub["label"] == "skilled_forgery"]

        for i, j in combinations(genuine.index, 2):
            diff_rows.append({
                "ink_mean_diff": abs(stats_df.loc[i, "ink_mean"] - stats_df.loc[j, "ink_mean"]),
                "ink_std_diff": abs(stats_df.loc[i, "ink_std"] - stats_df.loc[j, "ink_std"]),
                "y": 1,
            })
        for i in genuine.index:
            for j in forged.index:
                diff_rows.append({
                    "ink_mean_diff": abs(stats_df.loc[i, "ink_mean"] - stats_df.loc[j, "ink_mean"]),
                    "ink_std_diff": abs(stats_df.loc[i, "ink_std"] - stats_df.loc[j, "ink_std"]),
                    "y": 0,
                })

    diff_df = pd.DataFrame(diff_rows)
    print()
    print("=" * 70)
    print("KIỂM TRA KHÔNG GIAN DIFF (đúng cách 08/09 xử lý dữ liệu)")
    print("=" * 70)
    any_suspicious = False
    for col in ["ink_mean_diff", "ink_std_diff"]:
        auc = roc_auc_score(diff_df["y"], -diff_df[col])
        auc = max(auc, 1 - auc)
        flag = ""
        if auc > 0.75:
            flag = "🚨 VẪN CÒN RÒ RỈ"
            any_suspicious = True
        print(f"{col}: AUC = {auc:.4f}  {flag}")

    print()
    if any_suspicious:
        print("❌ VẪN CÒN RÒ RỈ — CHƯA NÊN chạy 09_siamese_network.py.")
        print("   Cần điều chỉnh thêm tham số chuẩn hóa hoặc phương pháp mạnh hơn.")
    else:
        print("✅ SẠCH — an toàn để chạy 08_pairwise_baseline.py và")
        print("   09_siamese_network.py với ảnh hiện tại.")


if __name__ == "__main__":
    main()