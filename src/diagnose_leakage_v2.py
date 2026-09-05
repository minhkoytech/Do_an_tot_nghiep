"""
Chẩn đoán bước 2 — kiểm tra khả năng phân biệt của từng đặc trưng DIFF
(|feature_ref - feature_query|) — đúng không gian mà 08_pairwise_baseline.py
thực sự dùng để train, khác với diagnose_leakage.py (kiểm tra giá trị thô).

Giả thuyết: dù giá trị thô của 1 đặc trưng không tách bạch rõ (AUC ~0.6-0.7),
nếu chữ ký THẬT có giá trị đặc trưng đó dao động hẹp giữa các lần ký (còn
chữ ký GIẢ lệch hệ thống so với thật), thì |diff| có thể tách bạch RẤT rõ
(AUC gần 1.0) dù giá trị thô không như vậy — đây chính là nghi vấn với
ink_intensity_mean/std và glcm_correlation.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
from sklearn.metrics import roc_auc_score

PROC_DIR = Path("data/processed")


def main():
    df = pd.read_csv(PROC_DIR / "features.csv")
    feature_cols = [c for c in df.columns if c not in ("filename", "writer_id", "label")]

    diff_rows = []
    for wid in df["writer_id"].unique():
        sub = df[df["writer_id"] == wid]
        genuine = sub[sub["label"] == "genuine"]
        forged = sub[sub["label"] == "skilled_forgery"]

        # positive: genuine-genuine cùng writer
        for i, j in combinations(genuine.index, 2):
            diff = np.abs(df.loc[i, feature_cols].values - df.loc[j, feature_cols].values)
            diff_rows.append(list(diff) + [1])

        # negative: genuine-forgery cùng writer
        for i in genuine.index:
            for j in forged.index:
                diff = np.abs(df.loc[i, feature_cols].values - df.loc[j, feature_cols].values)
                diff_rows.append(list(diff) + [0])

    diff_df = pd.DataFrame(diff_rows, columns=feature_cols + ["y"])
    print(f"Tổng số cặp kiểm tra: {len(diff_df)} (positive={sum(diff_df.y==1)}, negative={sum(diff_df.y==0)})\n")

    print("=" * 75)
    print("KHẢ NĂNG PHÂN BIỆT CỦA TỪNG ĐẶC TRƯNG DIFF (|ref - query|)")
    print("=" * 75)
    print(f"{'Đặc trưng (diff)':<25}{'AUC (1 mình)':<15}{'Đáng ngờ?'}")
    print("-" * 75)

    suspicious = []
    for col in feature_cols:
        # với diff feature: diff NHỎ nên nghiêng về "match" (positive=1)
        # nên AUC cần tính theo chiều "diff thấp -> match" => dùng -diff làm score
        auc = roc_auc_score(diff_df["y"], -diff_df[col])
        auc = max(auc, 1 - auc)
        flag = ""
        if auc > 0.85:
            flag = "🚨 CỰC KỲ ĐÁNG NGỜ"
            suspicious.append((col, auc))
        elif auc > 0.75:
            flag = "⚠️  đáng ngờ"
            suspicious.append((col, auc))
        print(f"{col:<25}{auc:<15.4f}{flag}")

    print("\n" + "=" * 75)
    if suspicious:
        print(f"PHÁT HIỆN {len(suspicious)} đặc trưng DIFF có AUC > 0.75:")
        for col, auc in sorted(suspicious, key=lambda x: -x[1]):
            print(f"  - {col}: AUC = {auc:.4f}")
        print("\n=> Đây rất có thể là nguồn gốc khiến RF/SVM đạt kết quả bất thường cao.")
        print("   Khuyến nghị: loại các đặc trưng này khỏi features.csv, chạy lại 08.")
    else:
        print("Không phát hiện đặc trưng diff đơn lẻ nào bất thường.")


if __name__ == "__main__":
    main()