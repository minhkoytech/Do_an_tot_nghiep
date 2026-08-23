"""
Bước 7 — Chia writer thành train/validation/test (NHIỀU LẦN, không chỉ 1 lần)

LÝ DO ĐỔI CÁCH LÀM (rút ra từ thực nghiệm): với CEDAR chỉ có 55 writer, một lần
chia train/val/test (ví dụ 38/8/9) cho kết quả có PHƯƠNG SAI RẤT CAO — chạy lại
với cấu hình gần giống nhau vẫn ra kết quả chênh lệch lớn, vì tập test/validation
chỉ có 8-9 writer là quá nhỏ để ước lượng ổn định.

Cách xử lý chuẩn trong nghiên cứu khi cỡ mẫu nhỏ: lặp lại quy trình với NHIỀU
cách chia khác nhau (ở đây dùng N_SPLITS lần, mỗi lần 1 seed khác nhau), rồi
báo cáo kết quả dạng "trung bình ± độ lệch chuẩn" thay vì tin vào 1 lần chạy.

Cả 08_pairwise_baseline.py và 09_siamese_network.py đều sẽ lặp qua ĐÚNG các
split này (cùng file writer_splits.json) để đảm bảo so sánh công bằng.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

PROC_DIR = Path("data/processed")
BASE_SEED = 42
N_SPLITS = 5
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# phần còn lại (0.15) là test


def make_one_split(writers, seed):
    rng = np.random.default_rng(seed)
    writers = np.array(writers)
    rng.shuffle(writers)

    n = len(writers)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_writers = sorted(writers[:n_train].tolist())
    val_writers = sorted(writers[n_train:n_train + n_val].tolist())
    test_writers = sorted(writers[n_train + n_val:].tolist())

    assert set(train_writers) & set(val_writers) == set()
    assert set(train_writers) & set(test_writers) == set()
    assert set(val_writers) & set(test_writers) == set()

    return train_writers, val_writers, test_writers


def main():
    labels = pd.read_csv(PROC_DIR / "labels.csv")
    writers = sorted(labels["writer_id"].unique())

    splits = []
    for i in range(N_SPLITS):
        seed = BASE_SEED + i
        train_w, val_w, test_w = make_one_split(writers, seed)
        splits.append({
            "split_id": i,
            "seed": seed,
            "train_writers": train_w,
            "val_writers": val_w,
            "test_writers": test_w,
        })
        print(f"Split {i} (seed={seed}): train={len(train_w)}, val={len(val_w)}, test={len(test_w)}")

    output = {
        "n_writers_total": len(writers),
        "n_splits": N_SPLITS,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "splits": splits,
    }

    with open(PROC_DIR / "writer_splits.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nĐã lưu {N_SPLITS} cách chia vào: {PROC_DIR / 'writer_splits.json'}")
    print("Cả 08_pairwise_baseline.py và 09_siamese_network.py sẽ lặp qua đúng các split này.")


if __name__ == "__main__":
    main()