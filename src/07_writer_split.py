import json
import numpy as np
import pandas as pd
from pathlib import Path

PROC_DIR = Path("data/processed")
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# phần còn lại (0.15) là test


def main():
    labels = pd.read_csv(PROC_DIR / "labels.csv")
    writers = sorted(labels["writer_id"].unique())

    rng = np.random.default_rng(RANDOM_SEED)
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

    split = {
        "random_seed": RANDOM_SEED,
        "n_writers_total": n,
        "train_writers": train_writers,
        "val_writers": val_writers,
        "test_writers": test_writers,
    }

    with open(PROC_DIR / "writer_split.json", "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)

    print(f"Tổng {n} writer -> Train: {len(train_writers)} | Val: {len(val_writers)} | Test: {len(test_writers)}")
    print(f"Đã lưu: {PROC_DIR / 'writer_split.json'}")
    print("\n⚠️  Từ giờ, mọi bước train/tune PHẢI chỉ dùng train_writers + val_writers.")
    print("    test_writers CHỈ được nạp vào lúc đánh giá cuối cùng, đúng 1 lần.")


if __name__ == "__main__":
    main()