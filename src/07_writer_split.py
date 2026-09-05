"""
Bước 7 — Chia writer thành train/validation/test, tạo N_SPLITS=5 cách chia
khác nhau (vì CEDAR chỉ có 55 writer, 1 lần chia có phương sai cao).
Cả 08 và 11 đều lặp qua đúng các split này để so sánh công bằng.
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
    return train_writers, val_writers, test_writers


def main():
    labels = pd.read_csv(PROC_DIR / "labels.csv")
    writers = sorted(labels["writer_id"].unique())

    splits = []
    for i in range(N_SPLITS):
        seed = BASE_SEED + i
        train_w, val_w, test_w = make_one_split(writers, seed)
        splits.append({"split_id": i, "seed": seed, "train_writers": train_w,
                        "val_writers": val_w, "test_writers": test_w})
        print(f"Split {i} (seed={seed}): train={len(train_w)}, val={len(val_w)}, test={len(test_w)}")

    output = {"n_writers_total": len(writers), "n_splits": N_SPLITS, "splits": splits}
    with open(PROC_DIR / "writer_splits.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu {N_SPLITS} cách chia vào: {PROC_DIR / 'writer_splits.json'}")


if __name__ == "__main__":
    main()