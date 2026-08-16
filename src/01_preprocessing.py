import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw/cedar")
OUT_DIR = Path("data/processed")
IMG_SIZE = (256, 256) 


def load_and_binarize(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {path}")
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def crop_to_signature(binary_img, margin=10):
    ys, xs = np.where(binary_img > 0)
    if len(xs) == 0:
        return binary_img
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin, binary_img.shape[1])
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin, binary_img.shape[0])
    return binary_img[y0:y1, x0:x1]


def process_one(path, label, writer_id, out_dir):
    binary = load_and_binarize(path)
    cropped = crop_to_signature(binary)
    resized = cv2.resize(cropped, IMG_SIZE, interpolation=cv2.INTER_AREA)
    out_name = f"{writer_id}_{label}_{path.stem}.png"
    cv2.imwrite(str(out_dir / out_name), resized)
    return out_name


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    org_dir = RAW_DIR / "full_org"
    forg_dir = RAW_DIR / "full_forg"

    if not org_dir.exists() or not forg_dir.exists():
        print(f"[LỖI] Không tìm thấy {org_dir} hoặc {forg_dir}.")
        print("Xem README.md để biết cách tải và đặt dữ liệu đúng vị trí.")
        return

    for path in sorted(org_dir.glob("*.png")):
        writer_id = path.stem.split("_")[-2] 
        out_name = process_one(path, "genuine", writer_id, OUT_DIR)
        records.append({"filename": out_name, "writer_id": writer_id, "label": "genuine"})

    for path in sorted(forg_dir.glob("*.png")):
        writer_id = path.stem.split("_")[-2]
        out_name = process_one(path, "skilled_forgery", writer_id, OUT_DIR)
        records.append({"filename": out_name, "writer_id": writer_id, "label": "skilled_forgery"})

    df = pd.DataFrame(records)
    df.to_csv(OUT_DIR / "labels.csv", index=False)
    print(f"Đã xử lý {len(df)} ảnh. Nhãn lưu tại {OUT_DIR / 'labels.csv'}")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()