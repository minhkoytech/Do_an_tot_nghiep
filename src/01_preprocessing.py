"""
Bước 1 — Tiền xử lý ảnh chữ ký (ĐÃ SỬA — v2)

3 THAY ĐỔI QUAN TRỌNG so với bản đầu (phát hiện khi soát lại pipeline):

1. GIỮ ẢNH XÁM (grayscale), KHÔNG nhị phân hóa cứng khi lưu ra.
   Lý do: nhị phân hóa (chỉ còn đen/trắng) làm mất thông tin độ đậm nhạt của
   nét mực (liên quan lực nhấn bút) — đây chính là thông tin GLCM (kết cấu)
   cần để đo, và cũng là thông tin hữu ích để CNN tự học. Nhị phân hóa CHỈ
   dùng nội bộ để tìm bounding box (cắt vùng chữ ký), không dùng để lưu ảnh
   cuối cùng.

2. GIỮ TỶ LỆ KHUNG HÌNH khi resize (pad về hình vuông trước khi resize),
   KHÔNG ép chữ ký dài/dẹt thành hình vuông trực tiếp.
   Lý do: resize ép tỷ lệ sẽ làm méo hình dạng chữ ký, ảnh hưởng trực tiếp
   đến Hu Moments (đặc trưng hình học) và cả CNN.

3. Chuẩn hóa nền về trắng đồng nhất, làm rõ tương phản nét chữ so với nền
   (CLAHE - Contrast Limited Adaptive Histogram Equalization) trước khi lưu,
   giúp giảm ảnh hưởng của việc scan ở điều kiện ánh sáng không đều.
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw/cedar")
OUT_DIR = Path("data/processed")
IMG_SIZE = 256  # kích thước chuẩn hóa (hình vuông sau khi pad)


def load_grayscale(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {path}")
    return img


def find_signature_bbox(gray_img, margin=10):
    """Nhị phân hóa CHỈ để tìm vùng chữ ký (bounding box), không dùng để lưu."""
    _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return 0, gray_img.shape[1], 0, gray_img.shape[0]
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin, gray_img.shape[1])
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin, gray_img.shape[0])
    return x0, x1, y0, y1


def pad_to_square(img, pad_value=255):
    """Đệm thêm viền (màu nền trắng) để ảnh thành hình vuông TRƯỚC khi resize,
    tránh resize ép làm méo tỷ lệ khung hình gốc của chữ ký."""
    h, w = img.shape
    size = max(h, w)
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_value)


def enhance_contrast(img):
    """CLAHE — tăng tương phản cục bộ, giảm ảnh hưởng của ánh sáng không đều lúc scan,
    làm nét mực rõ ràng hơn mà vẫn giữ thông tin sắc độ xám (không nhị phân hóa)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def process_one(path, label, writer_id, out_dir):
    gray = load_grayscale(path)
    x0, x1, y0, y1 = find_signature_bbox(gray)
    cropped = gray[y0:y1, x0:x1]
    squared = pad_to_square(cropped, pad_value=255)
    resized = cv2.resize(squared, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    enhanced = enhance_contrast(resized)

    out_name = f"{writer_id}_{label}_{path.stem}.png"
    cv2.imwrite(str(out_dir / out_name), enhanced)
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
    print(f"Đã xử lý {len(df)} ảnh (GIỮ SẮC ĐỘ XÁM, giữ tỷ lệ khung hình). Nhãn lưu tại {OUT_DIR / 'labels.csv'}")
    print(df["label"].value_counts())
    print("\n⚠️  Ảnh giờ là GRAYSCALE, không còn nhị phân đen/trắng như bản trước.")
    print("    Cần chạy lại 03_features.py để tính lại đặc trưng trên ảnh mới.")
    print("    09_siamese_network.py cũng cần chạy lại vì ảnh đầu vào đã đổi.")


if __name__ == "__main__":
    main()