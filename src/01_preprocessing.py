"""
Bước 1 — Tiền xử lý ảnh chữ ký (v3 — ĐÃ SỬA sau khi phát hiện rò rỉ dữ liệu)

LỊCH SỬ SỬA LỖI:
- v1: nhị phân hóa cứng khi lưu -> mất hết thông tin độ đậm nhạt (dùng để
  tính GLCM/CNN học được, nhưng cũng thiếu thông tin).
- v2: giữ ảnh xám + CLAHE toàn ảnh -> RÒ RỈ nghiêm trọng do CLAHE khuếch đại
  nhiễu nền/giấy, khiến model học "tắt" qua đặc điểm scan thay vì chữ ký.
- v2.5: ép nền về trắng đồng nhất, giữ xám trong vùng chữ ký -> RÒ RỈ vẫn
  còn (đã xác nhận qua chẩn đoán): ink_intensity_mean/std của vùng chữ ký
  có AUC diff 0.83-0.86 khi phân biệt genuine/forgery MỘT MÌNH — cho thấy
  CEDAR có khác biệt ĐỘ ĐẬM MỰC MANG TÍNH HỆ THỐNG giữa chữ ký thật và giả
  (khả năng do loại bút/lực nhấn khác nhau khi thu thập dữ liệu), không
  liên quan đến hình dạng nét chữ thật.
- v3 (bản này): CHUẨN HÓA độ đậm mực trong vùng chữ ký về cùng 1 phân phối
  (mean/std cố định) cho MỌI ảnh — loại bỏ mức đậm trung bình toàn cục (có
  thể bị lợi dụng làm shortcut), trong khi vẫn giữ biến thiên cục bộ trong
  từng ảnh (là thông tin hợp lệ, ví dụ độ đậm nhạt do tốc độ viết thay đổi
  giữa các đoạn nét).
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw/cedar")
OUT_DIR = Path("data/processed")
IMG_SIZE = 256
TARGET_INK_MEAN = 120
TARGET_INK_STD = 35


def load_grayscale(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {path}")
    return img


def find_signature_bbox(gray_img, margin=10):
    _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return 0, gray_img.shape[1], 0, gray_img.shape[0]
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin, gray_img.shape[1])
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin, gray_img.shape[0])
    return x0, x1, y0, y1


def pad_to_square(img, pad_value=255):
    h, w = img.shape
    size = max(h, w)
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_value)


def normalize_ink_intensity(gray_img, mask, target_mean=TARGET_INK_MEAN, target_std=TARGET_INK_STD, pad_value=255):
    """Chuẩn hóa độ đậm nhạt TRONG VÙNG NÉT CHỮ về cùng 1 phân phối
    (mean/std cố định) cho MỌI ảnh — loại bỏ khác biệt độ đậm mực HỆ THỐNG
    giữa các ảnh, chỉ giữ lại biến thiên TƯƠNG ĐỐI trong từng ảnh."""
    ink_pixels = gray_img[mask > 0].astype(np.float64)
    result = np.full_like(gray_img, pad_value)

    if len(ink_pixels) == 0 or ink_pixels.std() < 1e-6:
        return result

    normalized = (ink_pixels - ink_pixels.mean()) / (ink_pixels.std() + 1e-6)
    normalized = normalized * target_std + target_mean
    normalized = np.clip(normalized, 0, 255)
    result[mask > 0] = normalized.astype(np.uint8)
    return result


def process_one(path, label, writer_id, out_dir):
    gray = load_grayscale(path)
    x0, x1, y0, y1 = find_signature_bbox(gray)
    cropped = gray[y0:y1, x0:x1]
    squared = pad_to_square(cropped, pad_value=255)
    resized = cv2.resize(squared, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    _, mask = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    final = normalize_ink_intensity(resized, mask)

    out_name = f"{writer_id}_{label}_{path.stem}.png"
    cv2.imwrite(str(out_dir / out_name), final)
    return out_name


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    org_dir = RAW_DIR / "full_org"
    forg_dir = RAW_DIR / "full_forg"

    if not org_dir.exists() or not forg_dir.exists():
        print(f"[LỖI] Không tìm thấy {org_dir} hoặc {forg_dir}.")
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
    print(f"Đã xử lý {len(df)} ảnh (đã chuẩn hóa độ đậm mực, loại artifact). Nhãn lưu tại {OUT_DIR / 'labels.csv'}")
    print(df["label"].value_counts())
    print("\n⚠️  Cần chạy lại: 03_features.py, sau đó diagnose_leakage_v2.py để")
    print("    XÁC NHẬN đã hết rò rỉ, rồi mới chạy 08 và 09.")


if __name__ == "__main__":
    main()