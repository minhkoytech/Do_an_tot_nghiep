"""
Bước 3 — Trích xuất đặc trưng thủ công (ĐÃ SỬA — loại bỏ đặc trưng rò rỉ)

- Hu Moments (7 đặc trưng hình học bất biến) — tính trên ảnh NHỊ PHÂN
  (đúng bản chất: đo hình dạng, không cần độ đậm nhạt)
- GLCM (kết cấu) — tính trên ảnh XÁM, CHỈ giữ contrast/homogeneity/energy
- Tỷ lệ nét trên/dưới đường baseline — tính trên ảnh nhị phân

ĐÃ LOẠI BỎ (qua chẩn đoán diagnose_leakage_v2.py, xác nhận rò rỉ):
- ink_intensity_mean, ink_intensity_std: AUC diff = 0.83-0.86 khi dùng một
  mình để phân biệt genuine/forgery -> đây là ĐẶC ĐIỂM HỆ THỐNG của cách
  CEDAR được thu thập (khả năng khác biệt bút/lực nhấn giữa người ký thật
  và người giả mạo), KHÔNG phải đặc điểm nét chữ thật.
- glcm_correlation: AUC diff = 0.755, cũng đáng ngờ (cùng nguyên nhân, vì
  tính trên ảnh xám bị ảnh hưởng bởi cùng artifact độ đậm mực).

Lưu bảng đặc trưng vào data/processed/features.csv
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.feature import graycomatrix, graycoprops

PROC_DIR = Path("data/processed")


def binarize(img):
    """Nhị phân hóa CHỈ dùng cho Hu Moments/baseline_ratio — đặc trưng hình
    học vốn cần ranh giới rõ ràng, không cần độ đậm nhạt."""
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def hu_moments(binary_img):
    moments = cv2.moments(binary_img)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    return hu_log


def glcm_features(gray_img):
    """Dùng ảnh XÁM để đo kết cấu. CHỈ trả về contrast/homogeneity/energy —
    KHÔNG dùng correlation (đã xác nhận rò rỉ qua chẩn đoán)."""
    glcm = graycomatrix(gray_img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, "contrast")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
    energy = graycoprops(glcm, "energy")[0, 0]
    return contrast, homogeneity, energy


def baseline_ratio(binary_img):
    h = binary_img.shape[0]
    top = (binary_img[: h // 2, :] > 0).sum()
    bottom = (binary_img[h // 2 :, :] > 0).sum()
    return top / (bottom + 1e-6)


def extract_features_for_image(path):
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    binary = binarize(gray)

    hu = hu_moments(binary)
    contrast, homogeneity, energy = glcm_features(gray)
    ratio = baseline_ratio(binary)

    feat = {f"hu_{i}": v for i, v in enumerate(hu)}
    feat.update({
        "glcm_contrast": contrast,
        "glcm_homogeneity": homogeneity,
        "glcm_energy": energy,
        "baseline_ratio": ratio,
    })
    return feat


def main():
    labels = pd.read_csv(PROC_DIR / "labels.csv")
    rows = []
    for _, row in labels.iterrows():
        feat = extract_features_for_image(PROC_DIR / row["filename"])
        feat["filename"] = row["filename"]
        feat["writer_id"] = row["writer_id"]
        feat["label"] = row["label"]
        rows.append(feat)

    df = pd.DataFrame(rows)
    df.to_csv(PROC_DIR / "features.csv", index=False)
    print(f"Đã trích {df.shape[1] - 3} đặc trưng cho {len(df)} ảnh (đã loại bỏ đặc trưng rò rỉ).")
    print(f"Lưu tại {PROC_DIR / 'features.csv'}")
    print(f"Danh sách đặc trưng: {[c for c in df.columns if c not in ('filename','writer_id','label')]}")


if __name__ == "__main__":
    main()