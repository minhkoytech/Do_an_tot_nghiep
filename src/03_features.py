"""
Bước 3 — Trích xuất đặc trưng thủ công
- Hu Moments (7 đặc trưng hình học bất biến)
- GLCM (contrast, homogeneity, energy, correlation)
- Tỷ lệ nét trên/dưới đường baseline
Lưu bảng đặc trưng vào data/processed/features.csv
"""
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.feature import graycomatrix, graycoprops

PROC_DIR = Path("data/processed")


def binarize(img):
    """Nhị phân hóa CHỈ dùng cho Hu Moments — đặc trưng hình học vốn cần
    ranh giới rõ ràng (đâu là nét chữ, đâu là nền), không cần độ đậm nhạt."""
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def hu_moments(binary_img):
    moments = cv2.moments(binary_img)
    hu = cv2.HuMoments(moments).flatten()
    # log-scale để các giá trị dễ so sánh (Hu moments chênh lệch bậc rất lớn)
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    return hu_log


def glcm_features(gray_img):
    """QUAN TRỌNG: dùng ảnh XÁM (không nhị phân hóa) — GLCM đo biến thiên độ
    đậm nhạt (liên quan lực nhấn bút), trên ảnh nhị phân chỉ có 2 mức sáng thì
    GLCM gần như mất hết ý nghĩa đo kết cấu."""
    glcm = graycomatrix(gray_img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, "contrast")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
    energy = graycoprops(glcm, "energy")[0, 0]
    correlation = graycoprops(glcm, "correlation")[0, 0]
    return contrast, homogeneity, energy, correlation


def baseline_ratio(binary_img):
    """Tỷ lệ pixel nét chữ phần trên/dưới — dùng ảnh nhị phân, đúng bản chất
    đo phân bố hình học, không cần độ đậm nhạt."""
    h = binary_img.shape[0]
    top = (binary_img[: h // 2, :] > 0).sum()
    bottom = (binary_img[h // 2 :, :] > 0).sum()
    return top / (bottom + 1e-6)


def ink_intensity_stats(gray_img, binary_img):
    """Đặc trưng MỚI — chỉ tính được vì giờ đã giữ ảnh xám: đo độ đậm nhạt
    trung bình và độ lệch chuẩn của nét mực (vùng có chữ ký, theo mask nhị
    phân). Phản ánh lực nhấn bút / loại bút — thông tin hoàn toàn bị mất
    nếu chỉ có ảnh nhị phân đen/trắng."""
    mask = binary_img > 0
    if mask.sum() == 0:
        return 0.0, 0.0
    ink_pixels = gray_img[mask]
    return float(ink_pixels.mean()), float(ink_pixels.std())


def extract_features_for_image(path):
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    binary = binarize(gray)

    hu = hu_moments(binary)
    contrast, homogeneity, energy, correlation = glcm_features(gray)  # <- ảnh XÁM, không phải binary
    ratio = baseline_ratio(binary)
    ink_mean, ink_std = ink_intensity_stats(gray, binary)

    feat = {f"hu_{i}": v for i, v in enumerate(hu)}
    feat.update({
        "glcm_contrast": contrast,
        "glcm_homogeneity": homogeneity,
        "glcm_energy": energy,
        "glcm_correlation": correlation,
        "baseline_ratio": ratio,
        "ink_intensity_mean": ink_mean,
        "ink_intensity_std": ink_std,
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
    print(f"Đã trích {df.shape[1] - 3} đặc trưng cho {len(df)} ảnh.")
    print(f"Lưu tại {PROC_DIR / 'features.csv'}")


if __name__ == "__main__":
    main()