import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from skimage.feature import graycomatrix, graycoprops

PROC_DIR = Path("data/processed")


def hu_moments(img):
    moments = cv2.moments(img)
    hu = cv2.HuMoments(moments).flatten()
    # log-scale để các giá trị dễ so sánh (Hu moments chênh lệch bậc rất lớn)
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    return hu_log


def glcm_features(img):
    glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, "contrast")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
    energy = graycoprops(glcm, "energy")[0, 0]
    correlation = graycoprops(glcm, "correlation")[0, 0]
    return contrast, homogeneity, energy, correlation


def baseline_ratio(img):
    """Tỷ lệ pixel đen phần trên vs phần dưới đường giữa ảnh (baseline xấp xỉ)."""
    h = img.shape[0]
    top = (img[: h // 2, :] > 0).sum()
    bottom = (img[h // 2 :, :] > 0).sum()
    return top / (bottom + 1e-6)


def extract_features_for_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    hu = hu_moments(img)
    contrast, homogeneity, energy, correlation = glcm_features(img)
    ratio = baseline_ratio(img)
    feat = {f"hu_{i}": v for i, v in enumerate(hu)}
    feat.update({
        "glcm_contrast": contrast,
        "glcm_homogeneity": homogeneity,
        "glcm_energy": energy,
        "glcm_correlation": correlation,
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
    print(f"Đã trích {df.shape[1] - 3} đặc trưng cho {len(df)} ảnh.")
    print(f"Lưu tại {PROC_DIR / 'features.csv'}")


if __name__ == "__main__":
    main()