import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import convolve
from skimage.feature import graycomatrix, graycoprops
from skimage.morphology import skeletonize
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Duong dan mac dinh khop voi cau truc project DO_AN_TOT_NGHIEP
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "manifest.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "features.csv"

HU_MOMENT_NAMES = [f"hu_{i+1}" for i in range(7)]
GLCM_PROP_NAMES = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]


# ---------------------------------------------------------------------------
# 1. Hu Moments
# ---------------------------------------------------------------------------
def compute_hu_moments(img_binary: np.ndarray) -> np.ndarray:
    """
    Tinh 7 Hu Moments tu anh nhi phan, ap dung log-transform de on dinh
    thang do gia tri:  h' = -sign(h) * log10(|h|)
    """
    moments = cv2.moments(img_binary)
    hu = cv2.HuMoments(moments).flatten()
    with np.errstate(divide="ignore"):
        hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
    return hu_log


# ---------------------------------------------------------------------------
# 2. GLCM texture features
# ---------------------------------------------------------------------------
def compute_glcm_features(img_binary: np.ndarray) -> np.ndarray:
    """
    Tinh 6 dac trung GLCM tren anh nhi phan (levels=2), lay trung binh
    tren 4 goc (0/45/90/135 do) de bat bien voi huong nghieng net chu.
    """
    img01 = (img_binary > 127).astype(np.uint8)
    glcm = graycomatrix(
        img01, distances=[1], angles=GLCM_ANGLES, levels=2, symmetric=True, normed=True
    )
    features = []
    for prop in GLCM_PROP_NAMES:
        values = graycoprops(glcm, prop)  # shape (1, 4) - 1 distance x 4 angles
        features.append(values.mean())
    return np.array(features)


# ---------------------------------------------------------------------------
# 3. Ty le net tren / duoi baseline
# ---------------------------------------------------------------------------
def compute_baseline_ratio(img_binary: np.ndarray) -> float:
    """
    Uoc luong baseline bang trong tam theo truc y cua pixel net chu,
    sau do tinh ty le so pixel net chu nam TREN / DUOI baseline.
    Ty le > 1 nghia la chu ky co nhieu net o phia tren baseline hon
    (vd nhieu net vuot len tren nhu chu 'h', 'l', dau mu...).
    """
    ys, _ = np.nonzero(img_binary > 127)
    if len(ys) == 0:
        return np.nan
    baseline_y = ys.mean()
    above = np.sum(ys < baseline_y)
    below = np.sum(ys >= baseline_y)
    return above / below if below > 0 else np.nan


# ---------------------------------------------------------------------------
# 4. So diem giao cat net but
# ---------------------------------------------------------------------------
def compute_num_crossings(img_binary: np.ndarray) -> int:
    """
    Lam mong net chu (skeletonize), dem so pixel skeleton co >= 3 pixel
    lang gieng (8-connected) trong chinh skeleton do - day la cac diem
    net but giao nhau hoac phan nhanh (branch point).
    """
    skeleton = skeletonize(img_binary > 127).astype(np.uint8)
    if skeleton.sum() == 0:
        return 0
    neighbor_kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    neighbor_count = convolve(skeleton, neighbor_kernel, mode="constant", cval=0)
    crossings = np.sum((skeleton == 1) & (neighbor_count >= 3))
    return int(crossings)


# ---------------------------------------------------------------------------
# Dac trung hinh hoc don gian (giu lai tu Buoc 2 - da chung minh co tin hieu)
# ---------------------------------------------------------------------------
def compute_simple_geometry(img_binary: np.ndarray):
    ink_pixel_count = int(np.sum(img_binary > 127))
    total_pixels = img_binary.size
    pixel_density = ink_pixel_count / total_pixels if total_pixels > 0 else np.nan

    coords = cv2.findNonZero(img_binary)
    if coords is None:
        aspect_ratio = np.nan
    else:
        _, _, w, h = cv2.boundingRect(coords)
        aspect_ratio = w / h if h > 0 else np.nan
    return aspect_ratio, pixel_density


# ---------------------------------------------------------------------------
# Trich xuat toan bo dac trung cho MOT anh
# ---------------------------------------------------------------------------
def extract_all_features(image_path: str) -> dict:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {}

    aspect_ratio, pixel_density = compute_simple_geometry(img)
    hu = compute_hu_moments(img)
    glcm = compute_glcm_features(img)
    baseline_ratio = compute_baseline_ratio(img)
    num_crossings = compute_num_crossings(img)

    features = {
        "aspect_ratio": aspect_ratio,
        "pixel_density": pixel_density,
        "baseline_ratio": baseline_ratio,
        "num_crossings": num_crossings,
    }
    features.update({name: val for name, val in zip(HU_MOMENT_NAMES, hu)})
    features.update({f"glcm_{name}": val for name, val in zip(GLCM_PROP_NAMES, glcm)})
    return features


def batch_extract_features(manifest_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="Trich xuat dac trung"):
        features = extract_all_features(row["path"])
        if not features:
            print(f"[BO QUA] Khong doc duoc anh: {row['path']}")
            continue
        features.update(
            {
                "path": row["path"],
                "writer_id": row["writer_id"],
                "sample_id": row["sample_id"],
                "label": row["label"],
            }
        )
        rows.append(features)

    df = pd.DataFrame(rows)
    # Sap xep lai cot: metadata truoc, dac trung sau
    meta_cols = ["path", "writer_id", "sample_id", "label"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    return df[meta_cols + feature_cols]


def main():
    parser = argparse.ArgumentParser(description="Trich xuat dac trung thu cong cho anh chu ky")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    manifest_df = pd.read_csv(args.manifest)
    features_df = batch_extract_features(manifest_df)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)

    print(f"\nHoan tat. Da trich xuat dac trung cho {len(features_df)} anh.")
    print(f"So dac trung moi anh: {len(features_df.columns) - 4}")  # tru 4 cot metadata
    print(f"Da luu: {output_path}")
    print("\nKiem tra nhanh (mean theo label):")
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    print(features_df.groupby("label")[numeric_cols].mean().T)


if __name__ == "__main__":
    main()