"""
01_preprocessing.py
--------------------
Tien xu ly anh chu ky viet tay (CEDAR-style dataset).

Khop voi cau truc thu muc project DO_AN_TOT_NGHIEP:
    data/raw/cedar/full_org/   (chu ky that)
    data/raw/cedar/full_forg/  (chu ky gia)
    data/processed/            (output cua script nay)

Cac buoc:
    1. Doc anh, chuyen ve grayscale (neu can).
    2. Nhi phan hoa bang nguong Otsu (dao nguoc de net chu = 255, nen = 0).
    3. Khu nhieu bang morphological opening + loai bo cac connected
       component qua nho (dom nhieu, hat bui khi scan).
    4. Crop theo bounding box cua net chu (bo vien trang thua).
    5. Resize ve kich thuoc chuan va dat vao chinh giua canvas co dinh
       (giu ty le, khong lam meo net chu).

Ket qua: anh grayscale, nen den (0), net chu trang (255), kich thuoc
CANVAS_SIZE, da can giua -> san sang de trich xuat dac trung hoac dua
vao CNN.

Cach dung nhanh:
    python preprocess.py --input_dir /path/to/CEDAR --output_dir /path/to/processed

Cau truc input_dir mac dinh (chuan CEDAR public):
    CEDAR/
        full_org/   original_{writer}_{sample}.png   (chu ky that)
        full_forg/  forgeries_{writer}_{sample}.png  (chu ky gia)

Neu bo du lieu cua ban dat ten khac, chi can sua ham `parse_filename`
va bien GENUINE_DIR / FORGED_DIR ben duoi.
"""

import os
import re
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Cau hinh mac dinh (sua lai neu cau truc thu muc cua ban khac)
# ---------------------------------------------------------------------------
GENUINE_DIR_NAME = "full_org"
FORGED_DIR_NAME = "full_forg"

# Kich thuoc canvas chuan sau tien xu ly (rong x cao).
# 220x155 la kich thuoc pho bien duoc dung trong nhieu paper ve CEDAR.
CANVAS_SIZE = (220, 155)  # (width, height)

# Nguong dien tich toi thieu (pixel) de giu lai mot connected component.
# Cac vet nhieu nho hon se bi loai bo. Can chinh lai theo do phan giai anh.
MIN_COMPONENT_AREA = 15


def parse_filename(filename: str):
    """
    Tach writer_id va sample_id tu ten file.
    Vi du: 'original_12_7.png' -> writer_id=12, sample_id=7
           'forgeries_12_7.png' -> writer_id=12, sample_id=7

    Neu ten file cua ban khac dinh dang nay, CHI CAN SUA HAM NAY.
    """
    match = re.search(r"(\d+)_(\d+)", filename)
    if not match:
        raise ValueError(f"Khong doc duoc writer/sample id tu ten file: {filename}")
    writer_id, sample_id = match.groups()
    return int(writer_id), int(sample_id)


def binarize(img_gray: np.ndarray) -> np.ndarray:
    """
    Nhi phan hoa bang nguong Otsu. Tra ve anh nhi phan voi
    net chu = 255 (trang), nen = 0 (den).
    """
    # Otsu tu dong tim nguong toi uu, khong can chon thu cong.
    _, binary = cv2.threshold(
        img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return binary


def remove_noise(binary_img: np.ndarray, min_area: int = MIN_COMPONENT_AREA) -> np.ndarray:
    """
    Khu nhieu:
      - Morphological opening de loai bo cac diem nhieu li ti.
      - Loai bo cac connected component co dien tich nho hon min_area
        (thuong la hat bui / vet ban khi scan, khong phai net chu).

    AN TOAN VOI NET MONG: neu phep opening xoa sach TOAN BO noi dung
    anh (truong hop net chu qua mong, vd 1 pixel - phat hien qua kiem
    thu voi du lieu tong hop), bo qua buoc opening va chi loc theo dien
    tich connected component tren anh nhi phan GOC. Tranh lam mat toan
    bo chu ky chi vi net qua mong so voi kernel opening.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    opened = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)

    # An toan voi net mong: neu phep opening xoa mat PHAN LON noi dung
    # (con lai duoi 20% so pixel ban dau - truong hop net chu qua mong so
    # voi kernel opening, cac pixel con sot lai thuong la manh vun roi
    # rac se bi loc het boi min_area ben duoi), bo qua buoc opening va
    # chi loc theo dien tich connected component tren anh nhi phan GOC.
    original_count = np.count_nonzero(binary_img)
    if original_count > 0 and np.count_nonzero(opened) < 0.2 * original_count:
        opened = binary_img

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        opened, connectivity=8
    )
    cleaned = np.zeros_like(opened)
    for label_id in range(1, num_labels):  # bo qua label 0 = background
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == label_id] = 255
    return cleaned


def crop_to_signature(binary_img: np.ndarray) -> np.ndarray:
    """
    Crop anh theo bounding box cua vung co net chu (pixel = 255).
    Neu anh trong hoan toan (khong co net chu), tra ve anh goc.
    """
    coords = cv2.findNonZero(binary_img)
    if coords is None:
        return binary_img
    x, y, w, h = cv2.boundingRect(coords)
    return binary_img[y : y + h, x : x + w]


def resize_and_center(cropped_img: np.ndarray, canvas_size=CANVAS_SIZE) -> np.ndarray:
    """
    Resize anh da crop ve vua voi canvas_size, GIU TY LE (khong lam meo
    net chu), sau do dat vao chinh giua mot canvas nen den co dinh.
    """
    canvas_w, canvas_h = canvas_size
    h, w = cropped_img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((canvas_h, canvas_w), dtype=np.uint8)

    scale = min(canvas_w / w, canvas_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(cropped_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    top = (canvas_h - new_h) // 2
    left = (canvas_w - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    return canvas


def preprocess_image(image_path: str, canvas_size=CANVAS_SIZE) -> np.ndarray:
    """
    Pipeline day du cho MOT anh: doc -> grayscale -> nhi phan hoa ->
    khu nhieu -> crop -> resize + can giua.

    Tra ve anh nhi phan (0/255), kich thuoc canvas_size, san sang
    de trich xuat dac trung hoac lam input cho CNN.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Khong doc duoc anh: {image_path}")

    binary = binarize(img)
    cleaned = remove_noise(binary)
    cropped = crop_to_signature(cleaned)
    final_img = resize_and_center(cropped, canvas_size)
    return final_img


def batch_preprocess(input_dir: str, output_dir: str, canvas_size=CANVAS_SIZE):
    """
    Duyet toan bo thu muc full_org va full_forg, tien xu ly tung anh,
    va luu ket qua vao output_dir voi cung cau truc thu muc + ten file,
    dong thoi tra ve mot manifest (list dict) mo ta metadata cua tung anh:
    {path, writer_id, sample_id, label} voi label in {"genuine", "forged"}.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    manifest = []

    for label, subdir_name in [("genuine", GENUINE_DIR_NAME), ("forged", FORGED_DIR_NAME)]:
        src_dir = input_dir / subdir_name
        if not src_dir.exists():
            print(f"[CANH BAO] Khong tim thay thu muc: {src_dir} - bo qua.")
            continue

        dst_dir = output_dir / subdir_name
        dst_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(
            [f for f in os.listdir(src_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp"))]
        )
        for filename in tqdm(image_files, desc=f"Tien xu ly {subdir_name}"):
            src_path = src_dir / filename
            try:
                writer_id, sample_id = parse_filename(filename)
            except ValueError as e:
                print(f"[BO QUA] {e}")
                continue

            processed = preprocess_image(str(src_path), canvas_size)
            dst_path = dst_dir / filename
            cv2.imwrite(str(dst_path), processed)

            manifest.append(
                {
                    "path": str(dst_path),
                    "writer_id": writer_id,
                    "sample_id": sample_id,
                    "label": label,
                }
            )

    return manifest


# Duong dan mac dinh khop voi cau truc project DO_AN_TOT_NGHIEP:
# script nam o src/01_preprocessing.py -> project_root la thu muc cha cua src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw" / "cedar"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def main():
    parser = argparse.ArgumentParser(description="Tien xu ly anh chu ky CEDAR-style dataset")
    parser.add_argument(
        "--input_dir", default=str(DEFAULT_INPUT_DIR),
        help=f"Thu muc goc chua full_org/ va full_forg/ (mac dinh: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output_dir", default=str(DEFAULT_OUTPUT_DIR),
        help=f"Thu muc dich luu anh da tien xu ly (mac dinh: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--canvas_width", type=int, default=CANVAS_SIZE[0])
    parser.add_argument("--canvas_height", type=int, default=CANVAS_SIZE[1])
    args = parser.parse_args()

    canvas_size = (args.canvas_width, args.canvas_height)
    manifest = batch_preprocess(args.input_dir, args.output_dir, canvas_size)

    import pandas as pd
    manifest_df = pd.DataFrame(manifest)
    manifest_path = Path(args.output_dir) / "manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"\nHoan tat. Da xu ly {len(manifest_df)} anh.")
    print(f"Manifest luu tai: {manifest_path}")
    print(manifest_df["label"].value_counts())


if __name__ == "__main__":
    main()