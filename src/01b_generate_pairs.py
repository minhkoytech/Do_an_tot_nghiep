"""
01b_generate_pairs.py
----------------------
Tao writer-independent split (train/val/test) va sinh cac cap chu ky
(reference, query) de huan luyen/danh gia Siamese Network va Pairwise
Classical ML baseline (theo gop y cua GVHD).

Chay SAU 01_preprocessing.py va TRUOC 02_eda.py / 03_features.py, vi
writer split nay se duoc dung xuyen suot toan bo do an (EDA, feature
engineering, baseline, Siamese) de dam bao khong co writer nao bi
ro ri giua cac giai doan.

Khop voi cau truc thu muc project DO_AN_TOT_NGHIEP:
    input:  data/processed/manifest.csv  (tao boi 01_preprocessing.py)
    output: data/processed/pairs/pairs_{train,val,test}.csv

QUAN TRONG - writer-independent split:
    Moi writer CHI xuat hien o DUY NHAT mot trong ba tap train/val/test.
    Dieu nay mo phong dung tinh huong thuc te: he thong phai xac thuc
    chu ky cua nhung nguoi/mau chu ky CHUA TUNG THAY trong luc huan luyen.
    Neu khong lam vay (vd chia ngau nhien theo tung anh), mo hinh se
    "hoc thuoc" dac diem rieng cua tung writer va ket qua danh gia se
    bi lac quan gia tao (data leakage).

Ba loai cap (pair) duoc sinh ra, dung CHUNG cho ca Siamese Network va
Pairwise Classical ML baseline (RF/SVM tren |feat_ref - feat_query|)
de dam bao so sanh cong bang giua hai huong tiep can (dung gop y so 4
cua GVHD):

    1. genuine_genuine  (label = 1, MATCH)
       Hai chu ky that cua CUNG mot writer.

    2. skilled_forgery  (label = 0, NON-MATCH)
       Chu ky that (reference) ghep voi chu ky GIA cua CUNG writer do.
       Day la truong hop KHO nhat va quan trong nhat trong ung dung
       ngan hang (ke gian lan co ky nang, co luyen tap truoc).

    3. random_forgery   (label = 0, NON-MATCH)
       Chu ky that (reference) ghep voi chu ky that cua MOT writer
       KHAC. Mo phong truong hop ke gian lan khong biet chu ky that
       trong thuc te.

Threshold quyet dinh (EER-based) va lua chon model/feature CHI duoc
thuc hien tren train/val. Tap test chi dung DUY NHAT MOT LAN de bao
cao ket qua cuoi cung (dung gop y so 2 cua GVHD).

Cach dung:
    python generate_pairs.py --manifest /path/to/manifest.csv --output_dir /path/to/pairs
"""

import argparse
import itertools
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42


def split_writers(writer_ids, val_size=10, test_size=10, random_state=RANDOM_SEED):
    """
    Chia danh sach writer_id thanh 3 nhom KHONG GIAO NHAU: train/val/test.
    Mac dinh: 10 writer cho val, 10 writer cho test, phan con lai cho train
    (voi CEDAR 55 writer -> 35 train / 10 val / 10 test).

    Day la buoc CHONG DATA LEAKAGE quan trong nhat: moi writer chi thuoc
    ve dung MOT tap.
    """
    writer_ids = sorted(set(writer_ids))
    train_val_writers, test_writers = train_test_split(
        writer_ids, test_size=test_size, random_state=random_state
    )
    train_writers, val_writers = train_test_split(
        train_val_writers, test_size=val_size, random_state=random_state
    )
    return {
        "train": sorted(train_writers),
        "val": sorted(val_writers),
        "test": sorted(test_writers),
    }


def _genuine_genuine_pairs(genuine_paths, writer_id, max_pairs=None, seed=RANDOM_SEED):
    """Sinh cac cap (genuine, genuine) cung writer -> label = 1."""
    pairs = list(itertools.combinations(genuine_paths, 2))
    rng = random.Random(seed + int(writer_id))
    rng.shuffle(pairs)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    return [
        {
            "reference_path": p1,
            "query_path": p2,
            "label": 1,
            "pair_type": "genuine_genuine",
            "writer_id_ref": writer_id,
            "writer_id_query": writer_id,
        }
        for p1, p2 in pairs
    ]


def _skilled_forgery_pairs(genuine_paths, forged_paths, writer_id, max_pairs=None, seed=RANDOM_SEED):
    """Sinh cac cap (genuine, forged) cung writer -> label = 0 (skilled forgery)."""
    pairs = list(itertools.product(genuine_paths, forged_paths))
    rng = random.Random(seed + int(writer_id) + 1000)
    rng.shuffle(pairs)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    return [
        {
            "reference_path": p1,
            "query_path": p2,
            "label": 0,
            "pair_type": "skilled_forgery",
            "writer_id_ref": writer_id,
            "writer_id_query": writer_id,
        }
        for p1, p2 in pairs
    ]


def _random_forgery_pairs(manifest_df, writer_id, genuine_paths, num_pairs, split_writer_ids, seed=RANDOM_SEED):
    """
    Sinh cac cap (genuine cua writer_id, genuine cua writer KHAC) -> label = 0
    (random forgery / impostor khong biet chu ky that).

    QUAN TRONG: writer "khac" chi duoc lay trong pham vi split_writer_ids
    (cung split train/val/test voi writer_id). Neu lay tu ca manifest_df
    (bao gom ca writer thuoc split khac) se gay RO RI DU LIEU giua cac
    split, lam mat tinh writer-independent cua toan bo thi nghiem.
    """
    other_genuine = manifest_df[
        (manifest_df["label"] == "genuine")
        & (manifest_df["writer_id"] != writer_id)
        & (manifest_df["writer_id"].isin(split_writer_ids))
    ]
    if len(other_genuine) == 0 or len(genuine_paths) == 0:
        return []

    rng = random.Random(seed + int(writer_id) + 2000)
    pairs = []
    for _ in range(num_pairs):
        ref = rng.choice(genuine_paths)
        other_row = other_genuine.sample(n=1, random_state=rng.randint(0, 10**6)).iloc[0]
        pairs.append(
            {
                "reference_path": ref,
                "query_path": other_row["path"],
                "label": 0,
                "pair_type": "random_forgery",
                "writer_id_ref": writer_id,
                "writer_id_query": other_row["writer_id"],
            }
        )
    return pairs


def generate_pairs_for_split(
    manifest_df: pd.DataFrame,
    writer_ids,
    max_pos_pairs_per_writer=40,
    max_skilled_neg_pairs_per_writer=40,
    num_random_neg_pairs_per_writer=20,
    seed=RANDOM_SEED,
):
    """
    Sinh toan bo pairs (positive + 2 loai negative) cho mot tap writer
    (train, val, hoac test). Cac tham so max_*_per_writer dung de kiem
    soat quy mo dataset va can bang ty le positive/negative - dieu chinh
    theo nhu cau thuc nghiem cua ban.
    """
    all_pairs = []
    for writer_id in writer_ids:
        writer_genuine = manifest_df[
            (manifest_df["label"] == "genuine") & (manifest_df["writer_id"] == writer_id)
        ]["path"].tolist()
        writer_forged = manifest_df[
            (manifest_df["label"] == "forged") & (manifest_df["writer_id"] == writer_id)
        ]["path"].tolist()

        if len(writer_genuine) < 2:
            print(f"[CANH BAO] Writer {writer_id} co it hon 2 chu ky that, bo qua.")
            continue

        all_pairs += _genuine_genuine_pairs(
            writer_genuine, writer_id, max_pairs=max_pos_pairs_per_writer, seed=seed
        )
        if writer_forged:
            all_pairs += _skilled_forgery_pairs(
                writer_genuine, writer_forged, writer_id,
                max_pairs=max_skilled_neg_pairs_per_writer, seed=seed,
            )
        all_pairs += _random_forgery_pairs(
            manifest_df, writer_id, writer_genuine,
            num_pairs=num_random_neg_pairs_per_writer,
            split_writer_ids=writer_ids, seed=seed,
        )

    pairs_df = pd.DataFrame(all_pairs)
    return pairs_df.sample(frac=1, random_state=seed).reset_index(drop=True)


# Duong dan mac dinh khop voi cau truc project DO_AN_TOT_NGHIEP:
# script nam o src/01b_generate_pairs.py -> project_root la thu muc cha cua src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "manifest.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "pairs"


def main():
    parser = argparse.ArgumentParser(description="Sinh writer-independent split va cac cap chu ky")
    parser.add_argument(
        "--manifest", default=str(DEFAULT_MANIFEST),
        help=f"File manifest.csv tao boi 01_preprocessing.py (mac dinh: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--output_dir", default=str(DEFAULT_OUTPUT_DIR),
        help=f"Thu muc luu cac file pairs_*.csv (mac dinh: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--val_writers", type=int, default=10)
    parser.add_argument("--test_writers", type=int, default=10)
    parser.add_argument("--max_pos_pairs_per_writer", type=int, default=40)
    parser.add_argument("--max_skilled_neg_pairs_per_writer", type=int, default=40)
    parser.add_argument("--num_random_neg_pairs_per_writer", type=int, default=20)
    args = parser.parse_args()

    manifest_df = pd.read_csv(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = split_writers(
        manifest_df["writer_id"].unique(),
        val_size=args.val_writers,
        test_size=args.test_writers,
    )

    # Luu lai danh sach writer trong tung split de kiem tra / tai lap sau nay.
    with open(output_dir / "writer_split.txt", "w") as f:
        for split_name, writers in splits.items():
            f.write(f"{split_name} ({len(writers)} writers): {writers}\n")

    for split_name, writers in splits.items():
        pairs_df = generate_pairs_for_split(
            manifest_df,
            writers,
            max_pos_pairs_per_writer=args.max_pos_pairs_per_writer,
            max_skilled_neg_pairs_per_writer=args.max_skilled_neg_pairs_per_writer,
            num_random_neg_pairs_per_writer=args.num_random_neg_pairs_per_writer,
        )
        out_path = output_dir / f"pairs_{split_name}.csv"
        pairs_df.to_csv(out_path, index=False)
        print(f"\n=== {split_name.upper()} ({len(writers)} writers) ===")
        print(f"Tong so cap: {len(pairs_df)}")
        print(pairs_df["pair_type"].value_counts())
        print(f"Ty le label (1=match): {pairs_df['label'].mean():.3f}")
        print(f"Da luu: {out_path}")


if __name__ == "__main__":
    main()