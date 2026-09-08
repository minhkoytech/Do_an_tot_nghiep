"""
06_siamese_network.py
----------------------
Siamese Network (CNN backbone + Contrastive Loss) - "Huong tiep can de
xuat chinh" trong de cuong (Buoc 5).

Kien truc:
    Anh reference va anh query duoc dua qua CUNG MOT CNN (shared weights)
    de tao ra 2 vector embedding. Khoang cach Euclidean giua 2 embedding
    duoc dung de quyet dinh match/non-match - CUNG FORMULATION voi
    Pairwise Classical ML baseline o 05_pairwise_baseline.py:
        (Reference, Query) -> Distance -> Match/Non-match
    Nho vay so sanh Siamese vs Pairwise RF/SVM la cong bang (dung gop y
    so 4 cua GVHD) - cung writer split, cung dinh nghia pair, cung cach
    danh gia.

Contrastive Loss (Hadsell et al.):
    L = (1-Y) * 0.5 * D^2  +  Y * 0.5 * max(0, margin - D)^2
    trong do Y = 1 neu KHONG khop (non-match), Y = 0 neu khop (match).
    Luu y: label trong pairs_*.csv dinh nghia NGUOC lai (label=1 la
    match), nen Y = 1 - label khi tinh loss.
    Y cua cong thuc: cap giong nhau (match) -> D cang nho cang tot (ep
    ve 0). Cap khac nhau (non-match) -> D cang lon cang tot, nhung chi
    can lon hon margin la du (khong ep vo cuc).

Ky luat train/val/test (giong het 05_pairwise_baseline.py):
    - Huan luyen tren train, EARLY STOPPING dua tren EER cua VAL (khong
      phai chi dua tren validation loss - EER moi la chi so thuc su
      quan tam trong bai toan xac thuc).
    - Threshold quyet dinh (EER) duoc CHON va CO DINH tren VAL.
    - Test set CHI danh gia DUY NHAT MOT LAN sau khi model + threshold
      da co dinh.

Sau khi danh gia Siamese tren test, script se TU DONG load lai RF/SVM
da luu o 05_pairwise_baseline.py (model_artifacts/) va tinh lai diem tren
CUNG test set, tao ra MOT bang so sanh + MOT bieu do ROC chung cho ca 3
model - day la ket qua so sanh Classical ML vs Deep Learning ma GVHD yeu
cau o gop y so 4.

Ensemble (mac dinh 3 model doc lap, --num_models de doi):
    Thay vi train 1 model duy nhat, script train NHIEU model Siamese
    doc lap (seed khoi tao khac nhau), roi lay TRUNG BINH score (khoang
    cach) cua ca ensemble de ra quyet dinh cuoi cung. Day la ky thuat
    giam phuong sai (variance reduction) chuan muc trong ML, thuong cho
    ket qua on dinh va chinh xac hon 1 model don le - danh doi la thoi
    gian train tang len (gap args.num_models lan).

Weighted Contrastive Loss:
    Cap skilled_forgery duoc nhan trong so 1.5x trong loss (so voi 1.0x
    cho genuine_genuine va random_forgery), vi day la truong hop KHO va
    QUAN TRONG NHAT trong ngan hang (dung uu tien cua GVHD o gop y so 1).

Output:
    model_artifacts/siamese_model_{0,1,2}.pt (trong so tung model trong ensemble)
    model_artifacts/siamese_threshold.json
    results/tables/siamese_test_summary.csv
    results/tables/siamese_per_pairtype.csv
    results/tables/siamese_per_writer.csv
    results/tables/final_model_comparison.csv        (RF vs SVM vs Ensemble Siamese)
    results/figures/siamese_training_curve.png       (cua model dau tien trong ensemble)
    results/figures/final_model_comparison_roc.png   (ROC 3 model tren cung 1 hinh)

Cach dung (khong can tham so, path da khop san):
    cd src
    python 06_siamese_network.py

Luu y ve thoi gian chay: voi ensemble 3 model, tong thoi gian se GAP 3
LAN so voi train 1 model. Neu ban da tang so pairs o 01b_generate_pairs.py
(khuyen nghi), thoi gian moi epoch cung tang tuong ung. Tren may CPU thong
thuong, tong thoi gian co the tu 30 phut den vai gio tuy cau hinh may va
so luong pairs/model. Neu muon nhanh hon, giam --num_models xuong 1 hoac 2.
Neu co GPU NVIDIA, script se TU DONG dung GPU (khong can chinh gi them).

Ky vong ve EER (de dua vao bao cao, tranh ky vong sai lech):
    Cac nghien cuu writer-independent (WI) nghiem tuc tren CEDAR thuong
    dat EER trong khoang 2-8% (Kalera 2004: 21.9%, Kumar 2012: 8.33%,
    Kumar&Puhan 2014: 6.02%, Guerbai 2015: 5.60%, Zois 2019: 2.90%).
    Mot so paper cong bo EER < 1% (vd SigNet cua Dey et al. 2017,
    SigScatNet) DEU dua tren PRETRAINING mot CNN lon (hang trieu tham
    so) tren mot bo du lieu PHU rat lon (GPDS voi 581 writer dung cho
    writer-identification pretext task) roi moi transfer sang CEDAR -
    day la loi the du lieu ma de cuong nay CHU DONG KHONG su dung (GVHD
    da xac nhan CEDAR la du, khong can them dataset khac). Vi vay EER
    dat duoc trong pham vi do an nay (chi dung CEDAR, khong pretrain
    tren bo du lieu ngoai) can duoc so sanh voi nhom phuong phap WI
    KHONG dung pretraining/transfer learning tu bo du lieu lon, thay vi
    so sanh truc tiep voi cac con so SOTA <1% - su so sanh do la khap
    khieng ve mat phuong phap luan.

Nguon tham khao (dua vao bao cao neu can trich dan):
    - Dey et al. (2017), "SigNet: Convolutional Siamese Network for
      Writer Independent Offline Signature Verification", arXiv:1707.02131
    - Souza et al. (2021), "A white-box analysis on the writer-independent
      dichotomy transformation...", arxiv:2004.03370 (bang tong hop EER
      cac phuong phap WI tren CEDAR, Table 12)
"""

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    roc_curve, roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
)

# ---------------------------------------------------------------------------
# Duong dan mac dinh khop voi cau truc project DO_AN_TOT_NGHIEP
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PAIRS_DIR = PROJECT_ROOT / "data" / "processed" / "pairs"
DEFAULT_FEATURES = PROJECT_ROOT / "data" / "processed" / "features.csv"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "model_artifacts"
DEFAULT_TABLES_DIR = PROJECT_ROOT / "results" / "tables"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

CANVAS_HEIGHT, CANVAS_WIDTH = 155, 220  # khop voi CANVAS_SIZE trong 01_preprocessing.py
PRETRAINED_IMG_SIZE = 224  # kich thuoc chuan ma ResNet18 duoc pretrain tren ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
RANDOM_SEED = 42

# Phai khop CHINH XAC voi SELECTED_FEATURES trong 05_pairwise_baseline.py
# de tai su dung model RF/SVM da luu cho phan so sanh cuoi.
SELECTED_FEATURES = [
    "glcm_correlation", "hu_1", "hu_2", "hu_3", "hu_4",
    "aspect_ratio", "pixel_density", "glcm_homogeneity",
    "glcm_energy", "num_crossings", "baseline_ratio",
]

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Dataset: doc cap anh truc tiep tu pairs_*.csv
# ---------------------------------------------------------------------------
class SignaturePairDataset(Dataset):
    """
    Neu augment=True (chi dung cho TRAIN), moi anh se duoc xoay + dich
    chuyen ngau nhien nhe (doc lap giua reference va query) truoc khi
    dua vao model. Muc dich: chong overfitting - neu khong augment, CNN
    de "hoc thuoc" vi tri/goc nghieng chinh xac cua tung mau chu ky
    trong tap train (chi ~35 writer, ~3500 cap) thay vi hoc dac trung
    hinh dang tong quat. Augmentation KHONG ap dung cho val/test vi do
    la du lieu dung de danh gia, phai giu nguyen.
    """

    def __init__(self, pairs_df: pd.DataFrame, augment: bool = False, use_pretrained: bool = False):
        self.pairs_df = pairs_df.reset_index(drop=True)
        self.augment = augment
        # use_pretrained=True: resize anh ve 224x224, nhan ban thanh 3
        # kenh (gia lap RGB), chuan hoa theo mean/std cua ImageNet - bat
        # buoc phai lam vay de tuong thich voi trong so pretrain cua
        # ResNet18. use_pretrained=False (mac dinh): giu nguyen pipeline
        # cu (grayscale 1 kenh, [0,1], dung cho CNN tu xay).
        self.use_pretrained = use_pretrained

    def __len__(self):
        return len(self.pairs_df)

    def _augment_image(self, img: np.ndarray) -> np.ndarray:
        """Xoay ngau nhien nho (+-8 do) + dich chuyen ngau nhien nho (+-8 pixel)."""
        h, w = img.shape
        angle = np.random.uniform(-8, 8)
        tx = np.random.randint(-8, 9)
        ty = np.random.randint(-8, 9)
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        matrix[0, 2] += tx
        matrix[1, 2] += ty
        return cv2.warpAffine(img, matrix, (w, h), borderValue=0)

    def _load_image(self, path: str) -> torch.Tensor:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Khong doc duoc anh: {path}")
        if self.augment:
            img = self._augment_image(img)

        if self.use_pretrained:
            img = cv2.resize(img, (PRETRAINED_IMG_SIZE, PRETRAINED_IMG_SIZE), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            img_3ch = np.stack([img, img, img], axis=0)  # gia lap 3 kenh RGB tu grayscale
            tensor = torch.from_numpy(img_3ch)
            mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
            tensor = (tensor - mean) / std
            return tensor
        else:
            img = img.astype(np.float32) / 255.0  # chuan hoa ve [0, 1]
            return torch.from_numpy(img).unsqueeze(0)  # shape (1, H, W)

    def __getitem__(self, idx):
        row = self.pairs_df.iloc[idx]
        img_ref = self._load_image(row["reference_path"])
        img_query = self._load_image(row["query_path"])
        label = torch.tensor(row["label"], dtype=torch.float32)
        # Trong so cho loss: skilled forgery duoc phat nang hon (weight=1.5)
        # so voi genuine_genuine va random_forgery (weight=1.0), vi day la
        # truong hop KHO va QUAN TRONG NHAT trong ngan hang (dung uu tien
        # cua GVHD o gop y so 1) - buoc model tap trung hoc phan biet tot
        # hon cho dung truong hop nay thay vi doi xu binh dang voi moi loai
        # cap negative.
        weight = 1.5 if row["pair_type"] == "skilled_forgery" else 1.0
        weight = torch.tensor(weight, dtype=torch.float32)
        return img_ref, img_query, label, weight


# ---------------------------------------------------------------------------
# Kien truc Siamese Network (CNN backbone dung chung, nhe de train tren CPU)
# ---------------------------------------------------------------------------
class EmbeddingCNN(nn.Module):
    """
    CNN backbone tao embedding tu 1 anh chu ky.

    Kien truc 5 lop conv (dua theo do sau cua SigNet - Dey et al. 2017),
    ~3.1 trieu tham so - khop voi muc tham chieu "Only Siamese Network"
    trong nghien cuu SigScatNet (3,327,056 tham so, dat EER 0.069% tren
    CEDAR - xem Table III trong arxiv:2311.05579). Day la muc do sau/
    dung luong hop ly cho bai toan nay khi KHONG dung pretraining tu bo
    du lieu ngoai.

    BatchNorm2d/BatchNorm1d sau moi lop: on dinh va tang toc hoi tu,
    dong thoi co tac dung regularize nhe.

    AdaptiveAvgPool2d((4,4)) truoc FC de kiem soat kich thuoc FC dau
    tien, tranh so tham so bung no khi flatten truc tiep feature map.
    """

    def __init__(self, embedding_dim=64, dropout=0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),     # -> 32 x 77 x 110
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),    # -> 64 x 38 x 55
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),  # -> 128 x 19 x 27
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2), # -> 256 x 9 x 13
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),                                                                    # -> 256 x 4 x 4 (co dinh)
        )
        flat_dim = 256 * 4 * 4
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


class ResNetEmbedding(nn.Module):
    """
    Embedding network dung ResNet18 pretrain tren ImageNet (transfer
    learning) thay vi CNN tu xay tu dau.

    Vi sao thu huong nay: khi tang so luong pairs training (tu 3500 len
    13125) khong cai thien duoc EER (van dung o ~0.26-0.27), dieu do cho
    thay gioi han khong nam o SO LUONG PAIRS ma o SO LUONG WRITER/anh
    goc (chi 35 writer trong tap train) - tang to hop cap tu CUNG mot
    tap anh co dinh khong tao ra thong tin moi. Transfer learning la
    huong con lai HOP LE trong pham vi CEDAR-only: dung trong so da hoc
    tu ImageNet (mot bo du lieu ANH TONG QUAT, khong phai chu ky) lam
    diem khoi tao, thay vi random init - cac lop dau cua CNN pretrain
    tren ImageNet da hoc duoc cac bo loc phat hien canh/net/texture co
    ban, co the huu ich ngay ca voi anh chu ky (von cung la net ve/canh).

    Dong bang cac lop dau (conv1, bn1, layer1, layer2) - giu nguyen dac
    trung tong quat da hoc duoc. Fine-tune layer3, layer4 va embedding
    head moi - cho phep model dieu chinh cac dac trung muc cao hon de
    phu hop voi dac diem rieng cua chu ky.
    """

    def __init__(self, embedding_dim=64, dropout=0.5, freeze_early_layers=True):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        backbone = torchvision.models.resnet18(weights=weights)

        if freeze_early_layers:
            for name, param in backbone.named_parameters():
                if name.startswith("layer3") or name.startswith("layer4") or name.startswith("fc"):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        backbone.fc = nn.Identity()  # bo lop phan loai goc (1000 lop ImageNet), giu output 512-dim
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


class SiameseNetwork(nn.Module):
    """Boc embedding network (CNN tu xay hoac ResNet18 pretrain) trong Siamese + tinh khoang cach Euclidean."""

    def __init__(self, embedding_dim=64, dropout=0.5, backbone_type="custom"):
        super().__init__()
        if backbone_type == "resnet18":
            self.embedding_net = ResNetEmbedding(embedding_dim, dropout, freeze_early_layers=True)
        else:
            self.embedding_net = EmbeddingCNN(embedding_dim, dropout)

    def forward(self, img1, img2):
        emb1 = self.embedding_net(img1)
        emb2 = self.embedding_net(img2)
        distance = torch.nn.functional.pairwise_distance(emb1, emb2)
        return distance


class ContrastiveLoss(nn.Module):
    """L = (1-Y)*0.5*D^2 + Y*0.5*max(0, margin-D)^2 ; Y=1 neu non-match.
    Ho tro trong so per-sample (weight) de phat nang hon cac cap kho/quan
    trong hon (skilled forgery) - xem giai thich o SignaturePairDataset.
    """

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, distance, label, weight=None):
        y_dissimilar = 1 - label  # label=1 (match) -> y=0 ; label=0 (non-match) -> y=1
        loss_similar = (1 - y_dissimilar) * 0.5 * distance.pow(2)
        loss_dissimilar = y_dissimilar * 0.5 * torch.clamp(self.margin - distance, min=0).pow(2)
        per_sample_loss = loss_similar + loss_dissimilar
        if weight is not None:
            per_sample_loss = per_sample_loss * weight
        return per_sample_loss.mean()


# ---------------------------------------------------------------------------
# EER / FAR / FRR - giong het 05_pairwise_baseline.py de nhat quan
# ---------------------------------------------------------------------------
def find_eer_threshold(y_true, scores):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    frr = 1 - tpr
    idx = np.argmin(np.abs(fpr - frr))
    return thresholds[idx], (fpr[idx] + frr[idx]) / 2


def compute_far_frr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    frr = fn / (fn + tp) if (fn + tp) > 0 else np.nan
    return far, frr


def parse_writer_split(writer_split_path, split_name):
    with open(writer_split_path, "r") as f:
        for line in f:
            if line.strip().startswith(split_name):
                bracket_content = line.split(":", 1)[1]
                return [int(x) for x in re.findall(r"\d+", bracket_content)]
    raise ValueError(f"Khong tim thay split '{split_name}'")


# ---------------------------------------------------------------------------
# Tinh diem (distance) cho toan bo mot DataLoader, khong backprop
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_scores(model, loader, device):
    model.eval()
    all_scores, all_labels = [], []
    for img1, img2, label, _weight in loader:
        img1, img2 = img1.to(device), img2.to(device)
        distance = model(img1, img2)
        # score = -distance: khoang cach cang NHO nghia la cang GIONG NHAU
        # (cang co kha nang la match) -> dao dau de "score cao = match",
        # nhat quan voi quy uoc score trong 05_pairwise_baseline.py
        all_scores.append((-distance).cpu().numpy())
        all_labels.append(label.numpy())
    return np.concatenate(all_labels), np.concatenate(all_scores)


def train_siamese(model, train_loader, val_loader, device, epochs, patience, lr, margin, model_dir, weight_decay=1e-4, model_name="siamese_best"):
    criterion = ContrastiveLoss(margin=margin)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    # Giam learning rate khi val EER khong cai thien sau 3 epoch - giup model
    # hoi tu tinh te hon truoc khi early stopping kich hoat, giam overfitting
    # so voi giu nguyen learning rate cao suot qua trinh train.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_eer = np.inf
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_eer": []}
    best_model_path = model_dir / f"{model_name}.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for img1, img2, label, weight in train_loader:
            img1, img2, label, weight = img1.to(device), img2.to(device), label.to(device), weight.to(device)
            optimizer.zero_grad()
            distance = model(img1, img2)
            loss = criterion(distance, label, weight)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_loss = float(np.mean(epoch_losses))
        val_labels, val_scores = compute_scores(model, val_loader, device)
        _, val_eer = find_eer_threshold(val_labels, val_scores)
        scheduler.step(val_eer)

        history["train_loss"].append(train_loss)
        history["val_eer"].append(val_eer)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{epochs} | train_loss={train_loss:.4f} | val_EER={val_eer:.4f} | lr={current_lr:.6f}")

        if val_eer < best_val_eer:
            best_val_eer = val_eer
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Val EER cai thien, da luu model tot nhat (EER={best_val_eer:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"  -> Early stopping (khong cai thien sau {patience} epoch)")
                break

    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    return model, history, best_val_eer


def plot_training_curve(history, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"])
    axes[0].set_title("Train Contrastive Loss")
    axes[0].set_xlabel("Epoch")
    axes[1].plot(history["val_eer"])
    axes[1].set_title("Validation EER")
    axes[1].set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Da luu bieu do training: {out_path}")


def evaluate_model(y_true, scores, threshold, model_name, pairs_df):
    y_pred = (scores >= threshold).astype(int)
    far, frr = compute_far_frr(y_true, y_pred)
    overall = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, scores),
        "FAR": far,
        "FRR": frr,
        "threshold": threshold,
    }

    eval_df = pairs_df.copy()
    eval_df["y_pred"] = y_pred

    per_pairtype = []
    for pt in ["skilled_forgery", "random_forgery"]:
        mask = eval_df["pair_type"] == pt
        if mask.sum() == 0:
            continue
        false_accept = np.mean(eval_df.loc[mask, "y_pred"] == 1)
        per_pairtype.append({"model": model_name, "pair_type": pt, "n": int(mask.sum()), "FAR": false_accept})
    mask = eval_df["pair_type"] == "genuine_genuine"
    if mask.sum() > 0:
        false_reject = np.mean(eval_df.loc[mask, "y_pred"] == 0)
        per_pairtype.append({"model": model_name, "pair_type": "genuine_genuine", "n": int(mask.sum()), "FRR": false_reject})

    eval_df["correct"] = (eval_df["y_pred"] == eval_df["label"]).astype(int)
    per_writer = []
    for writer_id, group in eval_df.groupby("writer_id_ref"):
        w_far, w_frr = compute_far_frr(group["label"].values, group["y_pred"].values)
        per_writer.append({
            "model": model_name, "writer_id": writer_id, "n_pairs": len(group),
            "accuracy": group["correct"].mean(), "FAR": w_far, "FRR": w_frr,
        })

    return overall, pd.DataFrame(per_pairtype), pd.DataFrame(per_writer)


# ---------------------------------------------------------------------------
# Tai lai RF/SVM da luu o 05_pairwise_baseline.py de so sanh cong bang
# ---------------------------------------------------------------------------
def compute_pair_deltas(pairs_df, features_df, feature_cols):
    feat_indexed = features_df.set_index("path")[feature_cols]
    merged = pairs_df.merge(feat_indexed.add_suffix("_ref"), left_on="reference_path", right_index=True, how="inner")
    merged = merged.merge(feat_indexed.add_suffix("_query"), left_on="query_path", right_index=True, how="inner")
    for col in feature_cols:
        merged[f"delta_{col}"] = (merged[f"{col}_ref"] - merged[f"{col}_query"]).abs()
    delta_cols = [f"delta_{c}" for c in feature_cols]
    return merged, delta_cols


def try_load_classical_baselines(model_dir: Path, features_path: Path, test_pairs_df: pd.DataFrame):
    """
    Neu da chay 05_pairwise_baseline.py truoc do, load lai RF/SVM + scaler
    + imputer + threshold, tinh diem tren CUNG test set (khong huan luyen
    lai), de dua vao bang so sanh cuoi cung. Neu chua co, bo qua nhe nhang.
    """
    required_files = ["pairwise_rf.joblib", "pairwise_svm.joblib", "pairwise_scaler.joblib",
                       "pairwise_imputer.joblib", "pairwise_threshold.json"]
    if not all((model_dir / f).exists() for f in required_files):
        print("\n[THONG BAO] Chua tim thay model RF/SVM da luu (chay 05_pairwise_baseline.py truoc "
              "neu muon co bang so sanh day du). Bo qua phan so sanh voi Classical ML.")
        return {}

    if not features_path.exists():
        print("\n[THONG BAO] Khong tim thay features.csv, bo qua phan so sanh voi Classical ML.")
        return {}

    features_df = pd.read_csv(features_path)
    rf = joblib.load(model_dir / "pairwise_rf.joblib")
    svm = joblib.load(model_dir / "pairwise_svm.joblib")
    scaler = joblib.load(model_dir / "pairwise_scaler.joblib")
    imputer = joblib.load(model_dir / "pairwise_imputer.joblib")
    with open(model_dir / "pairwise_threshold.json") as f:
        thresholds = json.load(f)

    merged, delta_cols = compute_pair_deltas(test_pairs_df, features_df, SELECTED_FEATURES)
    X_test = scaler.transform(imputer.transform(merged[delta_cols].values))
    y_test = merged["label"].values

    results = {}
    rf_scores = rf.predict_proba(X_test)[:, 1]
    results["rf"] = evaluate_model(y_test, rf_scores, thresholds["rf_threshold"], "rf", merged)
    svm_scores = svm.decision_function(X_test)
    results["svm"] = evaluate_model(y_test, svm_scores, thresholds["svm_threshold"], "svm", merged)
    results["_raw_scores"] = {"rf": (y_test, rf_scores), "svm": (y_test, svm_scores)}
    print("\nDa load lai RF/SVM da luu va tinh diem tren test set de so sanh.")
    return results


def plot_combined_roc(scores_dict: dict, out_path: Path):
    plt.figure(figsize=(6.5, 6.5))
    for model_name, (y_true, scores) in scores_dict.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        plt.plot(fpr, tpr, label=f"{model_name.upper()} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("FAR (False Acceptance Rate)")
    plt.ylabel("1 - FRR (True Accept Rate)")
    plt.title("So sanh ROC: Classical ML Pairwise vs Siamese Network (Test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Da luu bieu do ROC so sanh: {out_path}")


@torch.no_grad()
def compute_ensemble_scores(models: list, loader, device):
    """
    Tinh score trung binh tren toan bo ensemble (list cac model da train).
    Ky thuat ensemble averaging: giam phuong sai cua du doan bang cach
    lay trung binh nhieu model duoc train doc lap (seed khac nhau) -
    thuong cho ket qua on dinh va tot hon mot model don le, voi dieu
    kien cac model du "da dang" (khac seed/augmentation ngau nhien).
    """
    all_scores_per_model = []
    labels_ref = None
    for model in models:
        labels, scores = compute_scores(model, loader, device)
        if labels_ref is None:
            labels_ref = labels
        all_scores_per_model.append(scores)
    ensemble_scores = np.mean(np.stack(all_scores_per_model, axis=0), axis=0)
    return labels_ref, ensemble_scores


def main():
    parser = argparse.ArgumentParser(description="Siamese Network cho xac thuc chu ky")
    parser.add_argument("--pairs_dir", default=str(DEFAULT_PAIRS_DIR))
    parser.add_argument("--features", default=str(DEFAULT_FEATURES))
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--tables_dir", default=str(DEFAULT_TABLES_DIR))
    parser.add_argument("--figures_dir", default=str(DEFAULT_FIGURES_DIR))
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--backbone", choices=["custom", "resnet18"], default="resnet18",
                         help="'custom': CNN tu xay tu dau (nhe, nhanh hon tren CPU). "
                              "'resnet18': transfer learning tu ResNet18 pretrain ImageNet, "
                              "dong bang lop dau, fine-tune layer3/layer4 + head moi (anh phai "
                              "resize 224x224 va nhan 3 kenh, nang hon va cham hon 'custom' tren CPU "
                              "nhung co the manh hon nho tan dung dac trung da hoc san).")
    parser.add_argument("--num_models", type=int, default=3,
                         help="So luong model trong ensemble. Moi model duoc train doc lap voi "
                              "seed khac nhau, ket qua cuoi la trung binh score cua ca ensemble. "
                              "Tang so nay (vd 5) neu muon on dinh hon nhung se ton nhieu thoi gian hon.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dang su dung device: {device}")
    print(f"Backbone: {args.backbone}")

    pairs_dir = Path(args.pairs_dir)
    train_pairs = pd.read_csv(pairs_dir / "pairs_train.csv")
    val_pairs = pd.read_csv(pairs_dir / "pairs_val.csv")
    test_pairs = pd.read_csv(pairs_dir / "pairs_test.csv")
    print(f"Train pairs: {len(train_pairs)} | Val pairs: {len(val_pairs)} | Test pairs: {len(test_pairs)}")

    use_pretrained = (args.backbone == "resnet18")
    train_loader = DataLoader(SignaturePairDataset(train_pairs, augment=True, use_pretrained=use_pretrained), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(SignaturePairDataset(val_pairs, augment=False, use_pretrained=use_pretrained), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(SignaturePairDataset(test_pairs, augment=False, use_pretrained=use_pretrained), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Train ENSEMBLE gom args.num_models model doc lap, seed khac nhau
    # -----------------------------------------------------------------
    trained_models = []
    all_histories = []
    for i in range(args.num_models):
        seed = RANDOM_SEED + i
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"\n{'='*70}\nTRAIN MODEL {i+1}/{args.num_models} (seed={seed})\n{'='*70}")

        model = SiameseNetwork(embedding_dim=args.embedding_dim, dropout=args.dropout, backbone_type=args.backbone).to(device)
        if i == 0:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Tong so tham so: {total_params:,} | So tham so duoc train: {trainable_params_count:,}")

        model, history, best_val_eer = train_siamese(
            model, train_loader, val_loader, device,
            epochs=args.epochs, patience=args.patience, lr=args.lr, margin=args.margin,
            model_dir=model_dir, weight_decay=args.weight_decay, model_name=f"siamese_model_{i}",
        )
        trained_models.append(model)
        all_histories.append(history)
        print(f"Model {i+1} hoan tat, best val EER = {best_val_eer:.4f}")

    plot_training_curve(all_histories[0], figures_dir / "siamese_training_curve.png")

    # -----------------------------------------------------------------
    # Threshold (EER) CO DINH tu VAL - dua tren SCORE TRUNG BINH CUA ENSEMBLE
    # -----------------------------------------------------------------
    val_labels, val_scores = compute_ensemble_scores(trained_models, val_loader, device)
    threshold, ensemble_val_eer = find_eer_threshold(val_labels, val_scores)
    print(f"\nThreshold Ensemble Siamese (tu val, EER={ensemble_val_eer:.4f}): {threshold:.4f}")

    with open(model_dir / "siamese_threshold.json", "w") as f:
        json.dump({
            "threshold": float(threshold), "embedding_dim": args.embedding_dim,
            "margin": args.margin, "num_models": args.num_models,
        }, f, indent=2)

    # -----------------------------------------------------------------
    # Danh gia TREN TEST - CHI MOT LAN, dung score trung binh ensemble
    # -----------------------------------------------------------------
    print("\n=== Danh gia Ensemble Siamese tren TEST (mot lan duy nhat) ===")
    test_labels, test_scores = compute_ensemble_scores(trained_models, test_loader, device)
    siamese_overall, siamese_pairtype, siamese_writer = evaluate_model(
        test_labels, test_scores, threshold, "siamese", test_pairs
    )

    pd.DataFrame([siamese_overall]).to_csv(tables_dir / "siamese_test_summary.csv", index=False)
    siamese_pairtype.to_csv(tables_dir / "siamese_per_pairtype.csv", index=False)
    siamese_writer.to_csv(tables_dir / "siamese_per_writer.csv", index=False)

    print(pd.DataFrame([siamese_overall]).to_string(index=False))
    print(f"\n=== FAR/FRR theo loai gia mao (Siamese) ===\n{siamese_pairtype.to_string(index=False)}")

    # -----------------------------------------------------------------
    # So sanh voi RF/SVM da luu (neu co) - dung gop y so 4 cua GVHD
    # -----------------------------------------------------------------
    classical_results = try_load_classical_baselines(model_dir, Path(args.features), test_pairs)

    all_overall = [siamese_overall]
    combined_scores = {"siamese": (test_labels, test_scores)}
    if classical_results:
        for name in ["rf", "svm"]:
            overall, _, _ = classical_results[name]
            all_overall.append(overall)
        combined_scores.update(classical_results["_raw_scores"])

    comparison_df = pd.DataFrame(all_overall)
    comparison_path = tables_dir / "final_model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n=== BANG SO SANH CUOI CUNG (Pairwise RF/SVM vs Ensemble Siamese Network) ===")
    print(comparison_df.to_string(index=False))
    print(f"Da luu: {comparison_path}")

    plot_combined_roc(combined_scores, figures_dir / "final_model_comparison_roc.png")

    print(f"\nHoan tat. {args.num_models} model Siamese da luu trong: {model_dir}")


if __name__ == "__main__":
    main()