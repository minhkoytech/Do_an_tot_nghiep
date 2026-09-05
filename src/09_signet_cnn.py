"""
Bước 9 — CNN theo kiến trúc SigNet (Dey et al., 2017)
"SigNet: Convolutional Siamese Network for Writer Independent Offline
Signature Verification" — arxiv.org/abs/1707.02131
Mã nguồn tham khảo: github.com/sounakdey/SigNet/blob/master/SigNet_v1.py

Kiến trúc CHUYÊN DÙNG cho bài toán xác thực chữ ký (dạng AlexNet):
  Conv1: 96 kernel 11x11 -> BN -> ReLU -> MaxPool -> Dropout
  Conv2: 256 kernel 5x5  -> BN -> ReLU -> MaxPool -> Dropout
  Conv3: 384 kernel 3x3  -> ReLU
  Conv4: 256 kernel 3x3  -> ReLU -> MaxPool -> Dropout
  FC1:   1024            -> ReLU -> Dropout
  FC2:   128 (embedding)

(Bài gốc dùng Local Response Normalization — thay bằng BatchNorm, cách
thay thế phổ biến trong cài đặt hiện đại.)

LƯU Ý KỲ VỌNG: bài gốc báo cáo 100% trên CEDAR, nhưng các nghiên cứu dùng
lại kiến trúc này chỉ đạt 90-95.66%; giới nghiên cứu khuyến cáo thận trọng
với con số 100%. Với writer-independent split nghiêm ngặt, mục tiêu thực
tế là 85-90%.
"""
import json
import random
import time
from itertools import combinations
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import cv2

from utils_metrics import evaluate_predictions, find_threshold_by_eer, find_threshold_at_target_far

PROC_DIR = Path("data/processed")
RESULTS_DIR = Path("results/tables/signet_cnn")
MODEL_DIR = Path("model_artifacts")

IMG_SIZE = 155
BATCH_SIZE = 64
LR = 1e-5
MARGIN = 1.0
MAX_EPOCHS = 400          # tăng mạnh, giống notebook tham khảo (họ chạy tới ~416 epoch)
PATIENCE = 40              # tăng mạnh từ 12->40, kiên nhẫn hơn nhiều trước khi dừng
WEIGHT_DECAY = 5e-4
RANDOM_SEED = 42
TARGET_FAR = 0.10
MAX_TRAIN_PAIRS = 12000

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cpu")


def load_writer_images(writer_ids, labels_df):
    data = {}
    for wid in writer_ids:
        sub = labels_df[labels_df["writer_id"] == wid]
        data[wid] = {
            "genuine": sub[sub["label"] == "genuine"]["filename"].tolist(),
            "skilled_forgery": sub[sub["label"] == "skilled_forgery"]["filename"].tolist(),
        }
    return data


_image_cache = {}

def load_image(filename):
    if filename in _image_cache:
        return _image_cache[filename].copy()
    img = cv2.imread(str(PROC_DIR / filename), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = (img.astype(np.float32) / 255.0)
    _image_cache[filename] = img
    return img.copy()


def build_fixed_pairs(writer_data, rng, max_pairs=None):
    writer_ids = [w for w, d in writer_data.items() if len(d["genuine"]) >= 2]
    rows = []
    for wid in writer_ids:
        genuine = writer_data[wid]["genuine"]
        forged = writer_data[wid]["skilled_forgery"]
        for a, b in combinations(genuine, 2):
            rows.append((a, b, 1))
        for a in genuine:
            for b in forged:
                rows.append((a, b, 0))
    n_random = sum(1 for r in rows if r[2] == 0)
    for _ in range(n_random):
        w1, w2 = rng.sample(writer_ids, 2)
        a = rng.choice(writer_data[w1]["genuine"])
        b = rng.choice(writer_data[w2]["genuine"])
        rows.append((a, b, 0))
    rng.shuffle(rows)
    if max_pairs is not None and len(rows) > max_pairs:
        rows = rows[:max_pairs]
    return rows


class FixedPairDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        fa, fb, label = self.pairs[idx]
        img_a = torch.from_numpy(load_image(fa)).unsqueeze(0)
        img_b = torch.from_numpy(load_image(fb)).unsqueeze(0)
        return img_a, img_b, torch.tensor(label, dtype=torch.float32)


class SigNetEmbedding(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.conv1 = nn.Sequential(
            # QUAN TRỌNG: stride=2 (thay vì 1 như bản gốc SigNet) — bản gốc
            # thiết kế cho GPU, xử lý toàn bộ độ phân giải đầu vào ở lớp đầu
            # tiên (155x155 x 96 kênh) là quá nặng cho CPU (~2.4 tiếng/epoch
            # đo thực tế). stride=2 giảm ngay kích thước đặc trưng từ đầu,
            # giữ tinh thần kiến trúc nhưng khả thi về thời gian.
            nn.Conv2d(1, 96, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm2d(96), nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.3),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.3),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.3),
        )
        self.gap = nn.AdaptiveAvgPool2d((6, 6))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, embedding_dim),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.gap(x)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.embedding_net = SigNetEmbedding(embedding_dim)

    def forward(self, x1, x2):
        return self.embedding_net(x1), self.embedding_net(x2)


def contrastive_loss(e1, e2, label, margin=MARGIN):
    dist = F.pairwise_distance(e1, e2)
    loss_pos = label * dist.pow(2)
    loss_neg = (1 - label) * F.relu(margin - dist).pow(2)
    return (loss_pos + loss_neg).mean(), dist


def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    all_dist, all_label = [], []
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for img_a, img_b, label in loader:
            img_a, img_b, label = img_a.to(device), img_b.to(device), label.to(device)
            e1, e2 = model(img_a, img_b)
            loss, dist = contrastive_loss(e1, e2, label)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            total_loss += loss.item() * len(label)
            all_dist.extend(dist.detach().numpy().tolist())
            all_label.extend(label.numpy().tolist())
    return total_loss / len(loader.dataset), np.array(all_dist), np.array(all_label)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(PROC_DIR / "labels.csv")
    with open(PROC_DIR / "writer_splits.json", encoding="utf-8") as f:
        split = json.load(f)["splits"][0]

    train_data = load_writer_images(split["train_writers"], labels_df)
    val_data = load_writer_images(split["val_writers"], labels_df)
    test_data = load_writer_images(split["test_writers"], labels_df)

    rng = random.Random(RANDOM_SEED)
    train_pairs = build_fixed_pairs(train_data, rng, max_pairs=MAX_TRAIN_PAIRS)
    val_pairs = build_fixed_pairs(val_data, rng, max_pairs=2000)
    print(f"Số cặp: train={len(train_pairs)}, val={len(val_pairs)}")

    train_loader = DataLoader(FixedPairDataset(train_pairs), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(FixedPairDataset(val_pairs), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SiameseNetwork().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Tổng tham số (train từ đầu, kiến trúc SigNet): {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    print(f"\nBắt đầu train — ảnh {IMG_SIZE}x{IMG_SIZE}, batch={BATCH_SIZE}, LR={LR}, tối đa {MAX_EPOCHS} epoch\n")

    t0 = time.time()
    epoch = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        ep_t0 = time.time()
        train_loss, _, _ = run_epoch(model, train_loader, optimizer)
        val_loss, val_dist, val_label = run_epoch(model, val_loader, optimizer=None)
        ep_time = time.time() - ep_t0

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = " <- tốt nhất, đã lưu"
            torch.save(best_state, MODEL_DIR / "signet_cnn_model.pt")
        else:
            patience_counter += 1

        print(f"[Epoch {epoch:03d}/{MAX_EPOCHS}] train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  ({ep_time:.0f}s/epoch){marker}")

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping tại epoch {epoch}")
            break

    total_time = time.time() - t0
    print(f"\nTổng thời gian train: {total_time/60:.1f} phút")

    model.load_state_dict(best_state)

    _, val_dist_final, val_label_final = run_epoch(model, val_loader, optimizer=None)
    val_scores = -val_dist_final
    thr_eer_score, eer_val = find_threshold_by_eer(val_label_final, val_scores)
    thr_far_score, achieved_far = find_threshold_at_target_far(val_label_final, val_scores, TARGET_FAR)
    print(f"\n>>> [EER cân bằng] EER trên val = {eer_val:.4f}")
    print(f">>> [FAR mục tiêu {TARGET_FAR:.0%}] FAR đạt được trên val = {achieved_far:.4f}")

    test_pairs_meta = []
    rng_test = random.Random(RANDOM_SEED)
    for wid, d in test_data.items():
        genuine, forged = d["genuine"], d["skilled_forgery"]
        if len(genuine) < 2:
            continue
        ref = rng_test.choice(genuine)
        for q in genuine:
            if q != ref:
                test_pairs_meta.append((ref, q, 1, "positive", wid))
        for q in forged:
            test_pairs_meta.append((ref, q, 0, "negative_skilled", wid))

    model.eval()
    test_dist, test_label, test_pair_type, test_writer = [], [], [], []
    with torch.no_grad():
        for fa, fb, label, ptype, wid in test_pairs_meta:
            img_a = torch.from_numpy(load_image(fa)).unsqueeze(0).unsqueeze(0)
            img_b = torch.from_numpy(load_image(fb)).unsqueeze(0).unsqueeze(0)
            e1, e2 = model(img_a, img_b)
            test_dist.append(F.pairwise_distance(e1, e2).item())
            test_label.append(label)
            test_pair_type.append(ptype)
            test_writer.append(wid)

    test_dist = np.array(test_dist)
    test_label = np.array(test_label)
    test_scores = -test_dist

    all_results = {}
    for name, thr_score in [("EER_balanced", thr_eer_score), (f"FAR_target_{TARGET_FAR:.0%}", thr_far_score)]:
        thr_dist = -thr_score
        y_pred = (test_dist <= thr_dist).astype(int)
        result, per_writer_df = evaluate_predictions(
            test_label, y_pred, test_scores,
            pair_type=np.array(test_pair_type), writer_id=np.array(test_writer)
        )
        all_results[name] = result
        print(f"\n=== KẾT QUẢ TEST SET — [{name}] ===")
        for k, v in result.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        pd.DataFrame([result]).to_csv(RESULTS_DIR / f"test_results_{name}.csv", index=False)
        per_writer_df.to_csv(RESULTS_DIR / f"per_writer_{name}.csv", index=False)

    print(f"\nĐã lưu kết quả vào {RESULTS_DIR}/")
    print("\n>>> So sánh nhanh 2 threshold:")
    for name, r in all_results.items():
        print(f"  [{name}] Accuracy={r['accuracy']:.4f}  FAR={r['FAR']:.4f}  FRR={r['FRR']:.4f}  ROC_AUC={r['ROC_AUC']:.4f}")


if __name__ == "__main__":
    main()