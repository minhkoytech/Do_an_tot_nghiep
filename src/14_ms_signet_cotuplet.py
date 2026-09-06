"""
Bước 14 — MS-SigNet + Co-tuplet Loss (PHIÊN BẢN XẤP XỈ CÓ CƠ SỞ)

Tham khảo: Huang & Lu (2023) "Multiscale Feature Learning Using Co-Tuplet
Loss for Offline Handwritten Signature Verification" — arxiv.org/abs/2308.00428
Mã nguồn gốc: github.com/ashleyfhh/MS-SigNet

QUAN TRỌNG — ĐÂY LÀ PHIÊN BẢN XẤP XỈ, KHÔNG PHẢI SAO CHÉP NGUYÊN VĂN:
Vì không có quyền truy cập đầy đủ công thức chính xác của "co-tuplet loss"
trong bài báo gốc, phiên bản này hiện thực hóa ĐÚNG TINH THẦN mô tả bằng
kỹ thuật đã được công nhận rộng rãi:

1. ĐA TỶ LỆ (Multi-scale): trích đặc trưng ở ảnh toàn cục VÀ 2 vùng cục bộ
   (nửa trên / nửa dưới — "dual-orientation regions"), nối lại thành 1
   vector đặc trưng cuối cùng.

2. LOSS XEM XÉT NHIỀU MẪU CÙNG LÚC: dùng Multi-Similarity Loss (Wang et al.,
   2019 — kỹ thuật chuẩn trong metric learning). Với mỗi anchor, xem xét
   NHIỀU positive và NHIỀU negative cùng lúc, tự động dồn trọng số vào các
   mẫu "khó" — đúng tinh thần co-tuplet loss mô tả.

KỲ VỌNG THỰC TẾ: KHÔNG đảm bảo đạt EER 3.51%/AUC 99.47% như bài gốc (họ
dùng GPU, tinh chỉnh sâu, công thức loss chính xác của họ).
"""
import json
import random
import time
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
RESULTS_DIR = Path("results/tables/ms_signet_cotuplet")
MODEL_DIR = Path("model_artifacts")

IMG_SIZE = 128
BATCH_SIZE = 8
N_POSITIVE = 3
N_NEGATIVE = 4
LR = 1e-5
MAX_EPOCHS = 150
PATIENCE = 15
WEIGHT_DECAY = 5e-4
RANDOM_SEED = 42
TARGET_FAR = 0.10
N_TUPLETS_TRAIN = 1500
N_TUPLETS_VAL = 300

MS_ALPHA = 2.0
MS_BETA = 50.0
MS_LAMBDA = 0.5

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


def build_tuplets(writer_data, rng, n_tuplets):
    writer_ids = [w for w, d in writer_data.items()
                  if len(d["genuine"]) >= N_POSITIVE + 1 and len(d["skilled_forgery"]) >= 1]
    tuplets = []
    for _ in range(n_tuplets):
        wid = rng.choice(writer_ids)
        genuine = writer_data[wid]["genuine"]
        forged = writer_data[wid]["skilled_forgery"]

        chosen = rng.sample(genuine, N_POSITIVE + 1)
        anchor, positives = chosen[0], chosen[1:]

        n_skilled = min(N_NEGATIVE - 1, len(forged))
        negatives = rng.sample(forged, n_skilled) if n_skilled > 0 else []
        n_random_needed = N_NEGATIVE - len(negatives)
        for _ in range(n_random_needed):
            other_wid = rng.choice([w for w in writer_ids if w != wid])
            negatives.append(rng.choice(writer_data[other_wid]["genuine"]))

        tuplets.append((anchor, positives, negatives))
    return tuplets


class TupletDataset(Dataset):
    def __init__(self, tuplets):
        self.tuplets = tuplets

    def __len__(self):
        return len(self.tuplets)

    def __getitem__(self, idx):
        anchor, positives, negatives = self.tuplets[idx]
        anchor_img = torch.from_numpy(load_image(anchor)).unsqueeze(0)
        pos_imgs = torch.stack([torch.from_numpy(load_image(p)) for p in positives]).unsqueeze(1)
        neg_imgs = torch.stack([torch.from_numpy(load_image(n)) for n in negatives]).unsqueeze(1)
        return anchor_img, pos_imgs, neg_imgs


class ScaleBranch(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 4, out_dim), nn.ReLU())

    def forward(self, x):
        return self.fc(self.conv(x))


class MultiScaleEmbedding(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.global_branch = ScaleBranch(out_dim=64)
        self.region_branch = ScaleBranch(out_dim=64)
        self.head = nn.Sequential(
            nn.Linear(64 + 64 + 64, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, embedding_dim),
        )

    def forward(self, x):
        h = x.shape[2]
        top = x[:, :, :h // 2, :]
        bottom = x[:, :, h // 2:, :]

        top = F.interpolate(top, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        bottom = F.interpolate(bottom, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)

        emb_global = self.global_branch(x)
        emb_top = self.region_branch(top)
        emb_bottom = self.region_branch(bottom)

        combined = torch.cat([emb_global, emb_top, emb_bottom], dim=1)
        out = self.head(combined)
        return F.normalize(out, p=2, dim=1)


def multi_similarity_loss(anchor_emb, pos_embs, neg_embs, alpha=MS_ALPHA, beta=MS_BETA, lam=MS_LAMBDA):
    B = anchor_emb.shape[0]
    losses = []
    for i in range(B):
        a = anchor_emb[i]
        pos_sim = (pos_embs[i] @ a)
        neg_sim = (neg_embs[i] @ a)

        pos_term = torch.logsumexp(-alpha * (pos_sim - lam), dim=0) / alpha
        neg_term = torch.logsumexp(beta * (neg_sim - lam), dim=0) / beta
        losses.append(pos_term + neg_term)
    return torch.stack(losses).mean()


def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    all_pos_sim, all_neg_sim = [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for anchor_img, pos_imgs, neg_imgs in loader:
            B, K_pos = pos_imgs.shape[0], pos_imgs.shape[1]
            K_neg = neg_imgs.shape[1]

            anchor_img = anchor_img.to(device)
            pos_flat = pos_imgs.view(B * K_pos, 1, IMG_SIZE, IMG_SIZE).to(device)
            neg_flat = neg_imgs.view(B * K_neg, 1, IMG_SIZE, IMG_SIZE).to(device)

            anchor_emb = model(anchor_img)
            pos_emb = model(pos_flat).view(B, K_pos, -1)
            neg_emb = model(neg_flat).view(B, K_neg, -1)

            loss = multi_similarity_loss(anchor_emb, pos_emb, neg_emb)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

            total_loss += loss.item() * B

            with torch.no_grad():
                for i in range(B):
                    a = anchor_emb[i].detach()
                    all_pos_sim.extend((pos_emb[i].detach() @ a).tolist())
                    all_neg_sim.extend((neg_emb[i].detach() @ a).tolist())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.array(all_pos_sim), np.array(all_neg_sim)


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
    train_tuplets = build_tuplets(train_data, rng, N_TUPLETS_TRAIN)
    val_tuplets = build_tuplets(val_data, rng, N_TUPLETS_VAL)
    print(f"Số tuplet: train={len(train_tuplets)}, val={len(val_tuplets)} "
          f"(mỗi tuplet = 1 anchor + {N_POSITIVE} positive + {N_NEGATIVE} negative)")

    train_loader = DataLoader(TupletDataset(train_tuplets), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(TupletDataset(val_tuplets), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = MultiScaleEmbedding().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Tổng tham số (đa tỷ lệ, train từ đầu): {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    print(f"\nBắt đầu train MS-SigNet xấp xỉ + Multi-Similarity Loss — "
          f"batch={BATCH_SIZE}, LR={LR}, tối đa {MAX_EPOCHS} epoch\n")

    t0 = time.time()
    epoch = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        ep_t0 = time.time()
        train_loss, _, _ = run_epoch(model, train_loader, optimizer)
        val_loss, val_pos_sim, val_neg_sim = run_epoch(model, val_loader, optimizer=None)
        ep_time = time.time() - ep_t0

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = " <- tốt nhất, đã lưu"
            torch.save(best_state, MODEL_DIR / "ms_signet_cotuplet_model.pt")
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

    _, val_pos_sim, val_neg_sim = run_epoch(model, val_loader, optimizer=None)
    val_scores = np.concatenate([val_pos_sim, val_neg_sim])
    val_labels = np.concatenate([np.ones_like(val_pos_sim), np.zeros_like(val_neg_sim)])
    thr_eer, eer_val = find_threshold_by_eer(val_labels, val_scores)
    thr_far, achieved_far = find_threshold_at_target_far(val_labels, val_scores, TARGET_FAR)
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
    test_scores, test_label, test_pair_type, test_writer = [], [], [], []
    with torch.no_grad():
        for fa, fb, label, ptype, wid in test_pairs_meta:
            img_a = torch.from_numpy(load_image(fa)).unsqueeze(0).unsqueeze(0)
            img_b = torch.from_numpy(load_image(fb)).unsqueeze(0).unsqueeze(0)
            e1 = model(img_a)[0]
            e2 = model(img_b)[0]
            sim = (e1 @ e2).item()
            test_scores.append(sim)
            test_label.append(label)
            test_pair_type.append(ptype)
            test_writer.append(wid)

    test_scores = np.array(test_scores)
    test_label = np.array(test_label)

    all_results = {}
    for name, thr in [("EER_balanced", thr_eer), (f"FAR_target_{TARGET_FAR:.0%}", thr_far)]:
        y_pred = (test_scores >= thr).astype(int)
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