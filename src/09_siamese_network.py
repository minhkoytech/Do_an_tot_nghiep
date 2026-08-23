"""
Bước 9 — Siamese Network, viết lại theo tham khảo notebook Kaggle
"Contrastive Loss Signature Verification" (Ishani Kathuria)

BÁM THEO file tham khảo ở các điểm sau:
- Ảnh 224x224, grayscale (1 kênh) — không dùng transfer learning, train từ đầu
- Batch size 128
- Adam, learning rate CỰC THẤP (1e-5) — chiến lược "học chậm mà chắc"
- Train rất lâu (patience cao), dùng 1 bộ cặp CỐ ĐỊNH cho train (không resample
  mỗi epoch, giống cách file gốc dùng dataset.pairs cố định)

KHÔNG bám theo 1 điểm — GIỮ NGUYÊN theo góp ý GVHD:
- File gốc dùng validation_split=0.2 (chia NGẪU NHIÊN theo cặp, không theo
  writer) -> đây là data leakage ở cấp writer. Bản này vẫn dùng đúng
  writer_splits.json (train/val/test writer tách biệt hoàn toàn), vì đây là
  yêu cầu bắt buộc của GVHD, không thể đánh đổi để lấy con số đẹp hơn.

Vì kiến trúc CNN thật sự của file gốc nằm trong contrastive_utils.py (không
được đính kèm), phần embedding_net dưới đây là kiến trúc CNN tiêu chuẩn được
dựng lại theo đúng tinh thần (nhiều lớp Conv2D + MaxPooling, ảnh vào 224x224x1),
không phải sao chép nguyên văn.
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
RESULTS_DIR = Path("results/tables/siamese_v3")
MODEL_DIR = Path("model_artifacts")

# ---- Theo đúng file tham khảo ----
IMG_SIZE = 224
BATCH_SIZE = 128
LR = 1e-5                    # cực thấp, giống file gốc
MARGIN = 1.0
MAX_EPOCHS = 300             # cho phép train rất lâu, giống file gốc (họ chạy tới ~416 epoch)
PATIENCE = 25                # kiên nhẫn cao — LR thấp thì loss cải thiện rất chậm mỗi epoch
WEIGHT_DECAY = 0.0           # file gốc không dùng weight decay
RANDOM_SEED = 42
TARGET_FAR = 0.10
MAX_TRAIN_PAIRS = 12000      # giới hạn để thời gian/epoch còn khả thi trên CPU

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cpu")


# ---------------------------------------------------------------------------
# 1. Dữ liệu — tạo 1 BỘ CẶP CỐ ĐỊNH (không resample mỗi epoch, giống file gốc)
# ---------------------------------------------------------------------------
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
    img = img.astype(np.float32) / 255.0
    _image_cache[filename] = img
    return img.copy()


def build_fixed_pairs(writer_data, rng, max_pairs=None):
    """Tạo 1 bộ cặp CỐ ĐỊNH (không đổi qua các epoch) — giống cách file gốc
    dùng dataset.pairs cố định thay vì lấy mẫu ngẫu nhiên mỗi epoch."""
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


# ---------------------------------------------------------------------------
# 2. Kiến trúc CNN (dựng lại theo tinh thần, vì file gốc không kèm contrastive_utils.py)
# ---------------------------------------------------------------------------
class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),     # 224->112
            nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),    # 112->56
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 56->28
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 28->14
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)  # không ép normalize về hình cầu đơn vị, giữ đơn giản như file gốc


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.embedding_net = EmbeddingNet(embedding_dim)

    def forward(self, x1, x2):
        return self.embedding_net(x1), self.embedding_net(x2)


def contrastive_loss(e1, e2, label, margin=MARGIN):
    dist = F.pairwise_distance(e1, e2)
    loss_pos = label * dist.pow(2)
    loss_neg = (1 - label) * F.relu(margin - dist).pow(2)
    return (loss_pos + loss_neg).mean(), dist


# ---------------------------------------------------------------------------
# 3. Train / eval
# ---------------------------------------------------------------------------
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
        splits_data = json.load(f)
    split = splits_data["splits"][0]   # dùng split đầu tiên (đúng writer-independent, đã có sẵn)

    train_data = load_writer_images(split["train_writers"], labels_df)
    val_data = load_writer_images(split["val_writers"], labels_df)
    test_data = load_writer_images(split["test_writers"], labels_df)

    rng = random.Random(RANDOM_SEED)
    train_pairs = build_fixed_pairs(train_data, rng, max_pairs=MAX_TRAIN_PAIRS)
    val_pairs = build_fixed_pairs(val_data, rng, max_pairs=2000)
    print(f"Số cặp CỐ ĐỊNH: train={len(train_pairs)}, val={len(val_pairs)}")

    train_loader = DataLoader(FixedPairDataset(train_pairs), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(FixedPairDataset(val_pairs), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SiameseNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    print(f"\nBắt đầu train — ảnh {IMG_SIZE}x{IMG_SIZE}, batch={BATCH_SIZE}, LR={LR}, tối đa {MAX_EPOCHS} epoch")
    print("(LR rất thấp -> mỗi epoch cải thiện chậm, cần train lâu — cứ để chạy)\n")

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
            torch.save(best_state, MODEL_DIR / "siamese_v3_model.pt")
        else:
            patience_counter += 1

        if epoch <= 10 or epoch % 5 == 0 or marker:
            print(f"[Epoch {epoch:03d}/{MAX_EPOCHS}] train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  ({ep_time:.0f}s/epoch){marker}")

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping tại epoch {epoch} (val_loss không cải thiện thêm {PATIENCE} epoch)")
            break

    total_time = time.time() - t0
    print(f"\nTổng thời gian train: {total_time/60:.1f} phút")

    model.load_state_dict(best_state)

    _, val_dist_final, val_label_final = run_epoch(model, val_loader, optimizer=None)
    val_scores = -val_dist_final
    thr_eer_score, eer_val = find_threshold_by_eer(val_label_final, val_scores)
    thr_far_score, achieved_far = find_threshold_at_target_far(val_label_final, val_scores, TARGET_FAR)
    print(f"\n>>> [EER cân bằng] khoảng cách={-thr_eer_score:.4f}  (EER trên val={eer_val:.4f})")
    print(f">>> [FAR mục tiêu {TARGET_FAR:.0%}] khoảng cách={-thr_far_score:.4f}  (FAR đạt được={achieved_far:.4f})")

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

    with open(RESULTS_DIR / "config.json", "w", encoding="utf-8") as f:
        json.dump({
            "img_size": IMG_SIZE, "batch_size": BATCH_SIZE, "lr": LR,
            "epochs_run": epoch, "n_train_pairs": len(train_pairs),
            "val_EER": float(eer_val), "achieved_FAR_on_val": float(achieved_far),
            "total_train_time_minutes": total_time / 60,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nĐã lưu kết quả vào {RESULTS_DIR}/")
    print("\n>>> So sánh nhanh 2 threshold:")
    for name, r in all_results.items():
        print(f"  [{name}] Accuracy={r['accuracy']:.4f}  FAR={r['FAR']:.4f}  FRR={r['FRR']:.4f}  ROC_AUC={r['ROC_AUC']:.4f}")


if __name__ == "__main__":
    main()