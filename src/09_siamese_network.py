"""
Bước 9 — Siamese Network (Deep Learning)

Chạy trên CPU (không có GPU) — vì bạn còn nhiều thời gian, cấu hình ưu tiên
CHẤT LƯỢNG hơn tốc độ: ảnh 128x128, CNN 4 khối tích chập + BatchNorm,
augmentation, hard-negative mining nhẹ, train nhiều epoch với early stopping.
Chấp nhận thời gian train lâu (có thể vài giờ tùy epoch) để đổi lấy kết quả tốt hơn.

QUY TẮC (giữ nguyên theo góp ý GVHD):
- Dùng đúng writer_split.json đã chia ở bước 07 (train/val/test writer tách biệt),
  CÙNG 1 split với 08_pairwise_baseline.py để so sánh công bằng.
- Threshold chọn qua EER trên VALIDATION, không đụng test.
- Test set chỉ đánh giá đúng 1 lần, ở cuối cùng.
"""
import json
import random
import time
from pathlib import Path

# QUAN TRỌNG: import torch TRƯỚC cv2/numpy để tránh xung đột DLL (MKL/OpenMP)
# thường gặp trên Windows khi torch và opencv-python cùng nạp trong 1 process.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
import cv2

from utils_metrics import evaluate_predictions, find_threshold_by_eer, find_threshold_at_target_far

PROC_DIR = Path("data/processed")
RESULTS_DIR = Path("results/tables")
MODEL_DIR = Path("model_artifacts")

# ---- Cấu hình bản 2 — đã điều chỉnh sau khi thấy overfitting ở bản đầu ----
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 60
PAIRS_PER_EPOCH = 8000
VAL_PAIRS = 5000              # tăng từ 2000 -> 5000: val_loss/threshold ổn định hơn, đỡ nhiễu
EMBEDDING_DIM = 64            # giảm từ 128 -> 64: giảm độ phức tạp, giảm overfit
MARGIN = 1.5
LR = 1e-3
WEIGHT_DECAY = 5e-4            # tăng từ 1e-4 -> 5e-4: regularize mạnh hơn
DROPOUT = 0.5                  # tăng từ 0.4 -> 0.5
GRAD_CLIP_NORM = 5.0           # gradient clipping, ổn định quá trình train
RANDOM_SEED = 42
PATIENCE = 12                  # tăng từ 8 -> 12: kiên nhẫn hơn với val_loss nhiễu
TARGET_FAR = 0.10              # FAR mục tiêu cho threshold "ưu tiên ngân hàng"
HARD_NEGATIVE_RATIO = 0.5

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cpu")


# ---------------------------------------------------------------------------
# 1. Chuẩn bị dữ liệu ảnh theo writer
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
    """Cache lại ảnh đã đọc/resize để tránh đọc đĩa lặp lại nhiều lần (tăng tốc đáng kể
    khi cùng 1 ảnh xuất hiện trong nhiều cặp khác nhau qua các epoch)."""
    if filename in _image_cache:
        return _image_cache[filename].copy()
    path = PROC_DIR / filename
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    _image_cache[filename] = img
    return img.copy()


def augment(img):
    """Augmentation cho tập train: xoay nhẹ, dịch chuyển nhẹ, scale nhẹ, thêm nhiễu nhẹ.
    Mô phỏng biến động tự nhiên khi ký (không ai ký giống hệt 100% giữa các lần)."""
    angle = random.uniform(-10, 10)
    scale = random.uniform(0.92, 1.08)
    tx = random.uniform(-0.06, 0.06) * IMG_SIZE
    ty = random.uniform(-0.06, 0.06) * IMG_SIZE
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    img = cv2.warpAffine(img, M, (w, h), borderValue=0)

    if random.random() < 0.3:
        noise = np.random.normal(0, 0.02, img.shape).astype(np.float32)
        img = np.clip(img + noise, 0, 1)

    # Cutout: che ngẫu nhiên 1 vùng nhỏ -> buộc model không phụ thuộc vào
    # 1 chi tiết cục bộ duy nhất, giảm overfitting
    if random.random() < 0.3:
        h, w = img.shape
        cut_h, cut_w = int(h * 0.15), int(w * 0.15)
        cy = random.randint(0, h - cut_h)
        cx = random.randint(0, w - cut_w)
        img[cy:cy + cut_h, cx:cx + cut_w] = 0

    return img


# ---------------------------------------------------------------------------
# 2. Sampler cặp — resample mỗi epoch, ưu tiên negative "khó" (skilled forgery)
# ---------------------------------------------------------------------------
class PairSampler(Dataset):
    def __init__(self, writer_data, n_pairs, train=True, hard_ratio=HARD_NEGATIVE_RATIO):
        self.writer_data = writer_data
        self.writer_ids = [w for w, d in writer_data.items() if len(d["genuine"]) >= 2]
        self.writers_with_forgery = [w for w in self.writer_ids if len(writer_data[w]["skilled_forgery"]) > 0]
        self.n_pairs = n_pairs
        self.train = train
        self.hard_ratio = hard_ratio
        self.pairs = self._sample_pairs()

    def _sample_pairs(self):
        pairs = []
        n_pos = self.n_pairs // 2
        n_neg = self.n_pairs - n_pos
        n_neg_skilled = int(n_neg * self.hard_ratio)
        n_neg_random = n_neg - n_neg_skilled

        for _ in range(n_pos):
            wid = random.choice(self.writer_ids)
            a, b = random.sample(self.writer_data[wid]["genuine"], 2)
            pairs.append((a, b, 1))

        for _ in range(n_neg_skilled):
            wid = random.choice(self.writers_with_forgery)
            a = random.choice(self.writer_data[wid]["genuine"])
            b = random.choice(self.writer_data[wid]["skilled_forgery"])
            pairs.append((a, b, 0))

        for _ in range(n_neg_random):
            w1, w2 = random.sample(self.writer_ids, 2)
            a = random.choice(self.writer_data[w1]["genuine"])
            b = random.choice(self.writer_data[w2]["genuine"])
            pairs.append((a, b, 0))

        random.shuffle(pairs)
        return pairs

    def resample(self):
        self.pairs = self._sample_pairs()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        fa, fb, label = self.pairs[idx]
        img_a, img_b = load_image(fa), load_image(fb)
        if self.train:
            img_a, img_b = augment(img_a), augment(img_b)
        img_a = torch.from_numpy(img_a).unsqueeze(0)
        img_b = torch.from_numpy(img_b).unsqueeze(0)
        return img_a, img_b, torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# 3. Kiến trúc Siamese Network — CNN 4 khối + BatchNorm, theo hướng SigNet
# ---------------------------------------------------------------------------
class EmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),    # 128->64
            nn.Conv2d(32, 64, 5, padding=2), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),    # 64->32
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),  # 32->16
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2), # 16->8
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(), nn.Dropout(DROPOUT),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return F.normalize(x, p=2, dim=1)


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=EMBEDDING_DIM):
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
# 4. Train / Eval loop
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

            total_loss += loss.item() * len(label)
            all_dist.extend(dist.detach().numpy().tolist())
            all_label.extend(label.numpy().tolist())

    return total_loss / len(loader.dataset), np.array(all_dist), np.array(all_label)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(PROC_DIR / "labels.csv")
    with open(PROC_DIR / "writer_split.json", encoding="utf-8") as f:
        split = json.load(f)

    train_data = load_writer_images(split["train_writers"], labels_df)
    val_data = load_writer_images(split["val_writers"], labels_df)
    test_data = load_writer_images(split["test_writers"], labels_df)

    train_ds = PairSampler(train_data, PAIRS_PER_EPOCH, train=True)
    val_ds = PairSampler(val_data, VAL_PAIRS, train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SiameseNetwork().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    print(f"Bắt đầu huấn luyện trên CPU — tối đa {EPOCHS} epoch, {PAIRS_PER_EPOCH} cặp/epoch, ảnh {IMG_SIZE}x{IMG_SIZE}")
    print("(có thể mất vài giờ tùy cấu hình máy — cứ để chạy, model tốt nhất sẽ tự được lưu lại)\n")

    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        ep_start = time.time()
        train_ds.resample()
        train_loss, _, _ = run_epoch(model, train_loader, optimizer)
        val_loss, val_dist, val_label = run_epoch(model, val_loader, optimizer=None)
        scheduler.step(val_loss)
        ep_time = time.time() - ep_start

        marker = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            marker = "  <- tốt nhất, đã lưu"
            torch.save(best_state, MODEL_DIR / "siamese_model.pt")
        else:
            patience_counter += 1

        print(f"[Epoch {epoch:02d}/{EPOCHS}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"({ep_time:.0f}s/epoch){marker}")

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping tại epoch {epoch} (val_loss không cải thiện thêm {PATIENCE} epoch liên tiếp)")
            break

    total_time = time.time() - t0
    print(f"\nTổng thời gian train: {total_time/60:.1f} phút")

    model.load_state_dict(best_state)

    # ---- Chọn threshold trên VALIDATION (không đụng test) — CẢ 2 CÁCH ----
    _, val_dist_final, val_label_final = run_epoch(model, val_loader, optimizer=None)
    val_scores = -val_dist_final

    # Cách 1: EER — điểm cân bằng FAR = FRR
    threshold_score_eer, eer_val = find_threshold_by_eer(val_label_final, val_scores)
    threshold_dist_eer = -threshold_score_eer
    print(f"\n>>> [Threshold A - EER cân bằng] khoảng cách={threshold_dist_eer:.4f}  (EER trên val={eer_val:.4f})")

    # Cách 2: Target FAR — ưu tiên đúng nhu cầu ngân hàng (chấp nhận FRR cao hơn)
    threshold_score_far, achieved_far_val = find_threshold_at_target_far(val_label_final, val_scores, TARGET_FAR)
    threshold_dist_far = -threshold_score_far
    print(f">>> [Threshold B - Target FAR={TARGET_FAR:.0%}] khoảng cách={threshold_dist_far:.4f}  "
          f"(FAR đạt được trên val={achieved_far_val:.4f})")

    # ==== TEST SET — CHỈ ĐÁNH GIÁ 1 LẦN DUY NHẤT (cho cả 2 threshold) ====
    test_pairs = []
    rng_test = random.Random(RANDOM_SEED)
    for wid, d in test_data.items():
        genuine, forged = d["genuine"], d["skilled_forgery"]
        if len(genuine) < 2:
            continue
        ref = rng_test.choice(genuine)
        for q in genuine:
            if q != ref:
                test_pairs.append((ref, q, 1, "positive", wid))
        for q in forged:
            test_pairs.append((ref, q, 0, "negative_skilled", wid))

    model.eval()
    test_dist, test_label, test_pair_type, test_writer = [], [], [], []
    with torch.no_grad():
        for fa, fb, label, ptype, wid in test_pairs:
            img_a = torch.from_numpy(load_image(fa)).unsqueeze(0).unsqueeze(0)
            img_b = torch.from_numpy(load_image(fb)).unsqueeze(0).unsqueeze(0)
            e1, e2 = model(img_a, img_b)
            d = F.pairwise_distance(e1, e2).item()
            test_dist.append(d)
            test_label.append(label)
            test_pair_type.append(ptype)
            test_writer.append(wid)

    test_dist = np.array(test_dist)
    test_label = np.array(test_label)
    test_scores = -test_dist

    all_results = {}
    for name, thr in [("EER_balanced", threshold_dist_eer), (f"FAR_target_{TARGET_FAR:.0%}", threshold_dist_far)]:
        y_pred_test = (test_dist <= thr).astype(int)
        result, per_writer_df = evaluate_predictions(
            test_label, y_pred_test, test_scores,
            pair_type=np.array(test_pair_type), writer_id=np.array(test_writer),
        )
        all_results[name] = result

        print(f"\n=== KẾT QUẢ TEST SET — Threshold [{name}] ===")
        for k, v in result.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        pd.DataFrame([result]).to_csv(RESULTS_DIR / f"siamese_test_results_{name}.csv", index=False)
        per_writer_df.to_csv(RESULTS_DIR / f"siamese_per_writer_{name}.csv", index=False)

    with open(RESULTS_DIR / "siamese_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "img_size": IMG_SIZE, "embedding_dim": EMBEDDING_DIM, "epochs_run": epoch,
            "pairs_per_epoch": PAIRS_PER_EPOCH, "val_pairs": VAL_PAIRS,
            "threshold_EER_balanced": float(threshold_dist_eer), "val_EER": float(eer_val),
            "threshold_FAR_target": float(threshold_dist_far), "target_FAR": TARGET_FAR,
            "achieved_FAR_on_val": float(achieved_far_val),
            "total_train_time_minutes": total_time / 60,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nĐã lưu kết quả (2 threshold) vào {RESULTS_DIR}/, model vào {MODEL_DIR}/siamese_model.pt")
    print("\n>>> So sánh nhanh 2 phương án threshold:")
    for name, r in all_results.items():
        print(f"  [{name}] Accuracy={r['accuracy']:.4f}  FAR={r['FAR']:.4f}  FRR={r['FRR']:.4f}")


if __name__ == "__main__":
    main()