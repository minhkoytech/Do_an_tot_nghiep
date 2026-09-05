"""
Bước 10 — Trích đặc trưng bằng VGG19 pretrained (Transfer Learning thuần,
KHÔNG fine-tune) — theo đúng hướng "Simple Baseline Pretrained CNN" trong
y văn (đạt AUC ~84.82% trên CEDAR).

Khác với 09_siamese_network.py (fine-tune CNN, train hàng chục giờ), cách
này CHỈ dùng VGG19 để trích đặc trưng (forward pass 1 lần, không backprop)
— nhanh hơn rất nhiều, và tránh rủi ro overfitting khi fine-tune trên dữ
liệu nhỏ (38 writer).

Đầu ra: mỗi ảnh -> 1 vector 512 chiều (global average pooling của VGG19),
lưu vào data/processed/features_vgg19.csv
"""
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T

import numpy as np
import pandas as pd
import cv2

PROC_DIR = Path("data/processed")
IMG_SIZE = 224  # kích thước chuẩn VGG19 được train
device = torch.device("cpu")


def load_vgg19_feature_extractor():
    vgg = tv_models.vgg19(weights=tv_models.VGG19_Weights.IMAGENET1K_V1)
    vgg.eval()
    for p in vgg.parameters():
        p.requires_grad = False  # ĐÓNG BĂNG HOÀN TOÀN — không fine-tune

    features = vgg.features  # phần conv, bỏ classifier
    pool = nn.AdaptiveAvgPool2d((1, 1))  # global average pooling -> 512-dim

    class Extractor(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = features
            self.pool = pool

        def forward(self, x):
            x = self.features(x)
            x = self.pool(x)
            return torch.flatten(x, 1)

    return Extractor().to(device).eval()


def load_and_prepare_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    img3 = np.stack([img, img, img], axis=0).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img3 = (img3 - mean) / std
    return torch.from_numpy(img3.astype(np.float32))


def main():
    labels = pd.read_csv(PROC_DIR / "labels.csv")
    extractor = load_vgg19_feature_extractor()

    print(f"Trích đặc trưng VGG19 cho {len(labels)} ảnh (đóng băng hoàn toàn, chỉ forward pass)...")
    t0 = time.time()

    rows = []
    with torch.no_grad():
        for i, row in labels.iterrows():
            img_tensor = load_and_prepare_image(PROC_DIR / row["filename"]).unsqueeze(0)
            emb = extractor(img_tensor).squeeze(0).numpy()
            rows.append(emb)
            if (i + 1) % 200 == 0:
                elapsed = time.time() - t0
                print(f"  Đã xử lý {i+1}/{len(labels)} ảnh ({elapsed:.0f}s)")

    emb_matrix = np.stack(rows)
    emb_cols = [f"vgg_{i}" for i in range(emb_matrix.shape[1])]
    emb_df = pd.DataFrame(emb_matrix, columns=emb_cols)
    result = pd.concat([labels.reset_index(drop=True), emb_df], axis=1)

    result.to_csv(PROC_DIR / "features_vgg19.csv", index=False)
    total_time = time.time() - t0
    print(f"\nHoàn thành trong {total_time/60:.1f} phút.")
    print(f"Đã lưu {emb_matrix.shape[1]} chiều đặc trưng/ảnh vào {PROC_DIR / 'features_vgg19.csv'}")


if __name__ == "__main__":
    main()