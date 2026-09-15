"""
09_error_analysis_gradcam.py
------------------------------
Phan tich loi bang Grad-CAM (Buoc 7 trong de cuong):
    "Su dung ky thuat Grad-CAM de truc quan hoa vung anh ma mo hinh CNN
    tap trung khi dua ra quyet dinh, tu do phan tich nhung truong hop
    du doan sai va rut ra nguyen nhan do chu ky qua ngan, qua phuc tap,
    hoac gia mao co do tuong dong cao,..."

CACH GRAD-CAM HOAT DONG VOI SIAMESE NETWORK (khac phan loai thong thuong):
    Grad-CAM chuan (cho bai toan phan loai) backprop tu diem so cua 1
    LOP CU THE. Siamese Network khong co "lop" - no tao ra KHOANG CACH
    giua 2 embedding. Vi vay o day dung chinh KHOANG CACH D = ||emb_A -
    emb_B|| lam scalar de backprop, thuc hien theo TUNG ANH mot:

    - CAM cua anh A: giu embedding cua anh B CO DINH (khong lan truyen
      nguoc qua nhanh B), backprop D qua nhanh A -> CAM_A cho biet
      "vung nao cua anh A khien no trong KHAC anh B hon" (dong gop lam
      TANG khoang cach).
    - Lam tuong tu de co CAM_B (doi vai tro A/B).

    Dieu nay giup tra loi dung cau hoi cua de cuong: voi 1 cap du doan
    SAI, vung nao cua tung anh la nguyen nhan khien model dua ra ket
    luan sai.

CAC TRUONG HOP LOI DUOC PHAN TICH:
    - False Accept (FA): cap gia (skilled/random forgery) nhung model
      du doan "khop" - nguy hiem nhat trong ngan hang.
    - False Reject (FR): cap that (genuine_genuine) nhung model du doan
      "khong khop".

YEU CAU: phai chay xong 06_siamese_network.py truoc (can model_artifacts/
siamese_model_*.pt va siamese_threshold.json).

Cach dung (khong can tham so, path da khop san):
    cd src
    python 09_error_analysis_gradcam.py

Output:
    results/figures/gradcam_false_accept_*.png  (vi du FA, kem heatmap)
    results/figures/gradcam_false_reject_*.png  (vi du FR, kem heatmap)
    results/tables/error_analysis_summary.csv   (danh sach loi da phan tich)
"""

import argparse
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent
DEFAULT_PAIRS_DIR = PROJECT_ROOT / "data" / "processed" / "pairs"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "model_artifacts"
DEFAULT_TABLES_DIR = PROJECT_ROOT / "results" / "tables"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

N_EXAMPLES_PER_CATEGORY = 4  # so vi du loi moi loai duoc truc quan hoa


def load_module(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, SRC_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


siamese_mod = load_module("06_siamese_network.py", "siamese_mod_gradcam")


def load_image_tensor(path: str, device) -> torch.Tensor:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)


def load_single_siamese_model(model_dir: Path, device):
    """
    Dung 1 MODEL DUY NHAT (model dau tien trong ensemble) cho Grad-CAM,
    khong dung ca ensemble - vi Grad-CAM can 1 do thi tinh toan (computation
    graph) ro rang de lan truyen nguoc, ket hop nhieu model se lam heatmap
    kho dien giai hon. Model dau tien la lua chon dai dien hop ly.
    """
    with open(model_dir / "siamese_threshold.json") as f:
        config = json.load(f)
    embedding_dim = config.get("embedding_dim", 64)
    threshold = config["threshold"]

    model_files = sorted(model_dir.glob("siamese_model_*.pt"))
    if not model_files:
        model_files = [model_dir / "siamese_best.pt"]
    model = siamese_mod.SiameseNetwork(embedding_dim=embedding_dim, backbone_type="custom").to(device)
    model.load_state_dict(torch.load(model_files[0], map_location=device, weights_only=True))
    model.eval()
    return model, threshold, model_files[0].name


def compute_pair_gradcam(model, target_layer, img_a, img_b):
    """
    Tra ve (cam_a, cam_b, distance): Grad-CAM cho tung anh trong cap,
    dua tren khoang cach D = ||emb_a - emb_b||. Xem giai thich chi tiet
    o dau file.
    """
    def run_single(img_forward, emb_fixed_detached):
        activations, gradients = [], []

        def fwd_hook(module, inp, out):
            activations.append(out)

        def bwd_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        h1 = target_layer.register_forward_hook(fwd_hook)
        h2 = target_layer.register_full_backward_hook(bwd_hook)

        model.zero_grad()
        emb_forward = model.embedding_net(img_forward)
        distance = F.pairwise_distance(emb_forward, emb_fixed_detached)
        distance.backward()

        h1.remove()
        h2.remove()

        act = activations[0].squeeze(0)   # (C, H, W)
        grad = gradients[0].squeeze(0)     # (C, H, W)
        weights = grad.mean(dim=(1, 2))    # (C,)
        cam = torch.zeros(act.shape[1:], dtype=torch.float32, device=act.device)
        for c in range(act.shape[0]):
            cam += weights[c] * act[c]
        cam = torch.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy(), distance.item()

    with torch.no_grad():
        emb_b_fixed = model.embedding_net(img_b)
    cam_a, dist_val = run_single(img_a, emb_b_fixed)

    with torch.no_grad():
        emb_a_fixed = model.embedding_net(img_a)
    cam_b, _ = run_single(img_b, emb_a_fixed)

    return cam_a, cam_b, dist_val


def overlay_heatmap(image_path: str, cam: np.ndarray):
    """Resize CAM ve kich thuoc anh goc, phu mau (colormap JET) len anh grayscale goc."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    cam_resized = cv2.resize(cam, (w, h))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(img_color, 0.6, heatmap, 0.4, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def plot_error_case(ref_path, query_path, cam_ref, cam_query, distance, threshold,
                     pair_type, true_label, out_path):
    ref_overlay = overlay_heatmap(ref_path, cam_ref)
    query_overlay = overlay_heatmap(query_path, cam_query)
    ref_orig = cv2.cvtColor(cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2RGB)
    query_orig = cv2.cvtColor(cv2.imread(query_path, cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2RGB)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].imshow(ref_orig); axes[0, 0].set_title("Reference (goc)"); axes[0, 0].axis("off")
    axes[0, 1].imshow(query_orig); axes[0, 1].set_title("Query (goc)"); axes[0, 1].axis("off")
    axes[1, 0].imshow(ref_overlay); axes[1, 0].set_title("Reference (Grad-CAM)"); axes[1, 0].axis("off")
    axes[1, 1].imshow(query_overlay); axes[1, 1].set_title("Query (Grad-CAM)"); axes[1, 1].axis("off")

    predicted = "Khop" if distance < 0 else "Khong khop"  # placeholder, se ghi de o duoi
    true_str = "That (genuine_genuine)" if true_label == 1 else f"Gia ({pair_type})"
    fig.suptitle(
        f"Nhan that: {true_str} | Distance={distance:.4f} (threshold~{-threshold:.4f} tren thang -distance)\n"
        f"Vung do/vang cang dam = anh huong cang lon den quyet dinh cua model",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Phan tich loi bang Grad-CAM cho Siamese Network")
    parser.add_argument("--pairs_dir", default=str(DEFAULT_PAIRS_DIR))
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--tables_dir", default=str(DEFAULT_TABLES_DIR))
    parser.add_argument("--figures_dir", default=str(DEFAULT_FIGURES_DIR))
    parser.add_argument("--n_examples", type=int, default=N_EXAMPLES_PER_CATEGORY)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dang su dung device: {device}")

    model_dir = Path(args.model_dir)
    model, threshold, model_filename = load_single_siamese_model(model_dir, device)
    target_layer = model.embedding_net.conv[-2]  # lop conv cuoi cung (sau ReLU), truoc AdaptiveAvgPool
    print(f"Dung model: {model_filename} | threshold={threshold:.4f}")

    test_pairs = pd.read_csv(Path(args.pairs_dir) / "pairs_test.csv")

    # Tinh du doan cho toan bo test set truoc, de xac dinh cac truong hop sai
    print("Dang tinh du doan cho toan bo test set...")
    predictions = []
    with torch.no_grad():
        for row in test_pairs.itertuples():
            img_a = load_image_tensor(row.reference_path, device)
            img_b = load_image_tensor(row.query_path, device)
            distance = model(img_a, img_b).item()
            score = -distance
            predictions.append(score)
    test_pairs = test_pairs.copy()
    test_pairs["score"] = predictions
    test_pairs["y_pred"] = (test_pairs["score"] >= threshold).astype(int)
    test_pairs["correct"] = test_pairs["y_pred"] == test_pairs["label"]

    false_accepts = test_pairs[(test_pairs["label"] == 0) & (test_pairs["y_pred"] == 1)]
    false_rejects = test_pairs[(test_pairs["label"] == 1) & (test_pairs["y_pred"] == 0)]
    print(f"\nSo luong False Accept (gia mao lot qua): {len(false_accepts)}")
    print(f"So luong False Reject (chu ky that bi tu choi nham): {len(false_rejects)}")

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    error_summary = []

    # Uu tien cac truong hop "sai nhat" (score cach xa threshold nhat theo huong sai)
    fa_examples = false_accepts.sort_values("score", ascending=False).head(args.n_examples)
    for i, row in enumerate(fa_examples.itertuples()):
        img_a = load_image_tensor(row.reference_path, device)
        img_b = load_image_tensor(row.query_path, device)
        cam_a, cam_b, distance = compute_pair_gradcam(model, target_layer, img_a, img_b)
        out_path = figures_dir / f"gradcam_false_accept_{i+1}.png"
        plot_error_case(row.reference_path, row.query_path, cam_a, cam_b, distance, threshold,
                         row.pair_type, row.label, out_path)
        error_summary.append({
            "error_type": "false_accept", "pair_type": row.pair_type,
            "writer_id_ref": row.writer_id_ref, "score": row.score,
            "distance": distance, "figure": str(out_path),
        })
        print(f"Da luu: {out_path}")

    fr_examples = false_rejects.sort_values("score", ascending=True).head(args.n_examples)
    for i, row in enumerate(fr_examples.itertuples()):
        img_a = load_image_tensor(row.reference_path, device)
        img_b = load_image_tensor(row.query_path, device)
        cam_a, cam_b, distance = compute_pair_gradcam(model, target_layer, img_a, img_b)
        out_path = figures_dir / f"gradcam_false_reject_{i+1}.png"
        plot_error_case(row.reference_path, row.query_path, cam_a, cam_b, distance, threshold,
                         row.pair_type, row.label, out_path)
        error_summary.append({
            "error_type": "false_reject", "pair_type": row.pair_type,
            "writer_id_ref": row.writer_id_ref, "score": row.score,
            "distance": distance, "figure": str(out_path),
        })
        print(f"Da luu: {out_path}")

    summary_df = pd.DataFrame(error_summary)
    summary_path = tables_dir / "error_analysis_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nDa luu bang tong hop: {summary_path}")

    # Thong ke nhanh: writer nao xuat hien nhieu trong loi (goi y cho phan
    # "phan tich nguyen nhan" trong bao cao)
    if len(error_summary) > 0:
        print("\n=== Writer xuat hien nhieu nhat trong cac truong hop loi (ca 2 loai) ===")
        print(summary_df["writer_id_ref"].value_counts().head(5))

    print("\nHoan tat phan tich loi bang Grad-CAM.")
    print("Mo cac file anh trong results/figures/gradcam_*.png de xem heatmap chi tiet.")


if __name__ == "__main__":
    main()