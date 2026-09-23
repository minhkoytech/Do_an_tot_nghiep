"""
app.py
------
Web demo (Streamlit) cho he thong xac thuc chu ky - Muc tieu cu the #5
trong de cuong: "Phat trien mot ung dung demo cho phep nguoi dung tai
len hai anh chu ky va he thong se tu dong phan tich, so sanh va dua ra
ket qua ve muc do giong nhau giua hai chu ky."

QUAN TRONG: app nay KHONG viet lai logic tien xu ly / trich dac trung /
kien truc model - no IMPORT TRUC TIEP tu 01_preprocessing.py,
03_features.py, 06_siamese_network.py, va 08_combined_model.py (cung
thu muc src/) de dam bao demo dung Y HET pipeline da dung khi train va
danh gia, tranh sai lech giua "code demo" va "code that".

4 MODEL duoc hien thi: RF, SVM (baseline), Siamese (de xuat chinh),
Combined (dac trung thu cong + embedding CNN - theo ket qua
10_threshold_analysis.py, day la model dang de xuat vi thang o moi
muc FAR muc tieu).

Cach chay (tu thu muc src/):
    pip install streamlit
    streamlit run app.py

Yeu cau: da chay xong 01_preprocessing.py, 01b_generate_pairs.py,
03_features.py, 05_pairwise_baseline.py, 06_siamese_network.py,
08_combined_model.py truoc do (can co san model_artifacts/*.joblib,
*.pt, *.json).
"""

import importlib.util
import json
import tempfile
from pathlib import Path

import cv2
import joblib
import numpy as np
import streamlit as st
import torch

# ---------------------------------------------------------------------------
# Import truc tiep cac module co san (khong duplicate code)
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
MODEL_DIR = PROJECT_ROOT / "model_artifacts"


def load_module(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, SRC_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preprocess_mod = load_module("01_preprocessing.py", "preprocess_mod")
features_mod = load_module("03_features.py", "features_mod")
siamese_mod = load_module("06_siamese_network.py", "siamese_mod")
combined_mod = load_module("08_combined_model.py", "combined_mod")
gradcam_mod = None
try:
    gradcam_mod = load_module("09_error_analysis_gradcam.py", "gradcam_mod")
except Exception:
    pass  # Grad-CAM la tinh nang tuy chon, khong lam sap app neu file 09 chua co

SELECTED_FEATURES = siamese_mod.SELECTED_FEATURES

# ---------------------------------------------------------------------------
# MODEL RA QUYET DINH - chi MOT model duy nhat duoc dung de ket luan
# Khop/Khong khop. Doi gia tri o day de chuyen sang model khac:
#   "combined" - dac trung thu cong + embedding CNN (tot nhat theo
#                10_threshold_analysis.py; can da chay 08_combined_model.py)
#   "siamese"  - Siamese Network (ensemble)
#   "rf"       - Random Forest pairwise
#   "svm"      - SVM pairwise
# Neu chon "combined" nhung chua co model Combined, app tu dong dung "siamese".
# ---------------------------------------------------------------------------
DECISION_MODEL = "combined"

MODEL_DISPLAY_NAMES = {
    "combined": "Combined (đặc trưng thủ công + học sâu)",
    "siamese": "Siamese Network",
    "rf": "Random Forest",
    "svm": "SVM",
}


# ---------------------------------------------------------------------------
# Load model artifacts (cache lai, chi load 1 lan du user thao tac nhieu lan)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_all_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Pairwise RF/SVM baseline ---
    rf = joblib.load(MODEL_DIR / "pairwise_rf.joblib")
    svm = joblib.load(MODEL_DIR / "pairwise_svm.joblib")
    scaler = joblib.load(MODEL_DIR / "pairwise_scaler.joblib")
    imputer = joblib.load(MODEL_DIR / "pairwise_imputer.joblib")
    with open(MODEL_DIR / "pairwise_threshold.json") as f:
        pairwise_thresholds = json.load(f)

    # --- Siamese Network (ensemble) ---
    with open(MODEL_DIR / "siamese_threshold.json") as f:
        siamese_config = json.load(f)
    embedding_dim = siamese_config.get("embedding_dim", 64)

    siamese_models = []
    model_files = sorted(MODEL_DIR.glob("siamese_model_*.pt"))
    if not model_files:
        legacy_path = MODEL_DIR / "siamese_best.pt"
        if legacy_path.exists():
            model_files = [legacy_path]
    for model_path in model_files:
        model = siamese_mod.SiameseNetwork(embedding_dim=embedding_dim, backbone_type="custom").to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        siamese_models.append(model)

    result = {
        "device": device,
        "rf": rf, "svm": svm, "scaler": scaler, "imputer": imputer,
        "pairwise_thresholds": pairwise_thresholds,
        "siamese_models": siamese_models,
        "siamese_threshold": siamese_config["threshold"],
        "embedding_dim": embedding_dim,
        "has_combined": False,
    }

    # --- Combined model (dac trung thu cong + embedding CNN) - optional ---
    combined_path = MODEL_DIR / "combined_rf.joblib"
    if combined_path.exists():
        result["combined_rf"] = joblib.load(combined_path)
        result["combined_scaler"] = joblib.load(MODEL_DIR / "combined_scaler.joblib")
        result["combined_imputer"] = joblib.load(MODEL_DIR / "combined_imputer.joblib")
        with open(MODEL_DIR / "combined_threshold.json") as f:
            result["combined_threshold"] = json.load(f)["threshold"]
        result["has_combined"] = True

    return result


# ---------------------------------------------------------------------------
# Xu ly 1 anh upload: luu tam -> tien xu ly (01_preprocessing) -> tra ve
# duong dan anh da tien xu ly
# ---------------------------------------------------------------------------
def preprocess_uploaded_image(uploaded_file, tmp_dir: Path) -> str:
    raw_path = tmp_dir / f"raw_{uploaded_file.name}"
    with open(raw_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    processed = preprocess_mod.preprocess_image(str(raw_path))
    processed_path = tmp_dir / f"processed_{uploaded_file.name}.png"
    cv2.imwrite(str(processed_path), processed)
    return str(processed_path)


def load_image_tensor(path: str) -> torch.Tensor:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)


@torch.no_grad()
def compute_siamese_verdict(models_dict, ref_path: str, query_path: str):
    device = models_dict["device"]
    img1 = load_image_tensor(ref_path).to(device)
    img2 = load_image_tensor(query_path).to(device)

    scores = []
    for model in models_dict["siamese_models"]:
        distance = model(img1, img2).item()
        scores.append(-distance)
    avg_score = float(np.mean(scores))
    threshold = models_dict["siamese_threshold"]
    return avg_score, threshold, avg_score >= threshold


def compute_classical_verdict(models_dict, ref_path: str, query_path: str, model_name: str):
    ref_features = features_mod.extract_all_features(ref_path)
    query_features = features_mod.extract_all_features(query_path)
    delta_vector = np.array([[abs(ref_features[f] - query_features[f]) for f in SELECTED_FEATURES]])

    delta_imputed = models_dict["imputer"].transform(delta_vector)
    delta_scaled = models_dict["scaler"].transform(delta_imputed)

    model = models_dict[model_name]
    threshold = models_dict["pairwise_thresholds"][f"{model_name}_threshold"]
    if model_name == "svm":
        score = float(model.decision_function(delta_scaled)[0])
    else:
        score = float(model.predict_proba(delta_scaled)[0, 1])
    return score, threshold, score >= threshold


@torch.no_grad()
def compute_combined_verdict(models_dict, ref_path: str, query_path: str):
    """Dac trung thu cong (delta) + embedding CNN (delta, trung binh ca ensemble Siamese) -> RF."""
    ref_features = features_mod.extract_all_features(ref_path)
    query_features = features_mod.extract_all_features(query_path)
    classical_delta = [abs(ref_features[f] - query_features[f]) for f in SELECTED_FEATURES]

    device = models_dict["device"]
    ref_emb = combined_mod.compute_embedding_for_image(models_dict["siamese_models"], ref_path, device)
    query_emb = combined_mod.compute_embedding_for_image(models_dict["siamese_models"], query_path, device)
    embedding_delta = np.abs(ref_emb - query_emb)

    full_vector = np.array([classical_delta + list(embedding_delta)])
    imputed = models_dict["combined_imputer"].transform(full_vector)
    scaled = models_dict["combined_scaler"].transform(imputed)

    score = float(models_dict["combined_rf"].predict_proba(scaled)[0, 1])
    threshold = models_dict["combined_threshold"]
    return score, threshold, score >= threshold


# ---------------------------------------------------------------------------
# Giao dien Streamlit
# ---------------------------------------------------------------------------
def compute_gradcam_for_pair(models_dict, ref_path: str, query_path: str):
    """
    Tinh Grad-CAM TRUC TIEP cho cap anh vua upload, dung model Siamese
    dau tien trong ensemble (nhat quan voi 09_error_analysis_gradcam.py -
    Grad-CAM can 1 do thi tinh toan ro rang, dung ca ensemble se kho
    dien giai hon). Tra ve 2 anh overlay (numpy RGB) cho reference/query.
    """
    if gradcam_mod is None:
        return None, None
    device = models_dict["device"]
    model = models_dict["siamese_models"][0]
    target_layer = model.embedding_net.conv[-2]

    img_a = gradcam_mod.load_image_tensor(ref_path, device)
    img_b = gradcam_mod.load_image_tensor(query_path, device)
    cam_a, cam_b, _ = gradcam_mod.compute_pair_gradcam(model, target_layer, img_a, img_b)

    overlay_a = gradcam_mod.overlay_heatmap(ref_path, cam_a)
    overlay_b = gradcam_mod.overlay_heatmap(query_path, cam_b)
    return overlay_a, overlay_b


def render_gauge_svg(value01: float, color: str, size: int = 84) -> str:
    """SVG gauge tron - tuong tu ban frontend HTML, dung lai trong Streamlit qua unsafe_allow_html."""
    r = 34
    circ = 2 * 3.14159265 * r
    offset = circ * (1 - value01)
    cx = cy = size / 2
    return f'''
    <div style="position:relative; width:{size}px; height:{size}px; margin:0 auto;">
      <svg width="{size}" height="{size}" style="transform:rotate(-90deg);">
        <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="6"/>
        <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="6"
          stroke-linecap="round" stroke-dasharray="{circ:.1f}" stroke-dashoffset="{offset:.1f}"/>
      </svg>
      <div style="position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
        font-family:'Bebas Neue',sans-serif; font-size:1.15rem; color:{color};">{round(value01*100)}</div>
    </div>'''


def normalize_score_for_display(score: float, model_key: str) -> float:
    """Chuan hoa score ve [0,1] CHI DE HIEN THI gauge - khong doi logic quyet dinh that (van dung score/threshold goc)."""
    import math
    if model_key == "siamese":
        return max(0.0, min(1.0, 1 / (1 + math.exp(-score / 3))))
    if model_key == "svm":
        return max(0.0, min(1.0, 1 / (1 + math.exp(-score))))
    return max(0.0, min(1.0, score))


@st.cache_data
def load_performance_metrics():
    """Doc bang so sanh 4 model tu ket qua da co san (neu co) de hien thi trong phan 'Ve he thong'."""
    candidates = [
        PROJECT_ROOT / "results" / "tables" / "final_model_comparison_with_combined.csv",
        PROJECT_ROOT / "results" / "tables" / "final_model_comparison.csv",
    ]
    for path in candidates:
        if path.exists():
            import pandas as pd
            return pd.read_csv(path)
    return None


def main():
    st.set_page_config(page_title="SignatureVerify", page_icon="✍️", layout="wide")

    # -------------------------------------------------------------------
    # CSS tuy chinh - phong cach Netflix: nen den, do nhan (#E50914),
    # typography dam, card bo tron, hover effect. AN TOAN BO khung mac
    # dinh cua Streamlit (menu, footer "Made with Streamlit", header
    # toolbar) de tranh cam giac "app demo" - thay bang navbar rieng.
    # -------------------------------------------------------------------
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        /* An toan bo khung mac dinh cua Streamlit */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header[data-testid="stHeader"] { display: none; }
        div[data-testid="stToolbar"] { display: none; }
        div[data-testid="stDecoration"] { display: none; }
        div[data-testid="stStatusWidget"] { display: none; }
        #stDecoration { display: none; }

        .stApp {
            background: #0A0A0A;
            background-image:
                radial-gradient(circle at 15% 20%, rgba(229,9,20,0.08), transparent 35%),
                radial-gradient(circle at 85% 75%, rgba(176,38,255,0.07), transparent 35%),
                radial-gradient(circle at 60% 15%, rgba(255,61,110,0.05), transparent 30%);
            background-attachment: fixed;
        }
        .block-container {
            padding-top: 0 !important;
            max-width: 980px;
            margin: 0 auto;
        }

        /* Thanh gradient nhieu mau o dau trang - lay cam hung tu Netflix,
           thay vi mau do phang don dieu */
        .top-gradient-bar {
            height: 5px;
            width: 100%;
            background: linear-gradient(90deg, #E50914 0%, #FF3D6E 35%, #B026FF 70%, #4A1FBE 100%);
            border-radius: 0 0 6px 6px;
            margin-bottom: 0;
        }

        /* Blob gradient trang tri - hieu ung mo (blur) tao chieu sau,
           dat o goc card giong phong cach Netflix hien dai */
        .blob {
            position: absolute;
            border-radius: 50%;
            filter: blur(28px);
            opacity: 0.35;
            z-index: 0;
            pointer-events: none;
        }
        .blob-red { background: radial-gradient(circle, #E50914, transparent 70%); }
        .blob-purple { background: radial-gradient(circle, #B026FF, transparent 70%); }
        .blob-pink { background: radial-gradient(circle, #FF3D6E, transparent 70%); }

        /* Custom scrollbar */
        ::-webkit-scrollbar { width: 10px; height: 10px; }
        ::-webkit-scrollbar-track { background: #0A0A0A; }
        ::-webkit-scrollbar-thumb { background: #333333; border-radius: 5px; }
        ::-webkit-scrollbar-thumb:hover { background: #E50914; }

        /* Navbar rieng - thay the header mac dinh cua Streamlit */
        .navbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 18px 4px;
            border-bottom: 1px solid #1F1F1F;
            margin-bottom: 40px;
            position: sticky;
            top: 0;
            background-color: rgba(10, 10, 10, 0.92);
            backdrop-filter: blur(8px);
            z-index: 999;
        }
        .navbar-logo {
            font-family: 'Bebas Neue', sans-serif;
            font-size: 1.5rem;
            letter-spacing: 1.5px;
            color: #FFFFFF;
        }
        .navbar-logo span { color: #E50914; }
        .navbar-tag {
            font-size: 0.72rem;
            color: #666666;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            font-weight: 600;
        }

        /* Hero */
        .hero-wrap { padding: 20px 4px 48px 4px; }
        .hero-title {
            font-family: 'Bebas Neue', sans-serif;
            font-size: 4.2rem;
            letter-spacing: 3px;
            color: #FFFFFF;
            margin-bottom: 0;
            line-height: 1;
        }
        .hero-accent {
            background: linear-gradient(90deg, #E50914, #FF3D6E, #B026FF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .hero-subtitle {
            color: #999999;
            font-size: 1.02rem;
            margin-top: 14px;
            max-width: 640px;
            line-height: 1.6;
        }
        .hero-badges { margin-top: 20px; display: flex; gap: 10px; flex-wrap: wrap; }
        .hero-badge {
            background-color: #161616;
            border: 1px solid #2A2A2A;
            border-radius: 20px;
            padding: 6px 16px;
            font-size: 0.78rem;
            color: #AAAAAA;
            font-weight: 500;
        }
        .hero-badge b { color: #E50914; }

        /* File uploader - khung bo tron, vien do khi hover */
        [data-testid="stFileUploader"] {
            background-color: #121212;
            border: 1.5px dashed #2A2A2A;
            border-radius: 14px;
            padding: 8px;
            transition: border-color 0.25s ease, background-color 0.25s ease;
        }
        [data-testid="stFileUploader"]:hover {
            border-color: #E50914;
            background-color: #161010;
        }
        [data-testid="stFileUploaderDropzone"] { background-color: transparent; }

        /* Nut chinh - do Netflix, bo tron, hover sang hon + shadow */
        .stButton > button {
            background: linear-gradient(90deg, #E50914, #FF3D6E);
            color: #FFFFFF;
            border: none;
            border-radius: 8px;
            font-weight: 700;
            font-size: 1.05rem;
            padding: 0.75rem 1.5rem;
            letter-spacing: 0.5px;
            box-shadow: 0 4px 14px rgba(229, 9, 20, 0.35);
            transition: all 0.2s ease;
        }
        .stButton > button:hover {
            background: linear-gradient(90deg, #F6121D, #FF5C87);
            transform: translateY(-1px);
            box-shadow: 0 6px 24px rgba(255, 61, 110, 0.45);
        }
        .stButton > button:active { transform: translateY(0); }

        /* Section header */
        .section-header {
            font-family: 'Bebas Neue', sans-serif;
            font-size: 1.5rem;
            letter-spacing: 1.5px;
            color: #FFFFFF;
            margin-top: 12px;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .section-header::before {
            content: "";
            width: 4px;
            height: 22px;
            background-color: #E50914;
            border-radius: 2px;
            display: inline-block;
        }

        /* Anh xem truoc - bo tron, vien nhe */
        [data-testid="stImage"] img {
            border-radius: 10px;
            border: 1px solid #2A2A2A;
        }

        /* Ket luan chinh - card lon, noi bat, animation fade-in */
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .verdict-card {
            border-radius: 16px;
            padding: 36px 28px;
            margin-bottom: 24px;
            text-align: center;
            animation: fadeInUp 0.4s ease;
        }
        .verdict-match {
            background: linear-gradient(145deg, #0f2a15, #0a1a0e);
            border: 1px solid #1f7a3f;
        }
        .verdict-nomatch {
            background: linear-gradient(145deg, #2e0b0e, #1a0808);
            border: 1px solid #E50914;
        }
        .verdict-icon { font-size: 2.6rem; margin-bottom: 6px; }
        .verdict-label {
            font-family: 'Bebas Neue', sans-serif;
            font-size: 2.8rem;
            letter-spacing: 2px;
            margin: 0;
        }
        .verdict-match .verdict-label { color: #2ecc71; }
        .verdict-nomatch .verdict-label { color: #FF3B44; }
        .verdict-sub { color: #999999; font-size: 0.92rem; margin-top: 6px; }

        /* Grid card ket qua tung model - dang Netflix content card */
        .model-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 14px;
            margin-bottom: 28px;
            animation: fadeInUp 0.5s ease;
        }
        .model-card {
            background-color: #131313;
            border-radius: 12px;
            padding: 20px 16px;
            border: 1px solid #232323;
            position: relative;
            overflow: hidden;
            transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
        }
        .model-card::after {
            content: "";
            position: absolute;
            width: 90px; height: 90px;
            bottom: -30px; right: -30px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(229,9,20,0.25), transparent 70%);
            filter: blur(10px);
            z-index: 0;
        }
        .model-card > * { position: relative; z-index: 1; }
        .model-card:hover {
            transform: translateY(-4px);
            border-color: #E50914;
            box-shadow: 0 8px 24px rgba(0,0,0,0.4);
        }
        .model-card.featured {
            border-color: #E50914;
            background: linear-gradient(160deg, #1a0e0e, #131313);
        }
        .model-name {
            font-weight: 700;
            font-size: 0.8rem;
            color: #CCCCCC;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            margin-bottom: 10px;
        }
        .model-score {
            font-family: 'Bebas Neue', sans-serif;
            font-size: 2.1rem;
            margin: 0;
        }
        .model-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.7rem;
            font-weight: 700;
            margin-top: 10px;
            text-transform: uppercase;
            letter-spacing: 0.4px;
        }
        .badge-match { background-color: #10291a; color: #2ecc71; }
        .badge-nomatch { background-color: #2e1214; color: #FF3B44; }

        .footnote {
            color: #666666;
            font-size: 0.8rem;
            line-height: 1.6;
            border-top: 1px solid #1F1F1F;
            padding-top: 16px;
            margin-top: 8px;
        }

        /* Feature grid - "Vi sao chon he thong nay" - 4 o, blob mau goc
           duoi, lay cam hung tu section "Them ly do de tham gia" cua
           Netflix nhung noi dung rieng cho bai toan xac thuc chu ky */
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin: 32px 0 44px 0;
        }
        .feature-card {
            background: linear-gradient(160deg, #171426, #0f0d1a);
            border: 1px solid #262238;
            border-radius: 14px;
            padding: 24px 20px;
            position: relative;
            overflow: hidden;
            min-height: 150px;
        }
        .feature-card-title {
            font-weight: 700;
            font-size: 1.05rem;
            color: #FFFFFF;
            margin-bottom: 8px;
            position: relative;
            z-index: 1;
        }
        .feature-card-desc {
            font-size: 0.85rem;
            color: #9A94B5;
            line-height: 1.55;
            position: relative;
            z-index: 1;
        }
        .feature-card-icon {
            font-size: 1.6rem;
            margin-bottom: 10px;
            position: relative;
            z-index: 1;
        }

        /* Ghi de mau mac dinh cua Streamlit cho st.success/info/warning/error */
        [data-testid="stAlertContentSuccess"],
        div[data-baseweb="notification"]:has(> div [data-testid="stAlertContentSuccess"]) {
            background-color: #0f2115 !important;
            border: 1px solid #1f7a3f !important;
            color: #4ade80 !important;
        }
        [data-testid="stAlertContentInfo"],
        div[data-baseweb="notification"]:has(> div [data-testid="stAlertContentInfo"]) {
            background-color: #0e1a26 !important;
            border: 1px solid #2563eb !important;
            color: #7dabf8 !important;
        }
        [data-testid="stAlertContentWarning"],
        div[data-baseweb="notification"]:has(> div [data-testid="stAlertContentWarning"]) {
            background-color: #241d0a !important;
            border: 1px solid #E5A600 !important;
            color: #f5c451 !important;
        }
        [data-testid="stAlertContentError"],
        div[data-baseweb="notification"]:has(> div [data-testid="stAlertContentError"]) {
            background-color: #240e0f !important;
            border: 1px solid #E50914 !important;
            color: #ff6b6b !important;
        }
        [data-testid="stAlert"] { border-radius: 10px !important; background-color: #121212 !important; }
        [data-testid="stAlert"] p { color: inherit !important; }

        /* Nut "Browse files" trong file uploader - do Netflix thay vi trang mac dinh */
        [data-testid="stFileUploader"] section button {
            background-color: transparent !important;
            border: 1px solid #E50914 !important;
            color: #E50914 !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
        }
        [data-testid="stFileUploader"] section button:hover {
            background-color: #E50914 !important;
            color: #FFFFFF !important;
        }

        [data-testid="stFileUploader"] label { color: #E0E0E0 !important; font-weight: 600; }
        [data-testid="stMarkdownContainer"] p { color: #C0C0C0; }
        [data-testid="stImageCaption"] { color: #888888 !important; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="top-gradient-bar"></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="navbar">'
        '<div class="navbar-logo">SIGNATURE<span>VERIFY</span></div>'
        '<div class="navbar-tag">Đồ án tốt nghiệp · Khoa học dữ liệu</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="hero-wrap">'
        '<p class="hero-title">XÁC THỰC<br><span class="hero-accent">CHỮ KÝ</span> THÔNG MINH</p>'
        '<p class="hero-subtitle">Hệ thống kết hợp đặc trưng thống kê thủ công (Hu Moments, GLCM) và '
        'học sâu (Siamese Network) để phát hiện chữ ký giả mạo, ứng dụng trong phòng chống gian lận '
        'tài chính-ngân hàng.</p>'
        '<div class="hero-badges">'
        '<div class="hero-badge">Dataset: <b>CEDAR</b></div>'
        '<div class="hero-badge">ROC-AUC: <b>~0.90</b></div>'
        '<div class="hero-badge">4 mô hình so sánh</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="section-header">VÌ SAO CHỌN HỆ THỐNG NÀY</p>'
        '<div class="feature-grid">'
        '<div class="feature-card">'
        '<div class="blob blob-red" style="width:90px;height:90px;bottom:-20px;right:-20px;"></div>'
        '<div class="feature-card-icon">🧬</div>'
        '<div class="feature-card-title">Kết hợp 2 hướng tiếp cận</div>'
        '<div class="feature-card-desc">Đặc trưng thống kê thủ công (Hu Moments, GLCM) kết hợp '
        'embedding học sâu từ Siamese Network.</div>'
        '</div>'
        '<div class="feature-card">'
        '<div class="blob blob-purple" style="width:90px;height:90px;bottom:-20px;right:-20px;"></div>'
        '<div class="feature-card-icon">📊</div>'
        '<div class="feature-card-title">So sánh 4 mô hình</div>'
        '<div class="feature-card-desc">RF, SVM, Siamese Network và Combined được đánh giá song song, '
        'minh bạch từng kết quả.</div>'
        '</div>'
        '<div class="feature-card">'
        '<div class="blob blob-pink" style="width:90px;height:90px;bottom:-20px;right:-20px;"></div>'
        '<div class="feature-card-icon">🎯</div>'
        '<div class="feature-card-title">Hiệu chỉnh theo rủi ro</div>'
        '<div class="feature-card-desc">Ngưỡng quyết định chọn bằng EER trên tập validation, phù hợp '
        'với mức rủi ro ngân hàng chấp nhận.</div>'
        '</div>'
        '<div class="feature-card">'
        '<div class="blob blob-red" style="width:90px;height:90px;bottom:-20px;right:-20px;"></div>'
        '<div class="feature-card-icon">🔍</div>'
        '<div class="feature-card-title">Phân tích minh bạch</div>'
        '<div class="feature-card-desc">Grad-CAM trực quan hóa vùng ảnh mô hình tập trung khi ra '
        'quyết định.</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    if not MODEL_DIR.exists() or not any(MODEL_DIR.iterdir()):
        st.error(
            f"Chưa tìm thấy model đã train tại `{MODEL_DIR}`. Hãy chạy đủ các bước "
            "`01_preprocessing.py` → `01b_generate_pairs.py` → `03_features.py` → "
            "`05_pairwise_baseline.py` → `06_siamese_network.py` → `08_combined_model.py` "
            "trước khi mở demo này."
        )
        return

    with st.spinner("Đang tải model..."):
        models_dict = load_all_models()

    # Neu da chay 11_optimize_accuracy.py, dung model + nguong toi uu
    # accuracy tu decision_config.json (ghi de DECISION_MODEL/threshold).
    decision_cfg = None
    cfg_path = MODEL_DIR / "decision_config.json"
    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                decision_cfg = json.load(f)
        except Exception:
            decision_cfg = None

    active_model_name = decision_cfg["model"] if decision_cfg else DECISION_MODEL
    if active_model_name == "combined" and not models_dict["has_combined"]:
        active_model_name = "siamese"

    if decision_cfg and decision_cfg.get("model") == active_model_name:
        thr = float(decision_cfg["threshold"])
        if active_model_name == "combined":
            models_dict["combined_threshold"] = thr
        elif active_model_name == "siamese":
            models_dict["siamese_threshold"] = thr
        else:
            models_dict["pairwise_thresholds"][f"{active_model_name}_threshold"] = thr

    msg = f"Mô hình đang sử dụng: {MODEL_DISPLAY_NAMES.get(active_model_name, active_model_name)}"
    if decision_cfg:
        msg += f" · ngưỡng tối ưu accuracy (test: {decision_cfg.get('accuracy_test', 0):.1%})"
    st.success(msg)

    st.markdown('<p class="section-header">TẢI LÊN CHỮ KÝ</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        ref_files = st.file_uploader(
            "Chữ ký mẫu (reference) — nên tải 3-5 mẫu để chính xác hơn",
            type=["png", "jpg", "jpeg", "bmp"], accept_multiple_files=True,
        )
    with col2:
        query_file = st.file_uploader("Chữ ký cần kiểm tra (query)", type=["png", "jpg", "jpeg", "bmp"])

    show_gradcam = st.checkbox(
        "🔬 Hiển thị Grad-CAM (giải thích vùng ảnh mô hình tập trung khi ra quyết định)",
        value=False,
        disabled=(gradcam_mod is None),
        help="Cần có file 09_error_analysis_gradcam.py trong src/" if gradcam_mod is None else None,
    )

    if ref_files and query_file:
        if st.button("🔍  SO SÁNH CHỮ KÝ", type="primary", use_container_width=True):
            with tempfile.TemporaryDirectory() as tmp:
                tmp_dir = Path(tmp)
                with st.spinner("Đang tiền xử lý ảnh..."):
                    ref_processed_list = [
                        preprocess_uploaded_image(f, tmp_dir) for f in ref_files
                    ]
                    ref_processed = ref_processed_list[0]
                    query_processed = preprocess_uploaded_image(query_file, tmp_dir)

                st.markdown('<p class="section-header">ẢNH SAU TIỀN XỬ LÝ</p>', unsafe_allow_html=True)
                pcol1, pcol2 = st.columns(2)
                with pcol1:
                    cap = "Reference" if len(ref_processed_list) == 1 else f"Reference (1/{len(ref_processed_list)} mẫu)"
                    st.image(ref_processed, caption=cap, use_container_width=True)
                with pcol2:
                    st.image(query_processed, caption="Query", use_container_width=True)

                with st.spinner("Đang chạy mô hình..."):
                    # CHI CHAY MOT MODEL DUY NHAT (xem DECISION_MODEL o dau file).
                    # MULTI-REFERENCE ENROLLMENT: neu nguoi dung tai len nhieu
                    # chu ky mau, tinh diem cua query voi TUNG mau roi lay TRUNG
                    # BINH. Day la cach he thong sinh trac hoc thuc te lam
                    # (dang ky nhieu mau khi mo tai khoan) - giam nhieu do mot
                    # lan ky bat thuong, cho quyet dinh on dinh hon 1 mau don.
                    active_model = active_model_name

                    def score_one(ref_path):
                        if active_model == "combined":
                            return compute_combined_verdict(models_dict, ref_path, query_processed)
                        if active_model == "siamese":
                            return compute_siamese_verdict(models_dict, ref_path, query_processed)
                        return compute_classical_verdict(models_dict, ref_path, query_processed, active_model)

                    per_ref = [score_one(rp) for rp in ref_processed_list]
                    score = float(np.mean([r[0] for r in per_ref]))
                    threshold = per_ref[0][1]
                    main_match = score >= threshold
                    main_model_label = MODEL_DISPLAY_NAMES.get(active_model, active_model)
                    if len(per_ref) > 1:
                        main_model_label += f" · trung bình {len(per_ref)} mẫu"

                # ----- Ket luan -----
                verdict_class = "verdict-match" if main_match else "verdict-nomatch"
                verdict_text = "KHỚP" if main_match else "KHÔNG KHỚP"
                verdict_sub = "Cùng người ký" if main_match else "Nghi ngờ giả mạo"
                st.markdown(
                    f'<div class="verdict-card {verdict_class}">'
                    f'<p class="verdict-label">{verdict_text}</p>'
                    f'<p class="verdict-sub">{verdict_sub} · {main_model_label}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # ----- Gauge diem so cua model ra quyet dinh -----
                gauge_color = "#2ecc71" if main_match else "#FF3B44"
                val01 = normalize_score_for_display(score, active_model)
                badge_class = "badge-match" if main_match else "badge-nomatch"
                badge_text = "Khớp" if main_match else "Không khớp"
                st.markdown(
                    '<div class="model-grid" style="grid-template-columns:1fr; max-width:260px; margin:0 auto 28px;">'
                    '<div class="model-card featured">'
                    f'<div class="model-name">Độ tương đồng</div>'
                    f'{render_gauge_svg(val01, gauge_color)}'
                    f'<span class="model-badge {badge_class}">{badge_text}</span>'
                    '</div></div>',
                    unsafe_allow_html=True,
                )

                # ----- Grad-CAM (neu duoc bat) -----
                if show_gradcam and gradcam_mod is not None:
                    st.markdown('<p class="section-header">GRAD-CAM · VÙNG ẢNH MÔ HÌNH TẬP TRUNG</p>', unsafe_allow_html=True)
                    with st.spinner("Đang tính Grad-CAM..."):
                        overlay_ref, overlay_query = compute_gradcam_for_pair(models_dict, ref_processed, query_processed)
                    if overlay_ref is not None:
                        gcol1, gcol2 = st.columns(2)
                        with gcol1:
                            st.image(overlay_ref, caption="Reference — vùng đỏ/vàng = ảnh hưởng lớn đến quyết định", use_container_width=True)
                        with gcol2:
                            st.image(overlay_query, caption="Query — vùng đỏ/vàng = ảnh hưởng lớn đến quyết định", use_container_width=True)
                    else:
                        st.warning("Không tính được Grad-CAM (thiếu module 09_error_analysis_gradcam.py).")

                st.markdown(
                    '<p class="footnote">Gauge hiển thị độ tương đồng đã chuẩn hóa về thang 0-100 CHỈ ĐỂ '
                    'TRỰC QUAN; quyết định Khớp/Không khớp dùng đúng điểm số và ngưỡng gốc của mô hình. '
                    'Ngưỡng quyết định được chọn trên tập validation và cố định trước khi đánh giá test.</p>',
                    unsafe_allow_html=True,
                )
    else:
        st.info("Tải lên cả 2 ảnh chữ ký để bắt đầu so sánh.")

    # ----- Section "Ve he thong" - hien thi so lieu hieu nang that -----
    with st.expander("📊 VỀ HỆ THỐNG · Số liệu hiệu năng đã đánh giá"):
        metrics_df = load_performance_metrics()
        if metrics_df is not None:
            display_cols = [c for c in ["model", "accuracy", "roc_auc", "FAR", "FRR", "precision", "recall", "f1"] if c in metrics_df.columns]
            st.dataframe(metrics_df[display_cols].round(4), use_container_width=True, hide_index=True)
            st.caption(
                "Kết quả đánh giá trên tập test (writer-independent, không rò rỉ dữ liệu). "
                "Model Combined kết hợp đặc trưng thủ công (Hu Moments, GLCM...) và embedding từ Siamese Network."
            )
        else:
            st.caption(
                "Chưa tìm thấy file kết quả (results/tables/final_model_comparison*.csv). "
                "Chạy 05_pairwise_baseline.py, 06_siamese_network.py, 08_combined_model.py để tạo số liệu."
            )
        st.markdown(
            "**Phương pháp**: kết hợp đặc trưng thống kê thủ công (Hu Moments, GLCM, tỷ lệ nét, "
            "số điểm giao cắt) và học sâu (Siamese Network, kiến trúc CNN + Contrastive/Triplet Loss). "
            "Đánh giá trên bộ dữ liệu CEDAR, chia writer-independent (writer ở tập test không xuất hiện "
            "ở tập train), ngưỡng quyết định chọn bằng Equal Error Rate trên tập validation."
        )


if __name__ == "__main__":
    main()