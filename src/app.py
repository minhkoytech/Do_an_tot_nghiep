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

SELECTED_FEATURES = siamese_mod.SELECTED_FEATURES


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

        .stApp { background-color: #0A0A0A; }
        .block-container {
            padding-top: 0 !important;
            max-width: 980px;
            margin: 0 auto;
        }

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
        .hero-accent { color: #E50914; }
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
            background-color: #E50914;
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
            background-color: #F6121D;
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(229, 9, 20, 0.5);
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
            transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
        }
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

    n_siamese = len(models_dict["siamese_models"])
    status_msg = f"Đã tải {n_siamese} model Siamese (ensemble) + RF + SVM"
    if models_dict["has_combined"]:
        status_msg += " + Combined (đề xuất chính)."
    else:
        status_msg += ". Chưa có model Combined — chạy `08_combined_model.py` để bổ sung."
    st.success(status_msg)

    st.markdown('<p class="section-header">TẢI LÊN CHỮ KÝ</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        ref_file = st.file_uploader("Chữ ký mẫu (reference)", type=["png", "jpg", "jpeg", "bmp"])
    with col2:
        query_file = st.file_uploader("Chữ ký cần kiểm tra (query)", type=["png", "jpg", "jpeg", "bmp"])

    if ref_file and query_file:
        if st.button("🔍  SO SÁNH CHỮ KÝ", type="primary", use_container_width=True):
            with tempfile.TemporaryDirectory() as tmp:
                tmp_dir = Path(tmp)
                with st.spinner("Đang tiền xử lý ảnh..."):
                    ref_processed = preprocess_uploaded_image(ref_file, tmp_dir)
                    query_processed = preprocess_uploaded_image(query_file, tmp_dir)

                st.markdown('<p class="section-header">ẢNH SAU TIỀN XỬ LÝ</p>', unsafe_allow_html=True)
                pcol1, pcol2 = st.columns(2)
                with pcol1:
                    st.image(ref_processed, caption="Reference", use_container_width=True)
                with pcol2:
                    st.image(query_processed, caption="Query", use_container_width=True)

                with st.spinner("Đang chạy các mô hình..."):
                    siamese_score, siamese_threshold, siamese_match = compute_siamese_verdict(
                        models_dict, ref_processed, query_processed
                    )
                    rf_score, rf_threshold, rf_match = compute_classical_verdict(
                        models_dict, ref_processed, query_processed, "rf"
                    )
                    svm_score, svm_threshold, svm_match = compute_classical_verdict(
                        models_dict, ref_processed, query_processed, "svm"
                    )

                    cards = [
                        {"name": "Siamese Network", "score": siamese_score, "match": siamese_match, "featured": False},
                        {"name": "Random Forest", "score": rf_score, "match": rf_match, "featured": False},
                        {"name": "SVM", "score": svm_score, "match": svm_match, "featured": False},
                    ]

                    if models_dict["has_combined"]:
                        combined_score, combined_threshold, combined_match = compute_combined_verdict(
                            models_dict, ref_processed, query_processed
                        )
                        cards.insert(0, {"name": "Combined ⭐", "score": combined_score, "match": combined_match, "featured": True})
                        main_match = combined_match
                        main_model_label = "Combined Model"
                    else:
                        main_match = siamese_match
                        main_model_label = "Siamese Network"

                # ----- Ket luan chinh -----
                verdict_class = "verdict-match" if main_match else "verdict-nomatch"
                verdict_text = "KHỚP" if main_match else "KHÔNG KHỚP"
                verdict_sub = "Cùng người ký" if main_match else "Nghi ngờ giả mạo"
                st.markdown(
                    f'<div class="verdict-card {verdict_class}">'
                    f'<p class="verdict-label">{verdict_text}</p>'
                    f'<p class="verdict-sub">{verdict_sub} · Kết luận theo {main_model_label}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                # ----- Grid card ket qua tung model -----
                st.markdown('<p class="section-header">CHI TIẾT TỪNG MÔ HÌNH</p>', unsafe_allow_html=True)
                cards_html = '<div class="model-grid">'
                for c in cards:
                    badge_class = "badge-match" if c["match"] else "badge-nomatch"
                    badge_text = "Khớp" if c["match"] else "Không khớp"
                    score_color = "#2ecc71" if c["match"] else "#E50914"
                    featured_class = "featured" if c["featured"] else ""
                    cards_html += (
                        f'<div class="model-card {featured_class}">'
                        f'<div class="model-name">{c["name"]}</div>'
                        f'<p class="model-score" style="color:{score_color}">{c["score"]:.3f}</p>'
                        f'<span class="model-badge {badge_class}">{badge_text}</span>'
                        f'</div>'
                    )
                cards_html += '</div>'
                st.markdown(cards_html, unsafe_allow_html=True)

                agree_count = sum(c["match"] for c in cards)
                total_models = len(cards)
                if agree_count in (0, total_models):
                    st.info("✓ Tất cả model đều đồng thuận kết quả — độ tin cậy cao.")
                else:
                    st.warning(
                        f"⚠ Các model KHÔNG đồng thuận ({agree_count}/{total_models} model kết luận 'Khớp'). "
                        "Đây là trường hợp khó, nên xem xét thêm bằng mắt hoặc chuyên gia."
                    )

                st.markdown(
                    '<p class="footnote">Điểm số Siamese là -khoảng cách embedding. Điểm RF/Combined là '
                    'xác suất phân loại. Điểm SVM là decision function. Ngưỡng quyết định chọn bằng '
                    'Equal Error Rate (EER) trên tập validation, cố định trước khi đánh giá test.</p>',
                    unsafe_allow_html=True,
                )
    else:
        st.info("Tải lên cả 2 ảnh chữ ký để bắt đầu so sánh.")


if __name__ == "__main__":
    main()