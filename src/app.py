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
    st.set_page_config(page_title="Xac thuc chu ky viet tay", page_icon="✍️", layout="centered")
    st.title("✍️ Xác thực chữ ký viết tay")
    st.caption(
        "Đồ án tốt nghiệp: Xây dựng hệ thống xác thực chữ ký viết tay dựa trên kết hợp "
        "đặc trưng thống kê và học sâu, ứng dụng trong phòng chống gian lận tài chính-ngân hàng."
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

    col1, col2 = st.columns(2)
    with col1:
        ref_file = st.file_uploader("Chữ ký mẫu (reference - đã xác nhận là thật)", type=["png", "jpg", "jpeg", "bmp"])
    with col2:
        query_file = st.file_uploader("Chữ ký cần kiểm tra (query)", type=["png", "jpg", "jpeg", "bmp"])

    if ref_file and query_file:
        if st.button("🔍 So sánh chữ ký", type="primary", use_container_width=True):
            with tempfile.TemporaryDirectory() as tmp:
                tmp_dir = Path(tmp)
                with st.spinner("Đang tiền xử lý ảnh..."):
                    ref_processed = preprocess_uploaded_image(ref_file, tmp_dir)
                    query_processed = preprocess_uploaded_image(query_file, tmp_dir)

                st.subheader("Ảnh sau tiền xử lý")
                pcol1, pcol2 = st.columns(2)
                with pcol1:
                    st.image(ref_processed, caption="Reference (đã xử lý)", use_container_width=True)
                with pcol2:
                    st.image(query_processed, caption="Query (đã xử lý)", use_container_width=True)

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

                    rows_model = ["Siamese Network", "Random Forest (baseline)", "SVM (baseline)"]
                    rows_score = [siamese_score, rf_score, svm_score]
                    rows_threshold = [siamese_threshold, rf_threshold, svm_threshold]
                    rows_match = [siamese_match, rf_match, svm_match]

                    if models_dict["has_combined"]:
                        combined_score, combined_threshold, combined_match = compute_combined_verdict(
                            models_dict, ref_processed, query_processed
                        )
                        rows_model.insert(0, "Combined (đề xuất chính) ⭐")
                        rows_score.insert(0, combined_score)
                        rows_threshold.insert(0, combined_threshold)
                        rows_match.insert(0, combined_match)
                        main_match = combined_match
                        main_model_label = "Combined"
                    else:
                        main_match = siamese_match
                        main_model_label = "Siamese Network"

                st.subheader("Kết quả")

                verdict_text = "✅ KHỚP (cùng người ký)" if main_match else "❌ KHÔNG KHỚP (nghi ngờ giả mạo)"
                verdict_color = "green" if main_match else "red"
                st.markdown(f"### Kết luận chính ({main_model_label}): :{verdict_color}[{verdict_text}]")

                results_table = {
                    "Model": rows_model,
                    "Điểm số": [f"{s:.4f}" for s in rows_score],
                    "Ngưỡng quyết định": [f"{t:.4f}" for t in rows_threshold],
                    "Kết luận": ["Khớp" if m else "Không khớp" for m in rows_match],
                }
                st.table(results_table)

                agree_count = sum(rows_match)
                total_models = len(rows_match)
                if agree_count in (0, total_models):
                    st.info("Tất cả model đều đồng thuận kết quả — độ tin cậy cao.")
                else:
                    st.warning(
                        f"Các model KHÔNG đồng thuận ({agree_count}/{total_models} model kết luận 'Khớp'). "
                        "Đây là trường hợp khó, nên xem xét thêm bằng mắt hoặc chuyên gia."
                    )

                st.caption(
                    "Lưu ý: điểm số Siamese là -khoảng cách embedding. Điểm RF/Combined là xác suất "
                    "phân loại. Điểm SVM là decision function. Ngưỡng quyết định của RF/SVM/Siamese "
                    "chọn bằng EER trên tập validation; model Combined dùng cùng nguyên tắc. Theo phân "
                    "tích tại nhiều mức FAR mục tiêu (xem 10_threshold_analysis.py), model Combined cho "
                    "kết quả tốt nhất trong hầu hết các mức rủi ro ngân hàng có thể chấp nhận."
                )
    else:
        st.info("Tải lên cả 2 ảnh chữ ký để bắt đầu so sánh.")


if __name__ == "__main__":
    main()
