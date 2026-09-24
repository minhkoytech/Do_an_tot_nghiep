"""
backend/main.py
----------------
FastAPI backend cho he thong xac thuc chu ky - phuc vu CA API (chay model
that) VA frontend tinh (index.html) tu CUNG MOT DOMAIN. Lam vay de tranh
van de CORS/CSP khi frontend goi API - toan bo chay tren 1 origin duy
nhat, chi can deploy MOT NOI (khong can 2 dich vu rieng biet nhu kien
truc "frontend rieng goi API rieng" thong thuong).

QUAN TRONG: KHONG viet lai logic tien xu ly / trich dac trung / kien
truc model - import TRUC TIEP tu 01_preprocessing.py, 03_features.py,
06_siamese_network.py, 08_combined_model.py (thu muc src/, ben canh
thu muc backend/ nay) de dam bao dung Y HET pipeline da dung khi train.

Cach chay local:
    cd backend
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000
    # Mo trinh duyet: http://localhost:8000

Cau truc thu muc can co (project root):
    project_root/
        src/                  (01_preprocessing.py, 03_features.py, ...)
        model_artifacts/      (*.joblib, *.pt, *.json - da train san)
        backend/main.py       (file nay)
        backend/requirements.txt
        frontend/index.html   (giao diện)
        Dockerfile
"""

import importlib.util
import json
import tempfile
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Duong dan
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
MODEL_DIR = PROJECT_ROOT / "model_artifacts"
FRONTEND_DIR = PROJECT_ROOT / "frontend"


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
    pass  # Grad-CAM la tinh nang tuy chon

SELECTED_FEATURES = siamese_mod.SELECTED_FEATURES

app = FastAPI(title="SignatureVerify API")

# CORS: cho phep goi tu bat ky origin nao (huu ich khi test frontend
# rieng le trong luc phat trien; khi deploy chung 1 domain thi khong
# thuc su can nhung de lai cho an toan/linh hoat)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Load toan bo model MOT LAN khi server khoi dong (khong load lai moi request)
# ---------------------------------------------------------------------------
MODELS = {}


@app.on_event("startup")
def load_all_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not MODEL_DIR.exists() or not any(MODEL_DIR.iterdir()):
        print(f"[CANH BAO] Khong tim thay model_artifacts tai {MODEL_DIR}. "
              f"API se bao loi khi goi /api/verify cho den khi co model.")
        MODELS["ready"] = False
        return

    rf = joblib.load(MODEL_DIR / "pairwise_rf.joblib")
    svm = joblib.load(MODEL_DIR / "pairwise_svm.joblib")
    scaler = joblib.load(MODEL_DIR / "pairwise_scaler.joblib")
    imputer = joblib.load(MODEL_DIR / "pairwise_imputer.joblib")
    with open(MODEL_DIR / "pairwise_threshold.json") as f:
        pairwise_thresholds = json.load(f)

    with open(MODEL_DIR / "siamese_threshold.json") as f:
        siamese_config = json.load(f)
    embedding_dim = siamese_config.get("embedding_dim", 64)

    siamese_models = []
    model_files = sorted(MODEL_DIR.glob("siamese_model_*.pt"))
    if not model_files:
        legacy = MODEL_DIR / "siamese_best.pt"
        if legacy.exists():
            model_files = [legacy]
    for model_path in model_files:
        model = siamese_mod.SiameseNetwork(embedding_dim=embedding_dim, backbone_type="custom").to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        siamese_models.append(model)

    MODELS.update({
        "ready": True,
        "device": device,
        "rf": rf, "svm": svm, "scaler": scaler, "imputer": imputer,
        "pairwise_thresholds": pairwise_thresholds,
        "siamese_models": siamese_models,
        "siamese_threshold": siamese_config["threshold"],
        "embedding_dim": embedding_dim,
        "has_combined": False,
    })

    combined_path = MODEL_DIR / "combined_rf.joblib"
    if combined_path.exists():
        MODELS["combined_rf"] = joblib.load(combined_path)
        MODELS["combined_scaler"] = joblib.load(MODEL_DIR / "combined_scaler.joblib")
        MODELS["combined_imputer"] = joblib.load(MODEL_DIR / "combined_imputer.joblib")
        with open(MODEL_DIR / "combined_threshold.json") as f:
            MODELS["combined_threshold"] = json.load(f)["threshold"]
        MODELS["has_combined"] = True

    # Neu da chay 11_optimize_accuracy.py: dung model + nguong toi uu accuracy
    cfg_path = MODEL_DIR / "decision_config.json"
    MODELS["decision_model"] = "combined" if MODELS["has_combined"] else "siamese"
    MODELS["decision_info"] = None
    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            m = cfg["model"]
            if m == "combined" and not MODELS["has_combined"]:
                m = "siamese"
            else:
                thr = float(cfg["threshold"])
                if m == "combined":
                    MODELS["combined_threshold"] = thr
                elif m == "siamese":
                    MODELS["siamese_threshold"] = thr
                else:
                    MODELS["pairwise_thresholds"][f"{m}_threshold"] = thr
            MODELS["decision_model"] = m
            MODELS["decision_info"] = cfg
        except Exception as e:
            print(f"[CANH BAO] Khong doc duoc decision_config.json: {e}")

    print(f"Model ra quyet dinh: {MODELS['decision_model'].upper()}")
    print(f"Da load {len(siamese_models)} model Siamese, RF, SVM"
          f"{', Combined' if MODELS['has_combined'] else ''}.")


# ---------------------------------------------------------------------------
# Ham xu ly - tai su dung dung logic voi app.py (Streamlit) truoc do
# ---------------------------------------------------------------------------
def preprocess_uploaded_bytes(file_bytes: bytes, filename: str, tmp_dir: Path) -> str:
    raw_path = tmp_dir / f"raw_{filename}"
    with open(raw_path, "wb") as f:
        f.write(file_bytes)
    processed = preprocess_mod.preprocess_image(str(raw_path))
    processed_path = tmp_dir / f"processed_{filename}.png"
    cv2.imwrite(str(processed_path), processed)
    return str(processed_path)


def load_image_tensor(path: str) -> torch.Tensor:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0)


@torch.no_grad()
def compute_siamese_verdict(ref_path: str, query_path: str):
    device = MODELS["device"]
    img1 = load_image_tensor(ref_path).to(device)
    img2 = load_image_tensor(query_path).to(device)
    scores = [-model(img1, img2).item() for model in MODELS["siamese_models"]]
    avg_score = float(np.mean(scores))
    threshold = MODELS["siamese_threshold"]
    return avg_score, threshold, avg_score >= threshold


def compute_classical_verdict(ref_path: str, query_path: str, model_name: str):
    ref_features = features_mod.extract_all_features(ref_path)
    query_features = features_mod.extract_all_features(query_path)
    delta_vector = np.array([[abs(ref_features[f] - query_features[f]) for f in SELECTED_FEATURES]])
    delta_imputed = MODELS["imputer"].transform(delta_vector)
    delta_scaled = MODELS["scaler"].transform(delta_imputed)

    model = MODELS[model_name]
    threshold = MODELS["pairwise_thresholds"][f"{model_name}_threshold"]
    if model_name == "svm":
        score = float(model.decision_function(delta_scaled)[0])
    else:
        score = float(model.predict_proba(delta_scaled)[0, 1])
    return score, threshold, score >= threshold


@torch.no_grad()
def compute_combined_verdict(ref_path: str, query_path: str):
    ref_features = features_mod.extract_all_features(ref_path)
    query_features = features_mod.extract_all_features(query_path)
    classical_delta = [abs(ref_features[f] - query_features[f]) for f in SELECTED_FEATURES]

    device = MODELS["device"]
    ref_emb = combined_mod.compute_embedding_for_image(MODELS["siamese_models"], ref_path, device)
    query_emb = combined_mod.compute_embedding_for_image(MODELS["siamese_models"], query_path, device)
    embedding_delta = np.abs(ref_emb - query_emb)

    full_vector = np.array([classical_delta + list(embedding_delta)])
    imputed = MODELS["combined_imputer"].transform(full_vector)
    scaled = MODELS["combined_scaler"].transform(imputed)
    score = float(MODELS["combined_rf"].predict_proba(scaled)[0, 1])
    threshold = MODELS["combined_threshold"]
    return score, threshold, score >= threshold


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "models_ready": MODELS.get("ready", False)}


@app.post("/api/verify")
async def verify(reference: UploadFile = File(...), query: UploadFile = File(...), include_gradcam: bool = False):
    if not MODELS.get("ready", False):
        raise HTTPException(status_code=503, detail="Model chua san sang. Kiem tra model_artifacts/ tren server.")

    ref_bytes = await reference.read()
    query_bytes = await query.read()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        try:
            ref_path = preprocess_uploaded_bytes(ref_bytes, reference.filename, tmp_dir)
            query_path = preprocess_uploaded_bytes(query_bytes, query.filename, tmp_dir)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Khong xu ly duoc anh: {e}")

        # CHI CHAY MOT MODEL DUY NHAT (model ra quyet dinh) - nhanh hon va
        # giao dien cung chi hien mot ket luan duy nhat.
        main_model = MODELS.get("decision_model", "siamese")
        display_names = {
            "combined": "Combined (đặc trưng thủ công + học sâu)",
            "siamese": "Siamese Network",
            "rf": "Random Forest",
            "svm": "SVM",
        }
        if main_model == "combined":
            score, threshold, match = compute_combined_verdict(ref_path, query_path)
        elif main_model == "siamese":
            score, threshold, match = compute_siamese_verdict(ref_path, query_path)
        else:
            score, threshold, match = compute_classical_verdict(ref_path, query_path, main_model)

        results = {main_model: {
            "name": display_names.get(main_model, main_model),
            "score": score, "threshold": threshold, "match": match, "featured": True,
        }}

        # Anh da tien xu ly, encode base64 de frontend hien thi truc tiep
        import base64
        with open(ref_path, "rb") as f:
            ref_b64 = base64.b64encode(f.read()).decode()
        with open(query_path, "rb") as f:
            query_b64 = base64.b64encode(f.read()).decode()

        gradcam_ref_b64, gradcam_query_b64 = None, None
        if include_gradcam and gradcam_mod is not None:
            device = MODELS["device"]
            model = MODELS["siamese_models"][0]
            target_layer = model.embedding_net.conv[-2]
            img_a = gradcam_mod.load_image_tensor(ref_path, device)
            img_b = gradcam_mod.load_image_tensor(query_path, device)
            cam_a, cam_b, _ = gradcam_mod.compute_pair_gradcam(model, target_layer, img_a, img_b)
            overlay_a = gradcam_mod.overlay_heatmap(ref_path, cam_a)
            overlay_b = gradcam_mod.overlay_heatmap(query_path, cam_b)
            gradcam_ref_path = tmp_dir / "gradcam_ref.png"
            gradcam_query_path = tmp_dir / "gradcam_query.png"
            cv2.imwrite(str(gradcam_ref_path), cv2.cvtColor(overlay_a, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(gradcam_query_path), cv2.cvtColor(overlay_b, cv2.COLOR_RGB2BGR))
            with open(gradcam_ref_path, "rb") as f:
                gradcam_ref_b64 = base64.b64encode(f.read()).decode()
            with open(gradcam_query_path, "rb") as f:
                gradcam_query_b64 = base64.b64encode(f.read()).decode()

    response = {
        "results": results,
        "main_model": main_model,
        "main_match": results[main_model]["match"],
        "reference_image": f"data:image/png;base64,{ref_b64}",
        "query_image": f"data:image/png;base64,{query_b64}",
    }
    if gradcam_ref_b64:
        response["gradcam_reference"] = f"data:image/png;base64,{gradcam_ref_b64}"
        response["gradcam_query"] = f"data:image/png;base64,{gradcam_query_b64}"
    return response


@app.get("/api/active-model")
def active_model():
    """Thong tin model dang duoc dung de ra quyet dinh (hien tren giao dien)."""
    names = {
        "combined": "Combined (đặc trưng thủ công + học sâu)",
        "siamese": "Siamese Network", "rf": "Random Forest", "svm": "SVM",
    }
    m = MODELS.get("decision_model")
    return {
        "model": m,
        "display_name": names.get(m, m or ""),
        "info": MODELS.get("decision_info"),
        "ready": MODELS.get("ready", False),
    }


@app.get("/api/metrics")
def get_metrics():
    """Doc bang so sanh 4 model tu ket qua da co san (neu co) - hien thi trong section 'Ve he thong'."""
    import pandas as pd
    for filename in ["final_model_comparison_with_combined.csv", "final_model_comparison.csv"]:
        path = PROJECT_ROOT / "results" / "tables" / filename
        if path.exists():
            df = pd.read_csv(path)
            return {"available": True, "rows": df.to_dict("records")}
    return {"available": False, "rows": []}



# ---------------------------------------------------------------------------
# Phuc vu frontend tinh (index.html, ...) tu CUNG server - tranh CORS/CSP
# ---------------------------------------------------------------------------
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(str(FRONTEND_DIR / "index.html"))
