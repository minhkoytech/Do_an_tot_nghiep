# Xác thực Chữ ký Viết tay — Kết hợp Đặc trưng Thống kê & Học sâu

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Khóa luận tốt nghiệp - Khoa học Dữ liệu**  
Đại học Kinh tế TP. Hồ Chí Minh · Khoa Công nghệ Thông tin và Kinh doanh

Hệ thống xác thực chữ ký viết tay **ngoại tuyến** (offline): Nhận hai ảnh chữ ký (mẫu + cần kiểm tra), trả về **khớp / không khớp**. Kết hợp đặc trưng thống kê thủ công với Siamese Network, đánh giá theo hướng **độc lập người ký**.

**Repository:** [github.com/minhkoytech/Do_an_tot_nghiep](https://github.com/minhkoytech/Do_an_tot_nghiep)

```
Python · OpenCV · scikit-image · scikit-learn · PyTorch · FastAPI
```

---

## Mục lục

- [Tổng quan](#tổng-quan)
- [Kết quả chính](#kết-quả-chính)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)
- [Cài đặt](#cài-đặt)
- [Chuẩn bị dữ liệu](#chuẩn-bị-dữ-liệu)
- [Chạy pipeline nghiên cứu](#chạy-pipeline-nghiên-cứu)
- [Chạy ứng dụng web](#chạy-ứng-dụng-web)
- [Phương pháp](#phương-pháp)
- [ICDAR SigComp 2011](#icdar-sigcomp-2011)
- [Hạn chế & hướng phát triển](#hạn-chế--hướng-phát-triển)
- [Trích dẫn](#trích-dẫn)

---

## Tổng quan

| Hạng mục | Nội dung |
| --- | --- |
| **Bài toán** | Xác thực **cặp** chữ ký (pair-wise), không phân loại từng ảnh riêng lẻ |
| **Hướng** | Độc lập người ký (writer-independent): người test không xuất hiện khi train |
| **Dataset chính** | [CEDAR Signature Database](https://www.kaggle.com/datasets/shreelakshmigp/cedardataset) - 55 người, 24 thật + 24 giả/người, 300 dpi |
| **External test** | ICDAR SigComp 2011 (chữ ký ngoại tuyến người Hà Lan) |
| **Mô hình** | Random Forest - SVM - Siamese Network (Triplet Loss, ensemble 3) - **Mô hình kết hợp** |
| **Chỉ số** | Accuracy, FAR, FRR, EER, ROC-AUC - theo loại giả mạo, theo người ký, nhiều lần chia |

---

## Kết quả chính

Trung bình ± độ lệch chuẩn qua **5 cách chia người ký** trên CEDAR (độc lập người ký):

| Mô hình | Accuracy (%) | ROC-AUC | EER (%) | FAR giả kỹ năng (%) | FAR giả ngẫu nhiên (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 77,8 ± 2,8 | 0,862 ± 0,030 | 21,7 ± 3,1 | 34,8 ± 5,9 | 4,1 ± 1,1 |
| SVM | 77,4 ± 2,4 | 0,857 ± 0,026 | 22,2 ± 2,6 | 36,8 ± 6,2 | 4,7 ± 1,0 |
| Siamese Network | 79,5 ± 1,3 | 0,882 ± 0,015 | 20,1 ± 1,5 | 26,8 ± 7,1 | 2,8 ± 2,4 |
| **Mô hình kết hợp** | **79,4 ± 2,0** | **0,878 ± 0,025** | **20,2 ± 2,1** | **28,2 ± 6,1** | **3,0 ± 2,5** |

Tại **FAR mục tiêu 10%**: mô hình kết hợp có FRR trung bình ~30%, thấp hơn RF/SVM (~38%).  
Giả mạo có kỹ năng vẫn là điểm nghẽn (FAR ~27-37%); giả mạo ngẫu nhiên đã kiểm soát tốt (FAR < 5%).

Trên **ICDAR SigComp 2011** (không fine-tune): accuracy ~74,6–76,4%.

---

## Cấu trúc thư mục

Cấu trúc **khớp với repository** [minhkoytech/Do_an_tot_nghiep](https://github.com/minhkoytech/Do_an_tot_nghiep):

```
Do_an_tot_nghiep/
├── data/
│   ├── raw/
│   │   ├── cedar/                 # full_org/ + full_forg/ (ảnh gốc, thường không commit)
│   │   └── icdar/                 # ICDAR - chỉ external test
│   └── processed/                 # ảnh sạch, labels, features, cặp
├── src/                           # Pipeline nghiên cứu (chạy tuần tự)
│   ├── 01_preprocessing.py
│   ├── 01b_generate_pairs.py
│   ├── 02_eda.py
│   ├── 03_features.py
│   ├── 04_stats_tests.py
│   ├── 05_pairwise_baseline.py    # RF & SVM dạng cặp
│   ├── 06_siamese_network.py
│   ├── 07_cross_validation.py
│   ├── 08_combined_model.py       # mô hình kết hợp
│   ├── 09_error_analysis_gradcam.py
│   ├── 10_threshold_analysis.py
│   ├── 11_optimize_accuracy.py
│   ├── 12_external_test_icdar.py
│   ├── 13_tables_section_4_6.py
│   └── 14_set_operating_point.py
├── backend/
│   ├── main.py                    # FastAPI: API + serve frontend cùng origin
│   └── requirements.txt
├── frontend/
│   └── index.html                 # Giao diện web (tab xác thực / dashboard / giới thiệu)
├── model_artifacts/               # *.joblib, *.pt, *.json (model đã train)
├── results/                       # figures/, tables/
├── requirements.txt               # dependency pipeline / train
├── .gitignore
└── README.md
```

**Backend** import trực tiếp logic từ `src/` (`01_preprocessing`, `03_features`, `06_siamese_network`, `08_combined_model`, …) để inference giống hệt lúc train. Frontend được serve từ cùng process FastAPI -> một domain, tránh CORS.

---

## Cài đặt

```bash
git clone https://github.com/minhkoytech/Do_an_tot_nghiep.git
cd Do_an_tot_nghiep

python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows
# venv\Scripts\activate

# Dependency pipeline + train
pip install -r requirements.txt

# Dependency chạy web app
pip install -r backend/requirements.txt
```

**Yêu cầu:** Python ≥ 3.9. GPU khuyến nghị khi train Siamese (vẫn chạy được CPU).

---

## Chuẩn bị dữ liệu

### CEDAR (bắt buộc cho train / evaluate)

1. Tải [CEDAR trên Kaggle](https://www.kaggle.com/datasets/shreelakshmigp/cedardataset).
2. Đặt vào:

```
data/raw/cedar/
├── full_org/      # chữ ký thật
└── full_forg/     # chữ ký giả (skilled)
```

### ICDAR SigComp 2011 (tùy chọn — external test)

- Phần ngoại tuyến người Hà Lan -> `data/raw/icdar/`.
- **Không** tham gia train / chọn ngưỡng.

### Model đã train

Thư mục `model_artifacts/` trên repo đã có các file cần cho demo (RF, SVM, Siamese ensemble, combined, threshold JSON). Nếu clone thiếu file lớn, tải lại từ Releases hoặc train lại bằng pipeline bên dưới.

---

## Chạy pipeline nghiên cứu

Chạy **từ thư mục gốc** project, theo thứ tự số:

```bash
# 1. Tiền xử lý + sinh cặp
python src/01_preprocessing.py
python src/01b_generate_pairs.py

# 2. EDA + đặc trưng + kiểm định
python src/02_eda.py
python src/03_features.py
python src/04_stats_tests.py

# 3. Huấn luyện baseline + Siamese + kết hợp
python src/05_pairwise_baseline.py
python src/06_siamese_network.py
python src/07_cross_validation.py
python src/08_combined_model.py

# 4. Phân tích lỗi, ngưỡng, external test
python src/09_error_analysis_gradcam.py
python src/10_threshold_analysis.py
python src/11_optimize_accuracy.py
python src/12_external_test_icdar.py
python src/13_tables_section_4_6.py
python src/14_set_operating_point.py
```

**Chống rò rỉ dữ liệu**

- Chia theo `writer_id` (35 train / 10 val / 10 test).
- Cặp giả ngẫu nhiên chỉ ghép trong cùng tập.
- Chọn đặc trưng, siêu tham số, ngưỡng chỉ trên train/validation.
- Tập test chỉ dùng một lần sau khi mọi thứ đã cố định.

---

## Chạy ứng dụng web

Backend FastAPI **vừa** phục vụ API **vừa** serve `frontend/index.html` trên cùng origin.

```bash
cd backend
pip install -r requirements.txt   # nếu chưa cài
uvicorn main:app --reload --port 8000
```

Mở trình duyệt: **http://localhost:8000**

| Thành phần | Đường dẫn / chức năng |
| --- | --- |
| Giao diện | `GET /` → `frontend/index.html` |
| Health | `GET /api/health` |
| Xác thực | `POST /api/verify` (2 file ảnh + tùy chọn Grad-CAM) |

**Không** dùng `uvicorn app.main:app` — repo **không** có thư mục `app/`.

Mô hình ra quyết định mặc định lấy từ `model_artifacts/decision_config.json` (thường là **combined**). Ảnh upload chỉ lưu tạm trong lúc xử lý rồi xóa.

---

## Phương pháp

| Bước | Kỹ thuật |
| --- | --- |
| Tiền xử lý | Xám → kiểm tra/đảo nền -> Otsu -> khử nhiễu -> crop -> **220×155** |
| Sinh cặp | Thật-thật, thật-giả kỹ năng, thật-người khác (độc lập người ký) |
| Đặc trưng | Hình học, Hu Moments, GLCM -> chọn đặc trưng có ý nghĩa ở **mức cặp** |
| Baseline | RF / SVM trên vector Δ đặc trưng (`05_pairwise_baseline.py`) |
| Siamese | CNN + Triplet Loss, ensemble 3 mạng (`06_siamese_network.py`) |
| Kết hợp | Δ đặc trưng + Δ embedding -> RF (`08_combined_model.py`) |
| Ngưỡng | EER - FAR mục tiêu - tối ưu accuracy (`10_`, `11_`, `14_`) |
| Giải thích | Grad-CAM (`09_error_analysis_gradcam.py`) |

---

## ICDAR SigComp 2011

Chỉ dùng làm **external test** (`12_external_test_icdar.py`). Mô hình và ngưỡng giữ nguyên từ CEDAR. Khi triển khai thực tế cần hiệu chỉnh ngưỡng bằng dữ liệu tại chỗ.

---

## Hạn chế & hướng phát triển

- Chỉ ~35 người ký train -> trần ROC-AUC ~0,88–0,90; FAR giả kỹ năng còn cao.
- So sánh với một chữ ký mẫu mỗi lần.
- Ảnh quét sạch; chưa xử lý ảnh điện thoại, dòng kẻ, con dấu.
- Ưu tiên tiếp: nhiều mẫu/người; pretrain bộ lớn (GPDS); ngưỡng theo khách hàng; tiền xử lý ảnh thực tế.

---

## Trích dẫn

```text
Nông Minh Khôi (2026). Xây dựng hệ thống xác thực chữ ký viết tay
dựa trên kết hợp đặc trưng thống kê và học sâu, ứng dụng trong
phòng chống gian lận tài chính - ngân hàng.
Khóa luận tốt nghiệp, Đại học Kinh tế TP. Hồ Chí Minh.
Giảng viên hướng dẫn: TS. Nguyễn Đăng Cao.
```

---

**Sinh viên:** Nông Minh Khôi · MSSV 31231025344 · Khoa học dữ liệu K49  
**GVHD:** TS. Nguyễn Đăng Cao  
**Repo:** [minhkoytech/Do_an_tot_nghiep](https://github.com/minhkoytech/Do_an_tot_nghiep)  
**TP. Hồ Chí Minh — 2026**
