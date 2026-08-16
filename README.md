# Xác thực Chữ ký Viết tay bằng Học máy & Học sâu

**Đồ án tốt nghiệp — Ngành Khoa học Dữ liệu**

Ứng dụng kết hợp đặc trưng thống kê và học máy để phát hiện chữ ký giả mạo, phục vụ phòng chống gian lận tài chính - ngân hàng.

`Python` · `OpenCV` · `scikit-learn` · `scikit-image` · `Streamlit` · `SciPy`

---

## Mục lục

- [1. Giới thiệu](#1-giới-thiệu)
- [2. Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
- [3. Sơ đồ pipeline](#3-sơ-đồ-pipeline)
- [4. Cài đặt](#4-cài-đặt)
- [5. Chuẩn bị dữ liệu](#5-chuẩn-bị-dữ-liệu)
- [6. Cách chạy](#6-cách-chạy)
- [7. Phương pháp luận](#7-phương-pháp-luận)
- [8. Web app demo](#8-web-app-demo)
- [9. Vai trò của ICDAR SigComp 2011](#9-vai-trò-của-icdar-sigcomp-2011)

---

## 1. Giới thiệu

Ngân hàng vẫn dùng chữ ký viết tay để xác thực trên séc, hợp đồng vay, ủy nhiệm chi. Xác thực thủ công bằng mắt tốn thời gian và dễ sai sót, đặc biệt với giả mạo có kỹ năng.

Dự án xây dựng hệ thống tự động phân loại **chữ ký thật / chữ ký giả** từ ảnh, theo đúng quy trình khoa học dữ liệu: phân tích khám phá dữ liệu -> trích đặc trưng có cơ sở thống kê -> kiểm định giả thuyết -> huấn luyện & đánh giá mô hình bằng cross-validation không rò rỉ dữ liệu -> demo ứng dụng thực tế.

| | |
|---|---|
| **Bài toán** | Phân loại nhị phân: chữ ký thật (genuine) vs giả mạo (skilled forgery) |
| **Loại dữ liệu** | Ảnh chữ ký tĩnh (offline signature) |
| **Dataset chính** | [CEDAR Signature Database](https://www.kaggle.com/datasets/shreelakshmigp/cedardataset) |
| **Dataset external test** | ICDAR SigComp 2011 |
| **Phương pháp** | Feature engineering thủ công (Hu Moments, GLCM) + kiểm định thống kê + SVM/Random Forest |

## 2. Cấu trúc thư mục

```
signature_project/
│
├── data/
│   ├── raw/                       # Dataset gốc
│   │   ├── cedar/
│   │   │   ├── full_org/          # Chữ ký thật
│   │   │   └── full_forg/         # Chữ ký giả
│   │   └── icdar/              
│   └── processed/                 # Tự sinh ra sau khi chạy pipeline
│       ├── labels.csv
│       ├── labels_with_eda.csv
│       └── features.csv
│                   
│
├── results/
│   ├── figures/                   # Biểu đồ EDA
│   └── tables/                    # Kết quả CV, kiểm định thống kê
│
├── src/
│   ├── 01_preprocessing.py        # Nhị phân hóa, crop, resize ảnh
│   ├── 02_eda.py                  # Phân tích khám phá dữ liệu
│   ├── 03_features.py             # Trích Hu Moments, GLCM, baseline ratio
│   ├── 04_stats_tests.py          # Mann-Whitney U
│   ├── 05_cv_pipeline.py          # Train + đánh giá bằng k-fold CV chuẩn
│   └── 06_train_final_model.py    # Train model cuối, lưu lại cho web app
│
├── model_artifacts/                # model.joblib, scaler.joblib...
│
├── app.py                          # Web demo
├── requirements.txt
└── README.md
```

## 3. Sơ đồ pipeline

```
┌─────────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Ảnh chữ ký gốc │ --> │ Tiền xử lý   │ --> │       EDA        │
│  (scan/chụp)    │     │ (01)         │     │       (02)       │
└─────────────────┘     └──────────────┘     └──────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Kiểm định      │ <-- │ Trích đặc    │ <-- │  Ảnh đã chuẩn hóa│
│  thống kê (04)  │     │ trưng (03)   │     │                  │
└─────────────────┘     └──────────────┘     └──────────────────┘
        │
        ▼ 
┌───────────────────────────────────────────────────────────────┐
│         05_cv_pipeline.py — k-fold CV ĐÚNG CHUẨN              │
│  Mỗi fold: feature selection CHỈ trên training fold           │
│  -> tránh data leakage -> kết quả đánh giá đáng tin cậy       │
└───────────────────────────────────────────────────────────────┘
                                                        │
                                                        ▼
                                        ┌──────────────────────┐
                                        │ 06_train_final_model │
                                        │ (train trên toàn bộ  │
                                        │  CEDAR, lưu model)   │
                                        └──────────────────────┘
                                                        │
                                                        ▼
                                        ┌──────────────────────┐
                                        │        app.py        │
                                        │   Web demo tương tác │
                                        └──────────────────────┘
```

## 4. Cài đặt

```bash
# Tạo môi trường ảo
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Cài thư viện
pip install -r requirements.txt
```

**Yêu cầu:** Python ≥ 3.9

## 5. Chuẩn bị dữ liệu

### CEDAR (bắt buộc — dataset chính)

| Bước | Nội dung |
|---|---|
| 1 | Tải tại [kaggle.com/datasets/shreelakshmigp/cedardataset](https://www.kaggle.com/datasets/shreelakshmigp/cedardataset) (cần tài khoản Kaggle, miễn phí) |
| 2 | Giải nén |
| 3 | Copy 2 thư mục `full_org` và `full_forg` vào `data/raw/cedar/` (giữ nguyên tên file bên trong) |

### ICDAR SigComp 2011 (tùy chọn — dùng cho external test sau này)

Bản dùng được trên Kaggle: `robinreni/signature-verification-dataset`. Đặt vào `data/raw/icdar/`. Script xử lý riêng sẽ được bổ sung khi làm tới giai đoạn này.

## 6. Cách chạy

Chạy tuần tự từ thư mục gốc project:

```bash
python src/01_preprocessing.py   # -> data/processed/{ảnh sạch, labels.csv}
python src/02_eda.py             # -> results/figures/{boxplot, violin plot}
python src/03_features.py        # -> data/processed/features.csv
python src/04_stats_tests.py     # -> results/tables/ (chỉ tham khảo)
python src/05_cv_pipeline.py     # -> results/tables/ (kết quả CHÍNH THỨC để báo cáo)
```

Sau khi có kết quả CV ổn, làm tiếp phần demo:

```bash
python src/06_train_final_model.py   # -> model_artifacts/
streamlit run app.py                  # -> http://localhost:8501
```

## 7. Phương pháp luận

| Bước | Kỹ thuật | Vai trò |
|---|---|---|
| Tiền xử lý | Otsu binarization, crop bounding box, resize 256×256 | Loại nhiễu nền, chuẩn hóa vị trí & kích thước |
| EDA | Boxplot, violin plot, phát hiện outlier (IQR) | Hiểu dữ liệu trước khi mô hình hóa |
| Feature engineering | Hu Moments (7 đặc trưng hình học), GLCM (texture), tỷ lệ baseline | Biểu diễn chữ ký bằng số liệu có ý nghĩa |
| Kiểm định thống kê | Mann-Whitney U test (p < 0.05) | Xác định đặc trưng nào thực sự phân biệt được thật/giả |
| Huấn luyện & đánh giá | SVM / Random Forest, GroupKFold 10-fold CV | Đánh giá khách quan, không rò rỉ dữ liệu |

> **Điểm quan trọng (theo góp ý của GVHD):** Feature selection bằng Mann-Whitney U **chỉ được chạy trên training fold** trong mỗi vòng CV — nếu chạy trên toàn bộ dữ liệu trước khi chia fold sẽ gây **data leakage**, khiến kết quả đánh giá bị thổi phồng, không phản ánh đúng khả năng tổng quát hóa thật của mô hình. File `04_stats_tests.py` chỉ mang tính khám phá ban đầu; kết quả đánh giá chính thức nằm ở `05_cv_pipeline.py`. Việc chia fold cũng dùng `GroupKFold` theo `writer_id` để tránh chữ ký cùng một người xuất hiện ở cả training và test.

## 8. Web app demo

## 9. Vai trò của ICDAR SigComp 2011

ICDAR **không** tham gia training hay feature selection cùng CEDAR. Dự kiến chỉ dùng làm tập test độc lập sau khi đã có model cuối, để trả lời: *"Mô hình học từ CEDAR (Mỹ) có tổng quát tốt sang chữ ký nguồn khác (Hà Lan/Trung Quốc) không?"*

- Hiệu năng trên ICDAR gần với CV trên CEDAR → mô hình tổng quát tốt
- Giảm nhiều → mô hình overfit vào đặc thù CEDAR, cần nêu rõ trong phần "Hạn chế của đề tài"
