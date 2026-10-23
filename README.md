# Xác thực Chữ ký Viết tay dựa trên Kết hợp Đặc trưng Thống kê và Học sâu

**Khóa luận tốt nghiệp — Ngành Khoa học Dữ liệu**  
**Đại học Kinh tế TP. Hồ Chí Minh · Khoa Công nghệ Thông tin và Kinh doanh**

Xây dựng hệ thống xác thực chữ ký viết tay ngoại tuyến (offline), kết hợp đặc trưng thống kê thủ công với học sâu (Siamese Network), ứng dụng trong phòng chống gian lận tài chính – ngân hàng.

`Python` · `OpenCV` · `scikit-image` · `scikit-learn` · `PyTorch` · `FastAPI`

---

## Mục lục

- [1. Giới thiệu](#1-giới-thiệu)
- [2. Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
- [3. Sơ đồ pipeline](#3-sơ-đồ-pipeline)
- [4. Cài đặt](#4-cài-đặt)
- [5. Chuẩn bị dữ liệu](#5-chuẩn-bị-dữ-liệu)
- [6. Cách chạy](#6-cách-chạy)
- [7. Phương pháp luận](#7-phương-pháp-luận)
- [8. Kết quả chính](#8-kết-quả-chính)
- [9. Ứng dụng web minh họa](#9-ứng-dụng-web-minh-họa)
- [10. Vai trò của ICDAR SigComp 2011](#10-vai-trò-của-icdar-sigcomp-2011)
- [11. Hạn chế và hướng phát triển](#11-hạn-chế-và-hướng-phát-triển)

---

## 1. Giới thiệu

Chữ ký viết tay vẫn được dùng phổ biến trong các giao dịch ngân hàng (séc, hợp đồng tín dụng, ủy nhiệm chi). Việc kiểm tra bằng mắt phụ thuộc kinh nghiệm nhân viên và dễ sai sót với giả mạo có kỹ năng.

Đề tài xây dựng hệ thống **so sánh hai ảnh chữ ký** (chữ ký mẫu vs chữ ký cần kiểm tra) và kết luận **khớp / không khớp**, theo hướng **độc lập người ký** (writer-independent): người ký trong tập kiểm thử không xuất hiện khi huấn luyện.

| Hạng mục | Nội dung |
| --- | --- |
| **Bài toán** | Xác thực cặp chữ ký (pair-wise verification), không phải phân loại từng ảnh riêng lẻ |
| **Loại dữ liệu** | Ảnh chữ ký tĩnh (offline signature) |
| **Dataset chính** | [CEDAR Signature Database](https://www.kaggle.com/datasets/shreelakshmigp/cedardataset) — 55 người, 24 thật + 24 giả/người, 300 dpi |
| **Dataset kiểm thử độc lập** | ICDAR SigComp 2011 (phần chữ ký ngoại tuyến người Hà Lan) |
| **Phương pháp** | Đặc trưng thủ công (hình học, Hu Moments, GLCM) + kiểm định thống kê + Random Forest / SVM dạng cặp + Siamese Network (Triplet Loss) + mô hình kết hợp |
| **Đánh giá** | FAR, FRR, EER, ROC-AUC; chia theo người ký; lặp 5 lần chia; so sánh tại cùng mức FAR mục tiêu |

**Đóng góp chính**

- So sánh công bằng bốn mô hình (RF, SVM, Siamese, kết hợp) trên cùng cách chia người ký và cùng bộ cặp chữ ký.
- Cho thấy đặc trưng thủ công gần như không phân biệt được thật/giả ở mức từng ảnh, nhưng có ý nghĩa rõ khi xét **độ chênh lệch giữa hai chữ ký trong cặp**.
- So sánh mô hình tại **cùng mức FAR mục tiêu** (không chỉ ở ngưỡng EER riêng của từng mô hình).
- Kiểm thử tổng quát hóa trên ICDAR SigComp 2011 (giữ nguyên mô hình và ngưỡng từ CEDAR).
- Ứng dụng web FastAPI minh họa xác thực + Grad-CAM + dashboard tương tác.

---

## 2. Cấu trúc thư mục

```
signature_project/
│
├── data/
│   ├── raw/
│   │   ├── cedar/
│   │   │   ├── full_org/          # Chữ ký thật
│   │   │   └── full_forg/         # Chữ ký giả (skilled forgery)
│   │   └── icdar/                 # ICDAR SigComp 2011 (chỉ kiểm thử)
│   └── processed/                 # Ảnh sau tiền xử lý, labels, features, cặp chữ ký
│
├── results/
│   ├── figures/                   # EDA, ROC, Grad-CAM, biểu đồ FAR–FRR
│   └── tables/                    # Kết quả kiểm định, so sánh mô hình, theo người ký
│
├── src/
│   ├── preprocessing.py           # Otsu, khử nhiễu, crop, resize 220×155
│   ├── split_and_pairs.py         # Chia theo người ký + sinh cặp (thật–thật, thật–giả, thật–người khác)
│   ├── eda.py                     # Phân tích mức ảnh và mức cặp
│   ├── features.py                # 17 đặc trưng → chọn 11 đặc trưng
│   ├── stats_tests.py             # Mann-Whitney U, Kruskal-Wallis, Benjamini–Hochberg
│   ├── train_classical.py         # Random Forest & SVM dạng cặp
│   ├── train_siamese.py           # Siamese CNN + Triplet Loss + ensemble 3 mạng
│   ├── train_hybrid.py            # Kết hợp Δ đặc trưng thủ công (11) + Δ embedding (64)
│   ├── evaluate.py                # FAR/FRR/EER/ROC-AUC, theo loại giả mạo, theo người ký, 5 lần chia
│   ├── gradcam.py                 # Grad-CAM trên Siamese
│   └── external_test_icdar.py     # Kiểm thử độc lập trên ICDAR
│
├── model_artifacts/               # scaler, RF/SVM, 3 mạng Siamese, hybrid, ngưỡng
│
├── app/                           # Ứng dụng web
│   ├── main.py                    # FastAPI
│   ├── static/                    # CSS, JS
│   └── templates/                 # HTML (tab Xác thực, Dashboard, Giới thiệu)
│
├── requirements.txt
└── README.md
```

---

## 3. Sơ đồ pipeline

```
Ảnh chữ ký gốc (scan/chụp)
        │
        ▼
┌───────────────────┐
│  Tiền xử lý       │  Xám → kiểm tra nền → Otsu → khử nhiễu → crop → 220×155
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Chia theo người ký│  35 train / 10 val / 10 test  (độc lập người ký)
│ + sinh cặp        │  150 thật–thật + 150 thật–giả kỹ năng + 75 thật–người khác / người
└───────────────────┘
        │
        ├──────────────────┬──────────────────┐
        ▼                  ▼                  ▼
   EDA (mức ảnh       Trích 17 đặc trưng   Kiểm định thống kê
   & mức cặp)         (hình học, Hu,       (Mann-Whitney,
                       GLCM)                Kruskal-Wallis)
        │                  │                  │
        └────────┬─────────┴────────┬─────────┘
                 ▼                  ▼
        RF / SVM (Δ 11 đt)    Siamese Network
                 │            (Triplet Loss, ensemble 3)
                 │                  │
                 └────────┬─────────┘
                          ▼
                 Mô hình kết hợp
                 (Δ 11 đt + Δ 64 emb → RF)
                          │
                          ▼
              Chọn ngưỡng trên validation
              (EER / FAR mục tiêu / accuracy)
                          │
                          ▼
              Đánh giá trên test (1 lần) + 5 lần chia
              + Grad-CAM + ICDAR external test
                          │
                          ▼
                 Ứng dụng web FastAPI
```

**Nguyên tắc chống rò rỉ dữ liệu**

- Chia theo `writer_id`, mỗi người chỉ thuộc một tập.
- Cặp giả mạo ngẫu nhiên chỉ ghép trong cùng tập.
- Chọn đặc trưng, tham số, ngưỡng **chỉ trên train/validation**.
- Tập kiểm thử chỉ dùng **một lần** sau khi mọi thứ đã cố định.

---

## 4. Cài đặt

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Yêu cầu:** Python ≥ 3.9 · GPU khuyến nghị khi huấn luyện Siamese (có thể chạy CPU).

Các thư viện chính: `opencv-python`, `scikit-image`, `scikit-learn`, `torch`, `fastapi`, `uvicorn`, `pandas`, `numpy`, `scipy`, `matplotlib`.

---

## 5. Chuẩn bị dữ liệu

### CEDAR (bắt buộc)

| Bước | Nội dung |
| --- | --- |
| 1 | Tải [CEDAR Signature Database](https://www.kaggle.com/datasets/shreelakshmigp/cedardataset) (cần tài khoản Kaggle) |
| 2 | Giải nén |
| 3 | Đặt `full_org` và `full_forg` vào `data/raw/cedar/` (giữ nguyên tên file) |

- 55 người ký · 24 chữ ký thật + 24 chữ ký giả (skilled) / người · ảnh xám 300 dpi · tổng 2.640 ảnh.

### ICDAR SigComp 2011 (tùy chọn — chỉ external test)

- Phần chữ ký **ngoại tuyến người Hà Lan**: 21 người, ~500 ảnh, 400 dpi, ảnh màu PNG.
- Chỉ dùng thư mục kiểm thử (tránh trùng ảnh train/test trong bản tải về).
- Đặt vào `data/raw/icdar/`.
- **Không** tham gia huấn luyện hay chọn ngưỡng; giữ nguyên mô hình + ngưỡng từ CEDAR.

---

## 6. Cách chạy

Chạy tuần tự từ thư mục gốc project:

```bash
# 1. Tiền xử lý + chia người ký + sinh cặp
python src/preprocessing.py
python src/split_and_pairs.py

# 2. EDA + đặc trưng + kiểm định
python src/eda.py
python src/features.py
python src/stats_tests.py

# 3. Huấn luyện các mô hình
python src/train_classical.py      # RF & SVM dạng cặp
python src/train_siamese.py        # Siamese + Triplet Loss + ensemble
python src/train_hybrid.py         # Mô hình kết hợp

# 4. Đánh giá (1 lần chia + 5 lần chia, theo FAR mục tiêu, theo người ký)
python src/evaluate.py

# 5. Grad-CAM + external test
python src/gradcam.py
python src/external_test_icdar.py

# 6. Ứng dụng web
uvicorn app.main:app --reload      # → http://localhost:8000
```

---

## 7. Phương pháp luận

| Bước | Kỹ thuật | Vai trò |
| --- | --- | --- |
| **Tiền xử lý** | Chuyển xám, kiểm tra/đảo nền, Otsu, phép mở hình thái + lọc diện tích, crop khung bao, resize **220×155** (giữ tỷ lệ) | Loại màu mực, nền, nhiễu; chuẩn hóa kích thước |
| **Chia dữ liệu** | Theo người ký: 35 / 10 / 10; sinh cặp thật–thật, thật–giả kỹ năng, thật–người khác | Đánh giá **độc lập người ký** |
| **EDA** | Boxplot / violin ở mức ảnh và mức cặp; outlier theo IQR | Hiểu sự khác biệt thật vs giả |
| **Đặc trưng thủ công** | Hình học (tỷ lệ khung, mật độ, baseline, điểm giao cắt), Hu Moments (7), GLCM (6) → chọn **11** đặc trưng | Vector Δ = \|f(mẫu) − f(cần kiểm tra)\| làm đầu vào RF/SVM |
| **Kiểm định** | Mann-Whitney U, Kruskal-Wallis, hiệu chỉnh Benjamini–Hochberg; báo cáo effect size (r, ε²) | Chỉ giữ đặc trưng có ý nghĩa ở **mức cặp** |
| **Học máy truyền thống** | Random Forest, SVM (RBF) trên vector Δ 11 chiều | Baseline dạng cặp (cùng bài toán với Siamese) |
| **Siamese Network** | CNN 5 lớp (~3,1M tham số) → embedding 64 chiều; Triplet Loss (margin 1,0 / 1,5 cho giả kỹ năng); ensemble 3 mạng; augmentation ±8° / ±8 px | Học độ tương đồng trực tiếp từ ảnh |
| **Mô hình kết hợp** | Nối Δ 11 đặc trưng + Δ 64 embedding → Random Forest | Tận dụng cả hai nguồn thông tin |
| **Ngưỡng** | (1) EER trên validation — chuẩn so sánh; (2) FAR mục tiêu 5/10/15/20% — điểm vận hành; (3) tối ưu accuracy — dùng trong app | Kiểm soát rủi ro chấp nhận nhầm |
| **Giải thích** | Grad-CAM trên khoảng cách embedding của Siamese | Phân tích lỗi chấp nhận nhầm / từ chối nhầm |

> **Lưu ý quan trọng:** Bài toán được đặt dưới dạng **so sánh cặp**, không phải phân loại từng ảnh. Cùng bộ đặc trưng, accuracy RF tăng từ ~65% (phân loại ảnh) lên ~77,5% (dạng cặp). Feature selection và mọi lựa chọn siêu tham số/ngưỡng chỉ thực hiện trên train/validation.

---

## 8. Kết quả chính

Kết quả dưới đây là trên CEDAR, đánh giá độc lập người ký (10 người test chưa từng xuất hiện khi huấn luyện). Giá trị trung bình ± độ lệch chuẩn qua **5 cách chia người ký**.

| Mô hình | Accuracy (%) | ROC-AUC | EER (%) | FAR giả kỹ năng (%) | FAR giả ngẫu nhiên (%) |
| --- | --- | --- | --- | --- | --- |
| Random Forest | 77,8 ± 2,8 | 0,862 ± 0,030 | 21,7 ± 3,1 | 34,8 ± 5,9 | 4,1 ± 1,1 |
| SVM | 77,4 ± 2,4 | 0,857 ± 0,026 | 22,2 ± 2,6 | 36,8 ± 6,2 | 4,7 ± 1,0 |
| Siamese Network | 79,5 ± 1,3 | 0,882 ± 0,015 | 20,1 ± 1,5 | 26,8 ± 7,1 | 2,8 ± 2,4 |
| **Mô hình kết hợp** | **79,4 ± 2,0** | **0,878 ± 0,025** | **20,2 ± 2,1** | **28,2 ± 6,1** | **3,0 ± 2,5** |

**Tại cùng mức FAR mục tiêu 10%** (ngưỡng chọn trên validation): mô hình kết hợp có FRR trung bình ~30%, thấp hơn rõ so với RF/SVM (~38%). Giả mạo có kỹ năng vẫn là điểm nghẽn chung (FAR ~27–37%); giả mạo ngẫu nhiên đã được kiểm soát tốt (FAR < 5%).

Trên **ICDAR SigComp 2011** (giữ nguyên mô hình + ngưỡng CEDAR): accuracy các mô hình đạt khoảng **74,6–76,4%** (giảm 1–4,5 điểm so với CEDAR), cho thấy khả năng tổng quát hóa chấp nhận được nhưng cần hiệu chỉnh ngưỡng khi đổi nguồn dữ liệu.

---

## 9. Ứng dụng web minh họa

Kiến trúc: **FastAPI** (máy chủ) + HTML/CSS/JavaScript (giao diện). Ba tab:

| Tab | Chức năng |
| --- | --- |
| **Xác thực** | Tải 2 ảnh → tiền xử lý → điểm số mô hình kết hợp → kết luận khớp/không khớp + độ tương đồng + ảnh sau tiền xử lý + Grad-CAM (tùy chọn) |
| **Dashboard** | So sánh 4 mô hình tại ngưỡng EER; thanh trượt FAR mục tiêu; FAR theo loại giả mạo; hiệu năng theo từng người ký; so sánh CEDAR vs ICDAR |
| **Giới thiệu** | Tóm tắt phương pháp và bảng số liệu |

```bash
uvicorn app.main:app --reload
# Mở http://localhost:8000
```

Mô hình ra quyết định trong app: **mô hình kết hợp**, ngưỡng theo FAR mục tiêu 10% (chọn trên validation). Ảnh tải lên chỉ lưu tạm trong lúc xử lý rồi xóa.

---

## 10. Vai trò của ICDAR SigComp 2011

ICDAR **không** tham gia training, feature selection hay chọn ngưỡng. Chỉ dùng làm tập **kiểm thử độc lập** sau khi mô hình và ngưỡng đã cố định trên CEDAR, nhằm trả lời:

> *Mô hình học từ CEDAR (Mỹ, 300 dpi, ảnh xám) có tổng quát sang chữ ký nguồn khác (Hà Lan, 400 dpi, ảnh màu) không?*

- Accuracy giảm nhẹ (1–4,5 điểm) → mô hình học được đặc điểm chung của chữ ký, không chỉ thuộc CEDAR.
- Phân bố điểm số dịch chuyển → ngưỡng tối ưu trên CEDAR không còn tối ưu trên ICDAR; khi triển khai thực tế cần hiệu chỉnh ngưỡng (và nếu có thể tinh chỉnh mô hình) bằng dữ liệu tại chỗ.

---

## 11. Hạn chế và hướng phát triển

**Hạn chế chính**

- Chỉ 35 người ký để huấn luyện → giới hạn ROC-AUC quanh 0,88–0,90; giả mạo có kỹ năng vẫn khó (FAR ~30%).
- So sánh với **một** chữ ký mẫu mỗi lần (nhiều nghiên cứu dùng 5–12 mẫu rồi lấy trung bình).
- Ảnh huấn luyện/đánh giá là ảnh quét sạch; chưa xử lý ảnh chụp điện thoại, dòng kẻ, con dấu.
- Ứng dụng chạy local, chưa có CSDL khách hàng / đăng nhập / nhật ký.

**Hướng ưu tiên**

1. Dùng nhiều chữ ký mẫu (3–5) rồi lấy trung bình điểm số.
2. Huấn luyện trước trên bộ dữ liệu lớn (ví dụ GPDS) rồi tinh chỉnh trên CEDAR.
3. Ngưỡng / xác minh bổ sung theo từng khách hàng (chữ ký dễ bắt chước).
4. Mở rộng tiền xử lý cho ảnh thực tế ngân hàng và triển khai máy chủ đầy đủ.

---

## Tài liệu tham khảo (rút gọn)

Xem đầy đủ trong khóa luận. Một số nguồn chính:

- CEDAR Signature Database (University at Buffalo)
- ICDAR SigComp 2011
- Bromley et al. (1993) — Siamese networks for signature verification
- Hafemann et al. (2017) — Deep features for offline signature verification
- Dey et al. (2017) — SigNet
- Selvaraju et al. (2017) — Grad-CAM

---

**Giảng viên hướng dẫn:** TS. Nguyễn Đăng Cao  
**Sinh viên:** Nông Minh Khôi — MSSV 31231025344 — Khoa học dữ liệu K49  
**TP. Hồ Chí Minh — 2026**
