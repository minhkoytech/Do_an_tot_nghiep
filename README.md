# Do_an_tot_nghiep
Đồ án: Xác thực chữ ký viết tay (Offline Signature Verification)

Ngành Khoa học Dữ liệu — Đồ án tốt nghiệp

1. Cấu trúc thư mục
signature_project/
├── data/
│   ├── raw/                     ← đặt dataset gốc vào đây (xem mục 2)
│   └── processed/                ← ảnh sau tiền xử lý + file csv (tự sinh ra)
├── notebooks/                    ← chỗ thử nghiệm nhanh (Jupyter), không bắt buộc
├── results/
│   ├── figures/                  ← biểu đồ EDA (tự sinh ra)
│   └── tables/                   ← bảng kết quả CV, kiểm định thống kê (tự sinh ra)
├── src/
│   ├── 01_preprocessing.py       ← tiền xử lý ảnh
│   ├── 02_eda.py                  ← phân tích khám phá dữ liệu
│   ├── 03_features.py             ← trích đặc trưng thủ công
│   ├── 04_stats_tests.py          ← kiểm định thống kê (tổng quan, tham khảo)
│   ├── 05_cv_pipeline.py          ← huấn luyện + đánh giá bằng k-fold CV chuẩn
│   └── 06_train_final_model.py    ← train model cuối để dùng cho web demo
├── model_artifacts/               ← model đã train, để web app load (tự sinh ra)
├── app.py                         ← web demo (Streamlit)
├── requirements.txt
└── README.md
2. Cách lấy dữ liệu
CEDAR (dataset chính, dùng để train + đánh giá qua CV)
Tải tại kaggle.com/datasets/shreelakshmigp/cedardataset (cần tài khoản Kaggle, miễn phí)
Giải nén, đặt 2 thư mục full_org và full_forg vào:
   data/raw/cedar/full_org/
   data/raw/cedar/full_forg/

(giữ nguyên tên file bên trong, không cần đổi)

ICDAR SigComp 2011 (dùng làm external test sau này, chưa cần ngay)

Bản dùng được trên Kaggle: robinreni/signature-verification-dataset. Đặt vào data/raw/icdar/. Chưa có script xử lý riêng cho ICDAR — sẽ bổ sung khi làm tới bước external test.

3. Thứ tự chạy
bash
python src/01_preprocessing.py   # -> data/processed/ (ảnh sạch + labels.csv)
python src/02_eda.py             # -> results/figures/ (boxplot, violin plot)
python src/03_features.py        # -> data/processed/features.csv
python src/04_stats_tests.py     # -> results/tables/ (chỉ tham khảo, xem lưu ý mục 4)
python src/05_cv_pipeline.py     # -> results/tables/ (kết quả CV chính thức để báo cáo)

Sau khi có kết quả CV ổn ở bước 05, làm tiếp phần web demo:

bash
python src/06_train_final_model.py   # -> model_artifacts/ (model.joblib, scaler.joblib...)
streamlit run app.py                  # chạy web demo local tại localhost:8501
4. Lưu ý quan trọng về phương pháp (theo góp ý của GVHD)

04_stats_tests.py chạy Mann-Whitney U trên toàn bộ dữ liệu, chỉ mang tính khám phá/báo cáo mô tả ban đầu — không dùng kết quả này để chọn feature đưa vào mô hình.

05_cv_pipeline.py mới là nơi feature selection được làm đúng: ở mỗi fold trong k-fold CV, Mann-Whitney U chỉ chạy trên training fold, hoàn toàn không đụng tới test fold. Đây là cách tránh data leakage — nếu chọn feature trên toàn bộ dữ liệu rồi mới chia fold, kết quả đánh giá sẽ bị thổi phồng, không phản ánh đúng khả năng tổng quát hóa của mô hình.

Ngoài ra, việc chia fold dùng GroupKFold theo writer_id, để chữ ký của cùng một người không vừa nằm ở training vừa ở test — tránh thêm một dạng leakage khác.

5. Web app demo

Chạy local:

bash
pip install -r requirements.txt
streamlit run app.py

Deploy public (miễn phí qua Streamlit Community Cloud):

Đẩy code (app.py, src/, model_artifacts/) lên GitHub — không đẩy data/raw hoặc data/processed vì dung lượng lớn và không cần thiết
Vào share.streamlit.io, đăng nhập GitHub, chọn repo và file app.py, nhấn Deploy
Nhận link dạng https://<tên-app>.streamlit.app để chia sẻ cho GVHD/hội đồng
6. Vai trò của ICDAR SigComp 2011 (bước sau này)

ICDAR không tham gia training hay feature selection cùng CEDAR. Dự kiến sẽ chỉ dùng làm tập test độc lập sau khi đã có model cuối từ CEDAR, để trả lời câu hỏi: "Mô hình học từ chữ ký CEDAR (Mỹ) có còn hoạt động tốt trên chữ ký nguồn khác (Hà Lan/Trung Quốc) không?"

Hiệu năng trên ICDAR gần với hiệu năng CV trên CEDAR → mô hình tổng quát tốt
Giảm nhiều → mô hình overfit vào đặc thù CEDAR, cần nêu rõ trong phần "Hạn chế của đề tài"

Script xử lý riêng cho bước này (07_external_test.py) sẽ được viết khi làm tới giai đoạn này.