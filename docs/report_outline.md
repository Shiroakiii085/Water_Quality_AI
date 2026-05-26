# Đề cương báo cáo — Phân loại chất lượng nước bằng ANN

## 1. Giới thiệu

**1.1. Đặt vấn đề**
- Nước sạch là tài nguyên thiết yếu; ô nhiễm nước ảnh hưởng nghiêm trọng đến sức khỏe.
- Đánh giá chất lượng nước dựa trên nhiều chỉ số hóa học, vi sinh — làm thủ công mất thời gian và thiếu nhất quán.
- Mạng nơ-ron nhân tạo (ANN) có khả năng học quan hệ phi tuyến phức tạp, phù hợp tự động phân loại.

**1.2. Mục tiêu**
- Xây dựng ANN phân loại nước sinh hoạt thành 4 mức: Tốt, Trung bình, Kém, Nguy hiểm.
- Cài đặt ANN hoàn toàn bằng Python thư viện chuẩn (không TensorFlow/PyTorch).
- So sánh ba kiến trúc ANN để chọn mô hình tốt nhất.
- Xây dựng giao diện đồ họa cho phép nhập chỉ số và dự đoán trực quan.

## 2. Mô tả bài toán

**2.1. Đầu vào — 8 chỉ số hóa học và vi sinh**

| Chỉ số | Mô tả | Đơn vị |
|--------|-------|--------|
| pH | Độ pH | — |
| Hardness | Độ cứng | mg/L |
| Solids | Chất rắn hòa tan tổng số | mg/L |
| Chlorine | Hàm lượng clo | mg/L |
| Sulfate | Hàm lượng sunfat | mg/L |
| Lead | Hàm lượng chì (Pb) | mg/L |
| Mercury | Hàm lượng thủy ngân (Hg) | mg/L |
| Coliform | Vi khuẩn Coliform | CFU/100mL |

**2.2. Đầu ra — 4 lớp chất lượng nước**
- **Tốt**: nước an toàn, không vượt ngưỡng ô nhiễm.
- **Trung bình**: một số chỉ số ở mức cảnh báo nhẹ.
- **Kém**: nhiều chỉ số vượt ngưỡng nghiêm trọng.
- **Nguy hiểm**: ít nhất một chỉ số ở mức nguy hại.

**2.3. Tính chất:** Phân loại đa lớp, 8 chiều đầu vào, 4 lớp đầu ra, mỗi mẫu thuộc đúng một lớp.

## 3. Dữ liệu

**3.1. Nguồn gốc:** Dữ liệu mô phỏng (synthetic) bằng Python — 10.000 mẫu, chia đều 2.500 mẫu/lớp.

**3.2. Cơ chế sinh dữ liệu:**
- Khởi tạo từ mẫu nền "tốt" với chỉ số an toàn.
- Với mỗi lớp, thay đổi ngẫu nhiên chỉ số tương ứng mức:
  - **Tốt**: giữ nguyên chỉ số an toàn.
  - **Trung bình**: chọn 2-3 chỉ số, gán giá trị "mild" (cảnh báo nhẹ).
  - **Kém**: chọn 3-4 chỉ số, gán giá trị "bad".
  - **Nguy hiểm**: chọn 1 chỉ số chính gán "danger", thêm 1-3 chỉ số ngẫu nhiên.
- Kiểm tra lại bằng luật gán nhãn để đảm bảo khớp lớp.

**3.3. Bộ luật gán nhãn** — tính điểm tích lũy (score) và cờ danger:

| Chỉ số | danger | +2 điểm | +1 điểm |
|--------|--------|---------|---------|
| pH | <5.0 hoặc >10.5 | <5.5 hoặc >9.5 | <6.5 hoặc >8.5 |
| Hardness | — | >500 | >300 |
| Solids | — | >1500 | >700 |
| Chlorine | >5.0 | — | <0.1 hoặc >3.0 |
| Sulfate | >1000 | >500 | >250 |
| Lead | >0.05 | >0.015 | >0.01 |
| Mercury | >0.01 | >0.004 | >0.002 |
| Coliform | >1000 | >200 | >50 |

Phân loại: danger hoặc score >= 10 → Nguy hiểm; >= 6 → Kém; >= 2 → Trung bình; còn lại → Tốt.

**3.4. Phân chia dữ liệu:** Stratified split 80/20 (8.000 train, 2.000 test), giữ tỷ lệ các lớp.

## 4. Tiền xử lý

- **StandardScaler** (tự cài đặt): `x' = (x - μ) / σ` — đưa các đặc trưng về cùng thang đo.
- **One-hot encoding**:
  - Tốt → [1, 0, 0, 0]
  - Trung bình → [0, 1, 0, 0]
  - Kém → [0, 0, 1, 0]
  - Nguy hiểm → [0, 0, 0, 1]

## 5. Mô hình ANN

**5.1. Kiến trúc:** Mạng truyền thẳng (feedforward), cài đặt hoàn toàn từ đầu (from scratch).

**5.2. Lan truyền tiến (Forward pass):**
- Mỗi layer: `z = W·a + b` → Hàm kích hoạt.
- Lớp ẩn: **ReLU** — `f(x) = max(0, x)`.
- Lớp đầu ra: **Softmax** — `p_i = e^(z_i) / Σ e^(z_j)` (phân phối xác suất 4 lớp).

**5.3. Hàm mất mát:** **Categorical Cross-Entropy** — `L = -Σ y_i · log(p_i)`.

**5.4. Lan truyền ngược (Backpropagation):**
- delta đầu ra: `p - y` (đạo hàm kết hợp Softmax + Cross-Entropy).
- delta lớp ẩn: `δ_h = (W^T·δ_next) · ReLU'(z)`.
- Cập nhật SGD: `w -= η · a_in · δ_out`, `b -= η · δ_out`.

**5.5. Khởi tạo trọng số:**
- Lớp ẩn: **He initialization** — `W ~ N(0, √(2/fan_in))`.
- Lớp đầu ra: **LeCun** — `W ~ N(0, √(1/fan_in))`.
- Bias: khởi tạo 0.

**5.6. Ba kiến trúc thử nghiệm:**

| Mô hình | Kiến trúc | Số tham số |
|---------|-----------|:----------:|
| ANN_1lop_32 | 8 → 32 → 4 | 420 |
| ANN_2lop_64_32 | 8 → 64 → 32 → 4 | 2.756 |
| ANN_2lop_128_64 | 8 → 128 → 64 → 4 | 9.636 |

## 6. Thực nghiệm

**6.1. Siêu tham số:**
- Số epoch: **100**
- Learning rate: 0.018
- Seed: 13 (đảm bảo tái lập kết quả)
- Batch size: 1 (SGD thuần túy)

**6.2. Quy trình:**
1. Sinh dữ liệu 10.000 mẫu → stratified split 80/20.
2. Chuẩn hóa bằng StandardScaler, one-hot encoding nhãn.
3. Huấn luyện 3 kiến trúc với cùng siêu tham số (seed khác nhau để độc lập).
4. Ghi nhận loss/accuracy trên train/test sau mỗi epoch.
5. Chọn best model theo test accuracy cao nhất.

## 7. Kết quả

**7.1. Bảng so sánh trên tập test (sau 100 epochs):**

| Mô hình | Kiến trúc | Test Loss | Test Accuracy |
|---------|-----------|:---------:|:------------:|
| ANN_1lop_32 | 8 → 32 → 4 | 0.0625 | **98.60%** |
| ANN_2lop_64_32 | 8 → 64 → 32 → 4 | 0.0157 | **99.75%** |
| ANN_2lop_128_64 | 8 → 128 → 64 → 4 | **0.0045** | **99.90%** |

**7.2. Nhận xét:**
- Cả 3 mô hình đều đạt accuracy rất cao (>98.5%) nhờ dữ liệu synthetic có luật rõ ràng.
- **ANN_2lop_128_64** đạt cao nhất (99.9%), nhưng chênh lệch không đáng kể so với ANN_1lop_32 (98.6%).
- So với 35 epoch trước (92.71%), 100 epoch cải thiện ~6-7%.
- ANN_1lop_32 có test loss cuối cùng (0.0625) cao hơn epoch 99 (0.0117) — dấu hiệu **overfitting nhẹ**.
- Mô hình được chọn làm best: **ANN_2lop_128_64** (test accuracy cao nhất).
- Đồ thị loss/accuracy theo epoch tham khảo file `outputs/training_report.html`.

## 8. Giao diện

**8.1. CLI (`predict_cli.py`):**
```bash
python predict_cli.py --pH 7.2 --Hardness 150 --Solids 320 --Chlorine 1.2 --Sulfate 120 --Lead 0.003 --Mercury 0.0008 --Coliform 8
```

**8.2. GUI Tkinter (`app.py`):**
- Form nhập 8 chỉ số bên trái, kết quả bên phải.
- Màu sắc trực quan:
  - Tốt → Xanh lá
  - Trung bình → Vàng
  - Kém → Cam
  - Nguy hiểm → Đỏ
- Thanh xác suất cho từng lớp kèm phần trăm.
- Tiện ích: random từ CSV, nút "Mẫu nguy hiểm".

## 9. Kết luận

**9.1. Kết quả đạt được:**
- Xây dựng thành công ANN từ đầu (không dùng framework) cho phân loại chất lượng nước.
- Mô hình tốt nhất đạt **99.9% accuracy** trên tập test với kiến trúc 8 → 128 → 64 → 4.
- Có GUI và CLI đầy đủ.

**9.2. Hạn chế:**
- Dữ liệu synthetic, chưa phản ánh dữ liệu thực tế.
- Chưa có kỹ thuật chống overfitting (dropout, early stopping, regularization).
- Chưa có kiểm định chéo (cross-validation).
- Ngưỡng chất lượng nước được đơn giản hóa cho mục đích học tập.

**9.3. Hướng phát triển:**
- Thu thập dữ liệu thực tế từ trạm quan trắc môi trường.
- Thêm regularization (L2, Dropout), early stopping.
- Áp dụng k-fold cross-validation.
- Mở rộng feature: độ đục, nitrat, nitrit, asen,...
- Thử nghiệm SVM, Random Forest, CNN.
- Phát triển web app (Streamlit/Flask).

## 10. Tài liệu tham khảo

1. LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521(7553), 436–444.
2. He, K., et al. (2015). Delving deep into rectifiers. In ICCV 2015.
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
4. QCVN 01-1:2018/BYT — Quy chuẩn chất lượng nước ăn uống.
5. Mã nguồn dự án: https://github.com/Shiroakiii085/Water_Quality_AI
