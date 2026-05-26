# BÁO CÁO BÀI TẬP LỚN MÔN TRÍ TUỆ NHÂN TẠO

**Đề tài:** Phân loại chất lượng nước sinh hoạt bằng mạng nơ-ron nhân tạo (ANN)

---

## 1. Giới thiệu

### 1.1. Đặt vấn đề

Nước sạch là tài nguyên thiết yếu đối với đời sống con người. Ô nhiễm nguồn nước đang là vấn đề nghiêm trọng, ảnh hưởng trực tiếp đến sức khỏe cộng đồng. Việc đánh giá chất lượng nước dựa trên nhiều chỉ số hóa học và vi sinh khác nhau. Nếu thực hiện thủ công, quá trình này tốn thời gian và thiếu nhất quán.

Trí tuệ nhân tạo, đặc biệt là mạng nơ-ron nhân tạo (Artificial Neural Network - ANN), có khả năng học các quan hệ phi tuyến phức tạp từ dữ liệu. Do đó, ANN là công cụ phù hợp để tự động hóa việc phân loại chất lượng nước.

### 1.2. Mục tiêu

- Xây dựng mô hình ANN phân loại nước sinh hoạt thành 4 mức: Tốt, Trung bình, Kém, Nguy hiểm.
- Cài đặt ANN hoàn toàn bằng Python thư viện chuẩn (không sử dụng TensorFlow, PyTorch hay bất kỳ framework học sâu nào).
- So sánh ba kiến trúc ANN khác nhau để chọn mô hình tốt nhất.
- Xây dựng giao diện đồ họa cho phép nhập chỉ số nước và dự đoán trực quan.

### 1.3. Phạm vi

- Dữ liệu sử dụng là dữ liệu mô phỏng (synthetic), không phải dữ liệu thực tế.
- Mô hình học có giám sát (supervised learning), bài toán phân loại đa lớp (multiclass classification).

---

## 2. Cơ sở lý thuyết

### 2.1. Mạng nơ-ron nhân tạo (ANN)

Mạng nơ-ron nhân tạo là mô hình tính toán lấy cảm hứng từ cấu trúc và hoạt động của bộ não con người. Một mạng ANN bao gồm nhiều nơ-ron (neuron) được kết nối với nhau, thường được tổ chức thành các lớp (layer):

- **Input layer:** lớp đầu vào, tiếp nhận dữ liệu.
- **Hidden layer(s):** lớp ẩn, thực hiện các phép biến đổi phi tuyến.
- **Output layer:** lớp đầu ra, đưa ra kết quả dự đoán.

Mỗi nơ-ron thực hiện phép tính tổ hợp tuyến tính `z = W · x + b`, sau đó qua một hàm kích hoạt phi tuyến để tạo ra đầu ra.

### 2.2. Hàm kích hoạt

**ReLU (Rectified Linear Unit):**
```
f(x) = max(0, x)
```
ReLU là hàm kích hoạt phổ biến cho các lớp ẩn nhờ tính toán đơn giản, giúp giảm hiện tượng vanishing gradient và tăng tốc hội tụ.

**Softmax:**
```
p_i = e^(z_i) / Σ(j=1..K) e^(z_j)
```
Softmax chuyển một vector logits thành phân phối xác suất trên K lớp, thường được dùng ở đầu ra của bài toán phân loại đa lớp.

### 2.3. Hàm mất mát

**Categorical Cross-Entropy:**
```
L = - Σ(i=1..K) y_i · log(p_i)
```
Hàm mất mát này đo độ khác biệt giữa phân phối xác suất dự đoán (p) và phân phối thực tế (y - one-hot). Trị số càng nhỏ, mô hình dự đoán càng chính xác.

### 2.4. Lan truyền ngược (Backpropagation)

Backpropagation là thuật toán tính gradient của hàm mất mát theo từng trọng số trong mạng, sử dụng quy tắc dây chuyền (chain rule). Các bước chính:

1. **Forward pass:** Tính đầu ra và hàm mất mát.
2. **Backward pass:** Tính gradient cho từng lớp từ output về input.
   - Đầu ra: đạo hàm kết hợp Softmax và Cross-Entropy cho `δ = p - y`.
   - Lớp ẩn: `δ_h = (W^T · δ_next) · ReLU'(z)` với `ReLU'(z) = 1 nếu z > 0, ngược lại = 0`.
3. **Cập nhật trọng số:** `w = w - η · δ · a_in` với `η` là learning rate.

### 2.5. Khởi tạo trọng số

Khởi tạo trọng số phù hợp giúp mô hình hội tụ nhanh và tránh vanishing/exploding gradient:

- **He initialization** (dùng cho lớp ẩn ReLU): `w ~ N(0, √(2 / fan_in))`.
- **LeCun initialization** (dùng cho lớp Softmax đầu ra): `w ~ N(0, √(1 / fan_in))`.
- Bias được khởi tạo bằng 0.

### 2.6. Chuẩn hóa dữ liệu (StandardScaler)

StandardScaler chuẩn hóa dữ liệu về dạng có trung bình 0 và độ lệch chuẩn 1:
```
x' = (x - μ) / σ
```
Việc này giúp các đặc trưng có cùng thang đo, tránh tình trạng một số đặc trưng có giá trị lớn lấn át các đặc trưng khác trong quá trình huấn luyện.

---

## 3. Mô tả bài toán

### 3.1. Đầu vào

Đầu vào gồm 8 chỉ số hóa học và vi sinh đặc trưng cho chất lượng nước sinh hoạt:

| STT | Chỉ số | Ký hiệu | Mô tả | Đơn vị |
|:---:|--------|---------|-------|:------:|
| 1 | pH | pH | Độ axit/kiềm của nước | — |
| 2 | Độ cứng | Hardness | Hàm lượng canxi và magie | mg/L |
| 3 | Chất rắn hòa tan | Solids | Tổng chất rắn hòa tan (TDS) | mg/L |
| 4 | Clo | Chlorine | Hàm lượng clo dư | mg/L |
| 5 | Sunfat | Sulfate | Hàm lượng ion sunfat | mg/L |
| 6 | Chì | Lead | Hàm lượng chì (Pb) | mg/L |
| 7 | Thủy ngân | Mercury | Hàm lượng thủy ngân (Hg) | mg/L |
| 8 | Coliform | Coliform | Mật độ vi khuẩn Coliform | CFU/100mL |

### 3.2. Đầu ra

Đầu ra là 4 lớp chất lượng nước:

| Lớp | Ý nghĩa | Mô tả |
|:---:|---------|-------|
| Tốt | An toàn | Các chỉ số đều trong ngưỡng cho phép |
| Trung bình | Cảnh báo | Một số chỉ số ở mức cảnh báo nhẹ |
| Kém | Ô nhiễm | Nhiều chỉ số vượt ngưỡng nghiêm trọng |
| Nguy hiểm | Nguy hại | Có ít nhất một chỉ số ở mức nguy hiểm |

### 3.3. Tính chất bài toán

- **Loại bài toán:** Phân loại đa lớp (multiclass classification).
- **Số chiều đầu vào:** 8 (cố định).
- **Số lớp đầu ra:** 4 (cố định).
- **Tính chất:** Mỗi mẫu thuộc về đúng một lớp (mutually exclusive).

---

## 4. Dữ liệu

### 4.1. Nguồn gốc dữ liệu

Dữ liệu được sinh nhân tạo (synthetic) bằng ngôn ngữ Python, không phải dữ liệu thu thập từ thực tế. Tổng cộng 10.000 mẫu, chia đều 2.500 mẫu cho mỗi lớp.

### 4.2. Cơ chế sinh dữ liệu

Dữ liệu được sinh ra theo quy trình sau:

1. Khởi tạo từ mẫu nền "tốt" (base good) với các chỉ số trong ngưỡng an toàn.
2. Tùy theo lớp muốn sinh, một hoặc nhiều chỉ số được thay đổi với các mức:
   - **Mức mild:** Giá trị cảnh báo nhẹ, hơi vượt ngưỡng an toàn.
   - **Mức bad:** Giá trị xấu, vượt ngưỡng đáng kể.
   - **Mức danger:** Giá trị nguy hiểm, vượt ngưỡng nghiêm trọng.
3. Sau khi sinh, mẫu được kiểm tra lại bằng bộ luật gán nhãn để đảm bảo khớp với lớp mong muốn. Nếu không khớp, sinh lại (tối đa 200 lần thử).

**Chi tiết sinh cho từng lớp:**
- **Tốt:** Giữ nguyên các chỉ số an toàn.
- **Trung bình:** Chọn ngẫu nhiên 2-3 chỉ số, gán giá trị mild.
- **Kém:** Chọn ngẫu nhiên 3-4 chỉ số, gán giá trị bad; 35% khả năng thêm 1 chỉ số mild.
- **Nguy hiểm:** Chọn 1 chỉ số chính (pH, Chlorine, Sulfate, Lead, Mercury hoặc Coliform) gán giá trị danger, thêm 1-3 chỉ số khác gán ngẫu nhiên mild/bad/danger.

### 4.3. Bộ luật gán nhãn (assess_quality)

Hệ thống tính điểm tích lũy và kiểm tra điều kiện nguy hiểm dựa trên 8 chỉ số:

| Chỉ số | Điều kiện danger | +2 điểm | +1 điểm |
|--------|:----------------:|:-------:|:-------:|
| pH | < 5.0 hoặc > 10.5 | < 5.5 hoặc > 9.5 | < 6.5 hoặc > 8.5 |
| Hardness | — | > 500 | > 300 |
| Solids | — | > 1500 | > 700 |
| Chlorine | > 5.0 | — | < 0.1 hoặc > 3.0 |
| Sulfate | > 1000 | > 500 | > 250 |
| Lead | > 0.05 | > 0.015 | > 0.01 |
| Mercury | > 0.01 | > 0.004 | > 0.002 |
| Coliform | > 1000 | > 200 | > 50 |

**Quy tắc phân loại cuối cùng:**
- Nếu `danger = True` hoặc `score >= 10` → **Nguy hiểm**
- Nếu `score >= 6` → **Kém**
- Nếu `score >= 2` → **Trung bình**
- Còn lại → **Tốt**

### 4.4. Phân chia dữ liệu

Sử dụng phương pháp **stratified split** (phân chia phân tầng):
- **80%** dữ liệu cho huấn luyện (8.000 mẫu)
- **20%** dữ liệu cho kiểm tra (2.000 mẫu)
- Phương pháp này đảm bảo tỷ lệ các lớp được giữ nguyên trong cả hai tập.

---

## 5. Phương pháp

### 5.1. Tiền xử lý

**StandardScaler:**
- Tính trung bình (`μ`) và độ lệch chuẩn (`σ`) cho từng đặc trưng trên tập huấn luyện.
- Chuẩn hóa: `x' = (x - μ) / σ`.
- Áp dụng cùng tham số lên cả tập huấn luyện và tập kiểm tra.
- Scaler được implement từ đầu (không dùng sklearn).

**One-hot encoding:**
- Chuyển nhãn dạng chuỗi ("Tốt") thành vector nhị phân:
  - Tốt → [1, 0, 0, 0]
  - Trung bình → [0, 1, 0, 0]
  - Kém → [0, 0, 1, 0]
  - Nguy hiểm → [0, 0, 0, 1]

### 5.2. Kiến trúc mô hình (SimpleANN)

Mô hình ANN được cài đặt hoàn toàn từ đầu bằng Python, không sử dụng bất kỳ framework nào (TensorFlow, PyTorch, Keras,...).

**Cấu trúc lớp:** Lớp `SimpleANN` quản lý toàn bộ quá trình huấn luyện và dự đoán.

**Forward pass:** Mỗi layer thực hiện:
```
z = W · a_in + b
a_out = activation(z)
```
Với `W` là ma trận trọng số, `b` là bias, `a_in` là đầu vào của layer.

**Backward pass (lan truyền ngược):**
- Gradient đầu ra: `δ = p - y` (kết hợp đạo hàm của Softmax và Cross-Entropy).
- Gradient lớp ẩn: `δ_current = (W_next^T · δ_next) * mask`, với `mask = 1 nếu z > 0`.
- Cập nhật trọng số: `W -= η · a_in · δ`, `b -= η · δ`.

**Ba kiến trúc được thử nghiệm:**

| Mô hình | Kiến trúc | Số lượng tham số |
|---------|-----------|:----------------:|
| ANN_1lop_32 | Input(8) → Hidden(32) → Output(4) | 8×32 + 32×4 + 32 + 4 = **420** |
| ANN_2lop_64_32 | Input(8) → Hidden(64) → Hidden(32) → Output(4) | 8×64 + 64×32 + 32×4 + 64 + 32 + 4 = **2.756** |
| ANN_2lop_128_64 | Input(8) → Hidden(128) → Hidden(64) → Output(4) | 8×128 + 128×64 + 64×4 + 128 + 64 + 4 = **9.636** |

### 5.3. Siêu tham số huấn luyện

| Tham số | Giá trị | Ý nghĩa |
|---------|:-------:|---------|
| Số epoch | 100 | Số lần duyệt toàn bộ tập huấn luyện |
| Learning rate | 0.018 | Tốc độ học (hệ số cập nhật trọng số) |
| Seed | 13 | Đảm bảo kết quả tái lập được |
| Batch size | 1 | Stochastic Gradient Descent (mỗi mẫu một lần cập nhật) |

### 5.4. Phương pháp đánh giá

- **Accuracy (độ chính xác):** Tỷ lệ số mẫu dự đoán đúng trên tổng số mẫu trên tập kiểm tra.
- **Cross-entropy loss (giá trị mất mát):** Trung bình hàm mất mát trên tập kiểm tra.
- Cả hai chỉ số được theo dõi trên cả tập huấn luyện và tập kiểm tra qua từng epoch để phát hiện overfitting.

---

## 6. Thực nghiệm và kết quả

### 6.1. Quá trình huấn luyện

- Mỗi mô hình được huấn luyện trong 100 epoch, mỗi epoch duyệt qua 8.000 mẫu huấn luyện.
- Ghi nhận loss và accuracy trên cả tập huấn luyện và tập kiểm tra sau mỗi epoch.
- Kết quả được xuất ra file `outputs/training_metrics.csv` và trực quan hóa bằng đồ thị trong `outputs/training_report.html`.

### 6.2. Kết quả so sánh trên tập kiểm tra

| Mô hình | Kiến trúc | Test Loss | Test Accuracy |
|---------|-----------|:---------:|:-------------:|
| ANN_1lop_32 | 8 → 32 → 4 | 0.0625 | **98.60%** |
| ANN_2lop_64_32 | 8 → 64 → 32 → 4 | 0.0157 | **99.75%** |
| ANN_2lop_128_64 | 8 → 128 → 64 → 4 | **0.0045** | **99.90%** |

### 6.3. Nhận xét và phân tích

**Về độ chính xác:**
- Cả 3 mô hình đều đạt độ chính xác rất cao (trên 98.5%).
- Mô hình **ANN_2lop_128_64** (8→128→64→4) đạt kết quả tốt nhất với 99.9% accuracy và 0.0045 test loss.
- Mô hình **ANN_1lop_32** (8→32→4) đơn giản nhất với chỉ 420 tham số cũng đạt 98.6%.

**Về quá trình hội tụ:**
- Cả 3 mô hình đều hội tụ nhanh trong khoảng 20-30 epoch đầu.
- ANN_2lop_128_64 đạt train accuracy 100% sau khoảng epoch 27.
- ANN_2lop_64_32 đạt train accuracy 100% sau khoảng epoch 25.
- ANN_1lop_32 có độ dao động nhẹ, train accuracy không bao giờ đạt 100% hoàn hảo.

**Về overfitting:**
- Mô hình càng lớn càng có dấu hiệu overfitting rõ hơn: train accuracy 100% nhưng test accuracy dừng ở 99.9%.
- ANN_1lop_32 có độ dao động (oscillation) cao hơn ở các epoch cuối, test loss cuối (0.0625) cao hơn epoch 99 (0.0117).

**Về so sánh với kết quả 35 epoch:**
- Kết quả 35 epoch trước: 92.71% (ANN_1lop_32).
- Với 100 epoch, accuracy tăng lên đáng kể (~6-7%), cho thấy thời gian huấn luyện dài hơn giúp mô hình học tốt hơn.

### 6.4. Lựa chọn mô hình tốt nhất

Mô hình được chọn làm best model: **ANN_2lop_128_64** (8 → 128 → 64 → 4) với test accuracy cao nhất (99.9%).

### 6.5. Đồ thị

(Xem đồ thị loss và accuracy theo epoch trong file `outputs/training_report.html`)

Đồ thị gồm:
1. **Loss chart** cho từng mô hình: đường train loss và test loss theo epoch.
2. **Accuracy chart** cho từng mô hình: đường train accuracy và test accuracy theo epoch.
3. Các đường được vẽ bằng SVG, không sử dụng thư viện đồ họa bên ngoài.

---

## 7. Giao diện ứng dụng

### 7.1. Giao diện dòng lệnh (CLI)

File `predict_cli.py` cho phép dự đoán từ dòng lệnh:

```bash
python predict_cli.py --pH 7.2 --Hardness 150 --Solids 320 --Chlorine 1.2 --Sulfate 120 --Lead 0.003 --Mercury 0.0008 --Coliform 8
```

Kết quả trả về: lớp dự đoán và xác suất cho từng lớp.

Các chế độ đặc biệt:
- `predict_cli.py --random`: lấy mẫu ngẫu nhiên từ file CSV.
- `predict_cli.py --danger`: tạo mẫu nguy hiểm để kiểm tra.

### 7.2. Giao diện đồ họa (GUI)

File `app.py` xây dựng giao diện desktop bằng thư viện Tkinter:

**Bố cục giao diện:**
- **Bên trái:** Form nhập 8 chỉ số nước, mỗi chỉ số có nhãn và ô nhập liệu.
- **Bên phải:** Khu vực hiển thị kết quả.

**Hiển thị kết quả:**
- Nhãn phân loại lớn với màu sắc trực quan:
  - **Tốt** → Màu xanh lá cây
  - **Trung bình** → Màu vàng
  - **Kém** → Màu cam
  - **Nguy hiểm** → Màu đỏ
- 4 thanh tiến trình (progress bar) hiển thị xác suất cho từng lớp kèm phần trăm.

**Tiện ích:**
- Checkbox "Random từ CSV khi bấm Dự đoán": tự động lấy mẫu ngẫu nhiên từ file dữ liệu.
- Nút "Mẫu nguy hiểm": tạo một mẫu vượt ngưỡng để kiểm tra khả năng phát hiện của mô hình.

---

## 8. Kết luận

### 8.1. Kết quả đạt được

- Đã xây dựng thành công mô hình ANN hoàn toàn từ đầu (from scratch) bằng Python mà không sử dụng bất kỳ framework học sâu nào.
- Mô hình ANN_2lop_128_64 với kiến trúc 8 → 128 → 64 → 4 đạt độ chính xác **99.9%** trên tập kiểm tra.
- Đã so sánh ba kiến trúc khác nhau và chọn được mô hình tốt nhất.
- Có giao diện đồ họa trực quan (Tkinter) và giao diện dòng lệnh (CLI) đầy đủ.

### 8.2. Hạn chế

- Dữ liệu là synthetic, chưa phản ánh đầy đủ độ phức tạp và nhiễu của dữ liệu thực tế.
- Chưa áp dụng các kỹ thuật chống overfitting (dropout, early stopping, L1/L2 regularization).
- Chưa thực hiện kiểm định chéo (k-fold cross-validation) để đánh giá ổn định hơn.
- Các ngưỡng đánh giá chất lượng nước được đơn giản hóa cho mục đích học tập, không thay thế các quy chuẩn kiểm định nước thực tế (QCVN).

### 8.3. Hướng phát triển

- Thu thập dữ liệu nước thực tế từ các trạm quan trắc môi trường để tăng tính ứng dụng.
- Thêm các kỹ thuật regularization (L2, Dropout) và early stopping để giảm overfitting.
- Áp dụng k-fold cross-validation để đánh giá mô hình tin cậy hơn.
- Mở rộng tập đặc trưng: thêm độ đục (turbidity), nitrat (NO3-), nitrit (NO2-), asen (As),...
- Thử nghiệm các mô hình khác như SVM, Random Forest, hoặc CNN (nếu có dữ liệu chuỗi thời gian).
- Phát triển phiên bản web (Streamlit / Flask) để dễ dàng triển khai và sử dụng.

---

## 9. Tài liệu tham khảo

[1] LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436–444.

[2] He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification. In *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*.

[3] Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

[4] Bộ Y tế Việt Nam (2018). *QCVN 01-1:2018/BYT — Quy chuẩn kỹ thuật quốc gia về chất lượng nước ăn uống*.

[5] Nielsen, M. A. (2015). *Neural Networks and Deep Learning*. Determination Press.

[6] Mã nguồn dự án: https://github.com/Shiroakiii085/Water_Quality_AI
