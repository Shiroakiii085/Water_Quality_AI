# Đề cương báo cáo

## 1. Giới thiệu

Đề tài xây dựng mạng nơ-ron nhân tạo để phân loại chất lượng nước sinh hoạt thành 4 mức: Tốt, Trung bình, Kém, Nguy hiểm.

## 2. Mô tả bài toán

Đây là bài toán phân loại đa lớp. Đầu vào gồm 8 chỉ số hóa học và vi sinh:

- pH
- Độ cứng
- Chất rắn hòa tan
- Hàm lượng clo
- Hàm lượng sunfat
- Chì
- Thủy ngân
- Coliform

Đầu ra gồm 4 neuron ứng với 4 lớp. Hàm kích hoạt đầu ra là Softmax.

## 3. Dữ liệu

Dữ liệu được mô phỏng bằng Python và gán nhãn theo ngưỡng. File `data/water_quality.csv` gồm 10.000 mẫu, chia đều 2.500 mẫu cho mỗi lớp. Mỗi mẫu có một nhãn duy nhất trong 4 lớp.

## 4. Tiền xử lý

- Tách biến đầu vào `X` và nhãn `y`.
- Chuẩn hóa `X` bằng StandardScaler.
- Mã hóa nhãn one-hot:
  - Tốt = [1, 0, 0, 0]
  - Trung bình = [0, 1, 0, 0]
  - Kém = [0, 0, 1, 0]
  - Nguy hiểm = [0, 0, 0, 1]

## 5. Mô hình ANN

Sử dụng mạng nơ-ron truyền thẳng nhiều lớp. Lớp ẩn dùng ReLU, lớp đầu ra dùng Softmax. Hàm mất mát là categorical cross-entropy.

## 6. Thực nghiệm

Ba kiến trúc được thử nghiệm:

| Mô hình | Kiến trúc |
|---|---|
| ANN 1 | 8 -> 32 -> 4 |
| ANN 2 | 8 -> 64 -> 32 -> 4 |
| ANN 3 | 8 -> 128 -> 64 -> 4 |

## 7. Kết quả

Trình bày bảng so sánh accuracy/loss trên tập test và chèn đồ thị từ file `outputs/training_report.html`.

## 8. Giao diện

Giao diện Tkinter cho phép nhập chỉ số nước và hiển thị kết quả bằng màu:

- Tốt: xanh
- Trung bình: vàng
- Kém: cam
- Nguy hiểm: đỏ

## 9. Kết luận

Mô hình ANN có thể học quan hệ phi tuyến giữa các chỉ số nước và mức chất lượng. Hướng phát triển tiếp theo là dùng dữ liệu thực tế, thêm kiểm định chéo và cải tiến giao diện.
