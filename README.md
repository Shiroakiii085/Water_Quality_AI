# Phân loại chất lượng nước sinh hoạt bằng ANN

Project cho Đề tài 13: phân loại đa lớp 4 mức chất lượng nước sinh hoạt.

## Chức năng đã có

- Tạo dữ liệu mô phỏng 10.000 dòng và gán nhãn theo ngưỡng.
- One-hot encoding cho 4 lớp: `Tốt`, `Trung bình`, `Kém`, `Nguy hiểm`.
- Chuẩn hóa đầu vào bằng `StandardScaler` tự cài đặt.
- Huấn luyện 3 kiến trúc ANN:
  - `8 -> 32 -> 4`
  - `8 -> 64 -> 32 -> 4`
  - `8 -> 128 -> 64 -> 4`
- Lớp đầu ra Softmax, loss categorical cross-entropy.
- So sánh accuracy trên tập test.
- Xuất biểu đồ loss/accuracy ra file HTML.
- Giao diện Tkinter nhập chỉ số nước và hiển thị màu trực quan.

## Cách chạy

Mở terminal tại thư mục này:

```powershell
cd water_quality_ann
python train.py
python app.py
```

File kết quả:

- `data/water_quality.csv`: dữ liệu mô phỏng 10.000 dòng.
- `models/best_model.json`: model tốt nhất.
- `models/scaler.json`: tham số chuẩn hóa.
- `outputs/model_comparison.csv`: bảng so sánh 3 mô hình.
- `outputs/training_metrics.csv`: loss/accuracy theo epoch.
- `outputs/training_report.html`: báo cáo có đồ thị.

Giao diện mặc định bật chế độ random từ CSV. Mỗi lần bấm `Dự đoán`, chương trình sẽ lấy một mẫu ngẫu nhiên trong `data/water_quality.csv`, điền các chỉ số lên form và dự đoán. Bỏ chọn `Random từ CSV khi bấm Dự đoán` nếu muốn nhập tay.

## Dự đoán bằng command line

```powershell
python predict_cli.py --pH 7.2 --Hardness 150 --Solids 320 --Chlorine 1.2 --Sulfate 120 --Lead 0.003 --Mercury 0.0008 --Coliform 8
```

## Ghi chú báo cáo

Đây là dữ liệu mô phỏng dùng cho bài tập môn học. Các ngưỡng trong code được đơn giản hóa để minh họa bài toán học máy, không thay thế quy chuẩn kiểm định nước thực tế.
