from __future__ import annotations

import csv
import json
import random
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from water_quality_ann.ann import SimpleANN
from water_quality_ann.data import FEATURES, FEATURE_LABELS, LABELS
from water_quality_ann.preprocessing import StandardScaler, argmax


# Màu sắc tương ứng cho từng lớp chất lượng nước
CLASS_COLORS = {
    "Tốt": "#15803d",          # Xanh lá
    "Trung bình": "#d6a300",   # Vàng
    "Kém": "#ea580c",          # Cam
    "Nguy hiểm": "#dc2626",    # Đỏ
}

# Giá trị mặc định (mẫu nước tốt) khi khởi tạo form
DEFAULT_VALUES = {
    "pH": "7.20",
    "Hardness": "150",
    "Solids": "320",
    "Chlorine": "1.20",
    "Sulfate": "120",
    "Lead": "0.003",
    "Mercury": "0.0008",
    "Coliform": "8",
}


def load_scaler(path: Path) -> StandardScaler:
    # Đọc tham số StandardScaler từ file JSON
    with path.open("r", encoding="utf-8") as file:
        return StandardScaler.from_dict(json.load(file))


class WaterQualityApp(tk.Tk):
    """Ứng dụng GUI Tkinter để dự đoán chất lượng nước bằng mô hình ANN."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Phan loai chat luong nuoc - ANN")
        self.geometry("900x650")
        self.minsize(820, 580)
        self.configure(bg="#e8f1ee")

        # Đường dẫn đến model, scaler và dữ liệu
        self.model_path = ROOT / "models" / "best_model.json"
        self.scaler_path = ROOT / "models" / "scaler.json"
        self.dataset_path = ROOT / "data" / "water_quality.csv"

        # Các biến lưu trạng thái
        self.model: SimpleANN | None = None        # Mô hình ANN
        self.scaler: StandardScaler | None = None   # Bộ chuẩn hóa
        self.entries: dict[str, tk.Entry] = {}      # Ô nhập liệu cho 8 chỉ số
        self.probability_bars: dict[str, ttk.Progressbar] = {}   # Thanh xác suất
        self.probability_labels: dict[str, tk.Label] = {}        # Nhãn phần trăm
        self.random_samples: list[dict[str, str]] = []           # Dữ liệu CSV để random
        self.random_mode = tk.BooleanVar(value=True)             # Chế độ random
        self.sample_label: tk.Label | None = None   # Nhãn thông tin mẫu đang dùng

        self._setup_style()
        self._load_assets()
        self._build_ui()

    def _setup_style(self) -> None:
        # Cấu hình giao diện ttk (Progressbar)
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TProgressbar", thickness=16, troughcolor="#dbe7e3", background="#0f766e")

    def _load_assets(self) -> None:
        # Tải model, scaler và dữ liệu CSV (nếu có)
        if self.model_path.exists() and self.scaler_path.exists():
            self.model = SimpleANN.load(self.model_path)
            self.scaler = load_scaler(self.scaler_path)
        if self.dataset_path.exists():
            with self.dataset_path.open("r", newline="", encoding="utf-8") as file:
                self.random_samples = list(csv.DictReader(file))

    def _build_ui(self) -> None:
        # Xây dựng toàn bộ giao diện người dùng
        # ----- Header -----
        header = tk.Frame(self, bg="#0f766e", padx=28, pady=20)
        header.pack(fill="x")
        tk.Label(
            header,
            text="ANN phân loại chất lượng nước sinh hoạt",
            bg="#0f766e",
            fg="#fffdf7",
            font=("Georgia", 23, "bold"),
            anchor="w",
        ).pack(fill="x")
        tk.Label(
            header,
            text="Softmax 4 lớp: Tốt, Trung bình, Kém, Nguy hiểm",
            bg="#0f766e",
            fg="#d8efe8",
            font=("Segoe UI", 11),
            anchor="w",
        ).pack(fill="x", pady=(5, 0))

        # ----- Body: panel trái (nhập liệu) + panel phải (kết quả) -----
        body = tk.Frame(self, bg="#e8f1ee", padx=24, pady=22)
        body.pack(fill="both", expand=True)

        left = tk.Frame(body, bg="#fffdf7", bd=1, relief="solid", padx=22, pady=18)
        left.pack(side="left", fill="both", expand=True, padx=(0, 14))

        right = tk.Frame(body, bg="#fffdf7", bd=1, relief="solid", padx=22, pady=18)
        right.pack(side="right", fill="both", expand=True)

        # ----- Panel trái: form nhập 8 chỉ số -----
        tk.Label(left, text="Chỉ số mẫu nước", bg="#fffdf7", fg="#0f172a", font=("Georgia", 16, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 14)
        )

        for row_index, feature in enumerate(FEATURES, start=1):
            tk.Label(
                left,
                text=FEATURE_LABELS[feature],
                bg="#fffdf7",
                fg="#334155",
                font=("Segoe UI", 10),
                anchor="w",
            ).grid(row=row_index, column=0, sticky="ew", pady=5)
            entry = tk.Entry(left, font=("Segoe UI", 11), relief="solid", bd=1)
            entry.insert(0, DEFAULT_VALUES[feature])
            entry.grid(row=row_index, column=1, sticky="ew", padx=(14, 0), ipady=5, pady=5)
            self.entries[feature] = entry

        left.columnconfigure(0, weight=1)
        left.columnconfigure(1, weight=1)

        # Nút bấm
        button_row = tk.Frame(left, bg="#fffdf7")
        button_row.grid(row=len(FEATURES) + 1, column=0, columnspan=2, sticky="ew", pady=(18, 0))
        tk.Button(
            button_row,
            text="Dự đoán",
            command=self.predict,
            bg="#0f766e",
            activebackground="#115e59",
            fg="white",
            activeforeground="white",
            font=("Segoe UI", 11, "bold"),
            relief="flat",
            padx=18,
            pady=10,
        ).pack(side="left")
        tk.Button(
            button_row,
            text="Mẫu nguy hiểm",
            command=self.load_danger_sample,
            bg="#334155",
            activebackground="#1e293b",
            fg="white",
            activeforeground="white",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            padx=16,
            pady=10,
        ).pack(side="left", padx=10)

        # Checkbox random mẫu từ CSV
        tk.Checkbutton(
            left,
            text="Random từ CSV khi bấm Dự đoán",
            variable=self.random_mode,
            bg="#fffdf7",
            activebackground="#fffdf7",
            fg="#334155",
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        ).grid(row=len(FEATURES) + 2, column=0, columnspan=2, sticky="w", pady=(14, 0))

        # ----- Panel phải: hiển thị kết quả -----
        tk.Label(right, text="Kết quả", bg="#fffdf7", fg="#0f172a", font=("Georgia", 16, "bold")).pack(anchor="w")
        # Nhãn kết quả chính (màu nền thay đổi theo lớp)
        self.result_label = tk.Label(
            right,
            text="Chưa dự đoán",
            bg="#64748b",
            fg="white",
            font=("Georgia", 25, "bold"),
            padx=18,
            pady=24,
            width=17,
        )
        self.result_label.pack(fill="x", pady=(18, 22))

        # Nhãn thông tin mẫu
        self.sample_label = tk.Label(
            right,
            text="Random đang bật: bấm Dự đoán để lấy một mẫu từ CSV",
            bg="#fffdf7",
            fg="#475569",
            font=("Segoe UI", 10),
            anchor="w",
            wraplength=340,
        )
        self.sample_label.pack(fill="x", pady=(0, 14))

        # 4 thanh xác suất cho 4 lớp
        for label in LABELS:
            line = tk.Frame(right, bg="#fffdf7")
            line.pack(fill="x", pady=7)
            tk.Label(line, text=label, bg="#fffdf7", fg="#334155", font=("Segoe UI", 10, "bold"), width=12, anchor="w").pack(
                side="left"
            )
            bar = ttk.Progressbar(line, orient="horizontal", maximum=100, value=0)
            bar.pack(side="left", fill="x", expand=True, padx=10)
            pct = tk.Label(line, text="0.00%", bg="#fffdf7", fg="#334155", font=("Segoe UI", 10), width=8, anchor="e")
            pct.pack(side="right")
            self.probability_bars[label] = bar
            self.probability_labels[label] = pct

        # Trạng thái ứng dụng
        if self.model and self.scaler:
            status_text = f"Model đã sẵn sàng | CSV random: {len(self.random_samples)} mẫu"
        else:
            status_text = "Chưa có model, hãy chạy python train.py"
        status_color = "#0f766e" if self.model and self.scaler else "#b45309"
        self.status = tk.Label(
            right,
            text=status_text,
            bg="#fffdf7",
            fg=status_color,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        )
        self.status.pack(fill="x", pady=(24, 0))

    def load_danger_sample(self) -> None:
        # Nạp mẫu nước nguy hiểm (các chỉ số vượt ngưỡng cao)
        self.random_mode.set(False)
        values = {
            "pH": "10.9",
            "Hardness": "760",
            "Solids": "2450",
            "Chlorine": "6.2",
            "Sulfate": "1080",
            "Lead": "0.08",
            "Mercury": "0.018",
            "Coliform": "1900",
        }
        for feature, value in values.items():
            self.entries[feature].delete(0, tk.END)
            self.entries[feature].insert(0, value)
        if self.sample_label:
            self.sample_label.configure(text="Đã nạp mẫu nguy hiểm. Random tạm tắt để bạn dự đoán mẫu này.")

    def load_random_sample(self) -> None:
        # Lấy ngẫu nhiên một mẫu từ file CSV và điền vào form
        if not self.random_samples:
            raise ValueError("Chưa có dữ liệu CSV để random. Hãy kiểm tra file data/water_quality.csv.")

        sample = random.choice(self.random_samples)
        for feature in FEATURES:
            self.entries[feature].delete(0, tk.END)
            self.entries[feature].insert(0, sample[feature])

        quality = sample.get("Quality", "không rõ")
        if self.sample_label:
            self.sample_label.configure(text=f"Mẫu random từ CSV | Nhãn dữ liệu: {quality}")

    def _read_values(self) -> list[float]:
        # Đọc giá trị 8 chỉ số từ form nhập liệu
        values = []
        for feature in FEATURES:
            raw_value = self.entries[feature].get().strip().replace(",", ".")
            try:
                values.append(float(raw_value))
            except ValueError as error:
                raise ValueError(f"Giá trị không hợp lệ: {FEATURE_LABELS[feature]}") from error
        return values

    def predict(self) -> None:
        """Xử lý sự kiện bấm nút Dự đoán:
        1. Random mẫu từ CSV (nếu bật chế độ)
        2. Đọc giá trị từ form
        3. Chuẩn hóa -> dự đoán -> hiển thị kết quả
        """
        if not self.model or not self.scaler:
            messagebox.showerror("Thiếu model", "Hãy chạy lệnh: python train.py")
            return

        try:
            if self.random_mode.get():
                self.load_random_sample()
            values = self._read_values()
        except ValueError as error:
            messagebox.showerror("Dữ liệu không hợp lệ", str(error))
            return

        # Dự đoán: chuẩn hóa dữ liệu -> forward pass -> lấy xác suất
        probabilities = self.model.predict_proba(self.scaler.transform_row(values))
        label = LABELS[argmax(probabilities)]
        color = CLASS_COLORS[label]
        self.result_label.configure(text=label, bg=color)

        # Cập nhật thanh xác suất cho từng lớp
        for label_name, probability in zip(LABELS, probabilities):
            value = probability * 100
            self.probability_bars[label_name].configure(value=value)
            self.probability_labels[label_name].configure(text=f"{value:.2f}%")


if __name__ == "__main__":
    app = WaterQualityApp()
    app.mainloop()
