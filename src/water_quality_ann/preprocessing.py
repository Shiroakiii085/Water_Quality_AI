from __future__ import annotations

import math
from dataclasses import dataclass

from .data import FEATURES, LABELS


def one_hot(label: str) -> list[float]:
    # Mã hóa one-hot: chuyển nhãn "Tốt" thành vector [1, 0, 0, 0]
    # Mỗi nhãn được biểu diễn bằng vector 4 phần tử, 1 ở vị trí tương ứng
    return [1.0 if item == label else 0.0 for item in LABELS]


def one_hot_many(labels: list[str]) -> list[list[float]]:
    # Mã hóa one-hot cho nhiều nhãn cùng lúc
    return [one_hot(label) for label in labels]


def argmax(values: list[float]) -> int:
    # Tìm chỉ số của phần tử có giá trị lớn nhất trong vector
    # Dùng để chọn lớp dự đoán từ xác suất đầu ra của Softmax
    best_index = 0
    best_value = values[0]
    for index, value in enumerate(values[1:], start=1):
        if value > best_value:
            best_index = index
            best_value = value
    return best_index


@dataclass
class StandardScaler:
    """Chuẩn hóa dữ liệu: x' = (x - mean) / std

    Giúp các đặc trưng có cùng thang đo (trung bình 0, độ lệch chuẩn 1),
    tránh tình trạng đặc trưng có giá trị lớn lấn át đặc trưng có giá trị nhỏ.
    """
    means: list[float]  # Giá trị trung bình của từng đặc trưng
    stds: list[float]   # Độ lệch chuẩn của từng đặc trưng

    @classmethod
    def fit(cls, rows: list[list[float]]) -> "StandardScaler":
        # Tính mean và std từ dữ liệu huấn luyện
        if not rows:
            raise ValueError("Cannot fit scaler on an empty dataset.")

        feature_count = len(rows[0])
        means = []
        stds = []
        for index in range(feature_count):
            column = [row[index] for row in rows]
            mean = sum(column) / len(column)
            variance = sum((value - mean) ** 2 for value in column) / len(column)
            std = math.sqrt(variance)
            means.append(mean)
            stds.append(std if std > 1e-12 else 1.0)  # Tránh chia cho 0
        return cls(means=means, stds=stds)

    def transform_row(self, row: list[float]) -> list[float]:
        # Chuẩn hóa một mẫu: (x - mean) / std
        return [(value - mean) / std for value, mean, std in zip(row, self.means, self.stds)]

    def transform(self, rows: list[list[float]]) -> list[list[float]]:
        # Chuẩn hóa nhiều mẫu cùng lúc
        return [self.transform_row(row) for row in rows]

    def to_dict(self) -> dict[str, list[float] | list[str]]:
        # Chuyển scaler thành dictionary để lưu JSON
        return {"features": FEATURES, "means": self.means, "stds": self.stds}

    @classmethod
    def from_dict(cls, payload: dict[str, list[float]]) -> "StandardScaler":
        # Đọc scaler từ dictionary đã lưu
        return cls(means=list(payload["means"]), stds=list(payload["stds"]))
