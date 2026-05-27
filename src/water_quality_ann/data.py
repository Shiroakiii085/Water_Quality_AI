from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Iterable


# Danh sách 8 chỉ số hóa học và vi sinh dùng để đánh giá chất lượng nước
FEATURES = [
    "pH",
    "Hardness",   # Độ cứng (mg/L)
    "Solids",     # Chất rắn hòa tan tổng số (mg/L)
    "Chlorine",   # Hàm lượng clo (mg/L)
    "Sulfate",    # Hàm lượng sunfat (mg/L)
    "Lead",       # Hàm lượng chì - Pb (mg/L)
    "Mercury",    # Hàm lượng thủy ngân - Hg (mg/L)
    "Coliform",   # Mật độ vi khuẩn Coliform (CFU/100mL)
]

# Nhãn hiển thị tiếng Việt cho từng chỉ số (dùng trong giao diện và CLI)
FEATURE_LABELS = {
    "pH": "pH",
    "Hardness": "Độ cứng (mg/L)",
    "Solids": "Chất rắn hòa tan (mg/L)",
    "Chlorine": "Clo (mg/L)",
    "Sulfate": "Sunfat (mg/L)",
    "Lead": "Chì - Pb (mg/L)",
    "Mercury": "Thủy ngân - Hg (mg/L)",
    "Coliform": "Coliform (CFU/100mL)",
}

# 4 lớp chất lượng nước đầu ra
LABELS = ["Tốt", "Trung bình", "Kém", "Nguy hiểm"]


def _round_row(row: dict[str, float]) -> dict[str, float]:
    # Làm tròn các giá trị theo độ chính xác phù hợp cho từng chỉ số
    return {
        "pH": round(row["pH"], 2),
        "Hardness": round(row["Hardness"], 1),
        "Solids": round(row["Solids"], 1),
        "Chlorine": round(row["Chlorine"], 3),
        "Sulfate": round(row["Sulfate"], 1),
        "Lead": round(row["Lead"], 5),
        "Mercury": round(row["Mercury"], 5),
        "Coliform": int(round(row["Coliform"])),
    }


def assess_quality(row: dict[str, float]) -> str:
    """Gán nhãn chất lượng nước dựa trên bộ luật ngưỡng (rule-based scoring).

    Hệ thống tính điểm tích lũy (score) và kiểm tra điều kiện nguy hiểm (danger).
    Các ngưỡng được đơn giản hóa cho mục đích học tập.
    """
    score = 0
    danger = False

    # pH: giá trị lý tưởng 6.5-8.5
    ph = row["pH"]
    if ph < 5.0 or ph > 10.5:       # Quá chua hoặc quá kiềm -> nguy hiểm
        danger = True
    elif ph < 5.5 or ph > 9.5:      # Lệch nhiều -> +2 điểm
        score += 2
    elif ph < 6.5 or ph > 8.5:      # Lệch nhẹ -> +1 điểm
        score += 1

    # Độ cứng: nước cứng ảnh hưởng đến sinh hoạt
    hardness = row["Hardness"]
    if hardness > 500:
        score += 2
    elif hardness > 300:
        score += 1

    # Chất rắn hòa tan: TDS cao gây vị khó chịu
    solids = row["Solids"]
    if solids > 1500:
        score += 2
    elif solids > 700:
        score += 1

    # Clo dư: quá nhiều hoặc quá ít đều không tốt
    chlorine = row["Chlorine"]
    if chlorine > 5.0:               # Clo quá cao -> nguy hiểm
        danger = True
    elif chlorine < 0.1 or chlorine > 3.0:
        score += 1

    # Sunfat: nồng độ cao gây mùi vị khó chịu
    sulfate = row["Sulfate"]
    if sulfate > 1000:               # Sunfat quá cao -> nguy hiểm
        danger = True
    elif sulfate > 500:
        score += 2
    elif sulfate > 250:
        score += 1

    # Chì: kim loại nặng độc hại
    lead = row["Lead"]
    if lead > 0.05:                  # Chì quá cao -> nguy hiểm
        danger = True
    elif lead > 0.015:
        score += 2
    elif lead > 0.01:
        score += 1

    # Thủy ngân: kim loại nặng cực độc
    mercury = row["Mercury"]
    if mercury > 0.01:               # Thủy ngân quá cao -> nguy hiểm
        danger = True
    elif mercury > 0.004:
        score += 2
    elif mercury > 0.002:
        score += 1

    # Vi khuẩn Coliform: chỉ thị ô nhiễm vi sinh
    coliform = row["Coliform"]
    if coliform > 1000:              # Coliform quá cao -> nguy hiểm
        danger = True
    elif coliform > 200:
        score += 2
    elif coliform > 50:
        score += 1

    # Phân loại cuối cùng dựa trên tổng điểm và cờ danger
    if danger or score >= 10:
        return "Nguy hiểm"
    if score >= 6:
        return "Kém"
    if score >= 2:
        return "Trung bình"
    return "Tốt"


def _base_good(rng: random.Random) -> dict[str, float]:
    # Tạo mẫu nước "tốt" - tất cả chỉ số trong ngưỡng an toàn
    return {
        "pH": rng.uniform(6.7, 8.2),
        "Hardness": rng.uniform(80, 220),
        "Solids": rng.uniform(80, 500),
        "Chlorine": rng.uniform(0.3, 2.0),
        "Sulfate": rng.uniform(40, 220),
        "Lead": rng.uniform(0.0, 0.008),
        "Mercury": rng.uniform(0.0, 0.0015),
        "Coliform": rng.uniform(0, 25),
    }


def _mild_value(feature: str, rng: random.Random) -> float:
    # Sinh giá trị "cảnh báo nhẹ" (mild) cho một chỉ số
    values = {
        "pH": lambda: rng.choice([rng.uniform(6.0, 6.45), rng.uniform(8.55, 9.1)]),
        "Hardness": lambda: rng.uniform(305, 430),
        "Solids": lambda: rng.uniform(710, 1100),
        "Chlorine": lambda: rng.choice([rng.uniform(0.02, 0.09), rng.uniform(3.1, 4.2)]),
        "Sulfate": lambda: rng.uniform(255, 420),
        "Lead": lambda: rng.uniform(0.0105, 0.0145),
        "Mercury": lambda: rng.uniform(0.0022, 0.0038),
        "Coliform": lambda: rng.uniform(55, 180),
    }
    return values[feature]()


def _bad_value(feature: str, rng: random.Random) -> float:
    # Sinh giá trị "xấu" (bad) cho một chỉ số
    values = {
        "pH": lambda: rng.choice([rng.uniform(5.1, 5.45), rng.uniform(9.6, 10.3)]),
        "Hardness": lambda: rng.uniform(510, 720),
        "Solids": lambda: rng.uniform(1510, 2200),
        "Chlorine": lambda: rng.uniform(4.2, 4.95),
        "Sulfate": lambda: rng.uniform(510, 850),
        "Lead": lambda: rng.uniform(0.016, 0.045),
        "Mercury": lambda: rng.uniform(0.0045, 0.009),
        "Coliform": lambda: rng.uniform(210, 900),
    }
    return values[feature]()


def _danger_value(feature: str, rng: random.Random) -> float:
    # Sinh giá trị "nguy hiểm" (danger) cho một chỉ số
    values = {
        "pH": lambda: rng.choice([rng.uniform(3.4, 4.9), rng.uniform(10.7, 12.0)]),
        "Hardness": lambda: rng.uniform(700, 1000),
        "Solids": lambda: rng.uniform(2200, 3200),
        "Chlorine": lambda: rng.uniform(5.2, 10.0),
        "Sulfate": lambda: rng.uniform(1050, 1400),
        "Lead": lambda: rng.uniform(0.055, 0.2),
        "Mercury": lambda: rng.uniform(0.011, 0.05),
        "Coliform": lambda: rng.uniform(1100, 5000),
    }
    return values[feature]()


def _sample_for_label(label: str, rng: random.Random) -> dict[str, float]:
    # Sinh một mẫu dữ liệu cho một lớp cụ thể
    row = _base_good(rng)
    if label == "Tốt":
        return _round_row(row)

    if label == "Trung bình":
        # Chọn 2-3 chỉ số, gán giá trị mild
        for feature in rng.sample(FEATURES, rng.choice([2, 3])):
            row[feature] = _mild_value(feature, rng)
        return _round_row(row)

    if label == "Kém":
        # Chọn 3-4 chỉ số gán bad, 35% thêm 1 chỉ số mild
        bad_features = rng.sample(FEATURES, rng.choice([3, 4]))
        for feature in bad_features:
            row[feature] = _bad_value(feature, rng)
        if rng.random() < 0.35:
            remaining = [feature for feature in FEATURES if feature not in bad_features]
            feature = rng.choice(remaining)
            row[feature] = _mild_value(feature, rng)
        return _round_row(row)

    # Nguy hiểm: 1 chỉ số danger làm trigger + 1-3 chỉ số phụ ngẫu nhiên
    trigger = rng.choice(["pH", "Chlorine", "Sulfate", "Lead", "Mercury", "Coliform"])
    row[trigger] = _danger_value(trigger, rng)
    for feature in rng.sample([feature for feature in FEATURES if feature != trigger], rng.choice([1, 2, 3])):
        row[feature] = rng.choice([_mild_value, _bad_value, _danger_value])(feature, rng)
    return _round_row(row)


def generate_dataset(samples_per_class: int = 120, seed: int = 13) -> list[dict[str, float | str]]:
    # Sinh toàn bộ dataset (mặc định 120 mẫu/lớp, 4 lớp = 480 mẫu)
    # Mỗi mẫu sau khi sinh được kiểm tra lại bằng assess_quality() để đảm bảo khớp nhãn
    rng = random.Random(seed)
    rows: list[dict[str, float | str]] = []

    for label in LABELS:
        accepted = 0
        attempts = 0
        while accepted < samples_per_class:
            attempts += 1
            row = _sample_for_label(label, rng)
            assigned = assess_quality(row)
            # Nếu luật đánh giá khớp với nhãn mong muốn thì chấp nhận
            # Nếu thử quá số lần cho phép thì chấp nhận luôn (tránh vô hạn)
            if assigned == label or attempts > samples_per_class * 80:
                item: dict[str, float | str] = dict(row)
                item["Quality"] = assigned
                rows.append(item)
                accepted += 1

    rng.shuffle(rows)
    return rows


def save_dataset_csv(path: str | Path, samples_per_class: int = 120, seed: int = 13) -> None:
    # Sinh dataset và lưu ra file CSV
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_dataset(samples_per_class=samples_per_class, seed=seed)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[*FEATURES, "Quality"])
        writer.writeheader()
        writer.writerows(rows)


def load_dataset_csv(path: str | Path) -> tuple[list[list[float]], list[str]]:
    # Đọc file CSV, trả về (X: list[list[float]], y: list[str])
    x: list[list[float]] = []
    y: list[str] = []
    with Path(path).open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append([float(row[feature]) for feature in FEATURES])
            y.append(row["Quality"])
    return x, y


def stratified_split(
    x: list[list[float]],
    y: list[str],
    test_size: float = 0.2,
    seed: int = 13,
) -> tuple[list[list[float]], list[list[float]], list[str], list[str]]:
    # Phân chia dữ liệu train/test theo phương pháp stratified
    # (giữ nguyên tỷ lệ các lớp trong cả hai tập)
    rng = random.Random(seed)
    groups: dict[str, list[int]] = {label: [] for label in LABELS}
    for index, label in enumerate(y):
        groups[label].append(index)

    train_indices: list[int] = []
    test_indices: list[int] = []
    for indices in groups.values():
        rng.shuffle(indices)
        test_count = max(1, round(len(indices) * test_size))
        test_indices.extend(indices[:test_count])
        train_indices.extend(indices[test_count:])

    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return (
        [x[index] for index in train_indices],
        [x[index] for index in test_indices],
        [y[index] for index in train_indices],
        [y[index] for index in test_indices],
    )


def label_counts(labels: Iterable[str]) -> dict[str, int]:
    # Đếm số lượng mẫu theo từng nhãn (dùng để kiểm tra phân phối dữ liệu)
    counts = {label: 0 for label in LABELS}
    for label in labels:
        counts[label] += 1
    return counts
