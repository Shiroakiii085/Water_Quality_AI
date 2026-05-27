from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from water_quality_ann.ann import SimpleANN
from water_quality_ann.data import FEATURES, FEATURE_LABELS, LABELS
from water_quality_ann.preprocessing import StandardScaler, argmax


def load_scaler(path: Path) -> StandardScaler:
    # Đọc tham số chuẩn hóa từ file JSON
    with path.open("r", encoding="utf-8") as file:
        return StandardScaler.from_dict(json.load(file))


def parse_args() -> argparse.Namespace:
    # Nhập 8 chỉ số nước từ dòng lệnh để dự đoán
    parser = argparse.ArgumentParser(description="Predict water quality from chemical and microbiological indicators.")
    parser.add_argument("--pH", type=float, required=True)
    parser.add_argument("--Hardness", type=float, required=True)
    parser.add_argument("--Solids", type=float, required=True)
    parser.add_argument("--Chlorine", type=float, required=True)
    parser.add_argument("--Sulfate", type=float, required=True)
    parser.add_argument("--Lead", type=float, required=True)
    parser.add_argument("--Mercury", type=float, required=True)
    parser.add_argument("--Coliform", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    model_path = ROOT / "models" / "best_model.json"
    scaler_path = ROOT / "models" / "scaler.json"
    if not model_path.exists() or not scaler_path.exists():
        raise SystemExit("Chưa có model. Hãy chạy: python train.py")

    # Đọc model và scaler đã huấn luyện
    args = parse_args()
    row = [getattr(args, feature) for feature in FEATURES]
    model = SimpleANN.load(model_path)
    scaler = load_scaler(scaler_path)

    # Dự đoán: chuẩn hóa -> forward pass -> argmax -> nhãn
    probabilities = model.predict_proba(scaler.transform_row(row))
    label = LABELS[argmax(probabilities)]

    # In kết quả
    print("Chỉ số đầu vào:")
    for feature, value in zip(FEATURES, row):
        print(f"  {FEATURE_LABELS[feature]}: {value}")
    print()
    print(f"Kết quả dự đoán: {label}")
    print("Xác suất:")
    for label_name, probability in zip(LABELS, probabilities):
        print(f"  {label_name}: {probability * 100:.2f}%")


if __name__ == "__main__":
    main()
