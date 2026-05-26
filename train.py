from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path
from typing import Mapping, TypedDict

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from water_quality_ann.ann import SimpleANN
from water_quality_ann.data import FEATURES, LABELS, label_counts, load_dataset_csv, save_dataset_csv, stratified_split
from water_quality_ann.preprocessing import StandardScaler, one_hot_many


ARCHITECTURES = [
    ("ANN_1lop_32", [32]),
    ("ANN_2lop_64_32", [64, 32]),
    ("ANN_2lop_128_64", [128, 64]),
]


class ResultRow(TypedDict):
    model: str
    architecture: str
    test_loss: float
    test_accuracy: float


def _format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def _save_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _save_metrics_csv(path: Path, histories: dict[str, list[dict[str, float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["model", "epoch", "train_loss", "test_loss", "train_accuracy", "test_accuracy"],
        )
        writer.writeheader()
        for model_name, rows in histories.items():
            for row in rows:
                writer.writerow({"model": model_name, **row})


def _save_comparison_csv(path: Path, results: list[ResultRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["model", "architecture", "test_loss", "test_accuracy"])
        writer.writeheader()
        for row in results:
            writer.writerow(row)


def _chart(series: dict[str, list[float]], title: str, y_min: float | None = None, y_max: float | None = None) -> str:
    width = 520
    height = 260
    padding_left = 46
    padding_right = 18
    padding_top = 30
    padding_bottom = 36
    plot_width = width - padding_left - padding_right
    plot_height = height - padding_top - padding_bottom
    colors = ["#0f766e", "#f59e0b", "#dc2626", "#2563eb"]

    all_values = [value for values in series.values() for value in values]
    if not all_values:
        return ""
    min_value = min(all_values) if y_min is None else y_min
    max_value = max(all_values) if y_max is None else y_max
    if abs(max_value - min_value) < 1e-9:
        max_value = min_value + 1.0

    def point(index: int, value: float, total: int) -> tuple[float, float]:
        x = padding_left + (index / max(total - 1, 1)) * plot_width
        y = padding_top + (1 - (value - min_value) / (max_value - min_value)) * plot_height
        return x, y

    polylines = []
    legends = []
    for color_index, (name, values) in enumerate(series.items()):
        color = colors[color_index % len(colors)]
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in [point(index, value, len(values)) for index, value in enumerate(values)])
        polylines.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{points}" />')
        legend_x = padding_left + color_index * 155
        legends.append(
            f'<circle cx="{legend_x}" cy="238" r="5" fill="{color}" />'
            f'<text x="{legend_x + 10}" y="243" font-size="12" fill="#334155">{html.escape(name)}</text>'
        )

    y_label_top = f"{max_value:.2f}"
    y_label_bottom = f"{min_value:.2f}"
    return f"""
    <svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">
      <rect x="0" y="0" width="{width}" height="{height}" rx="10" fill="#ffffff" />
      <text x="{padding_left}" y="20" font-size="15" font-weight="700" fill="#0f172a">{html.escape(title)}</text>
      <line x1="{padding_left}" y1="{padding_top}" x2="{padding_left}" y2="{padding_top + plot_height}" stroke="#cbd5e1" />
      <line x1="{padding_left}" y1="{padding_top + plot_height}" x2="{padding_left + plot_width}" y2="{padding_top + plot_height}" stroke="#cbd5e1" />
      <text x="8" y="{padding_top + 5}" font-size="11" fill="#64748b">{y_label_top}</text>
      <text x="8" y="{padding_top + plot_height}" font-size="11" fill="#64748b">{y_label_bottom}</text>
      {"".join(polylines)}
      {"".join(legends)}
    </svg>
    """


def _render_report(path: Path, results: list[ResultRow], histories: dict[str, list[dict[str, float]]]) -> None:
    best = max(results, key=lambda row: float(row["test_accuracy"]))
    result_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row['model']))}</td>"
        f"<td>{html.escape(str(row['architecture']))}</td>"
        f"<td>{float(row['test_loss']):.4f}</td>"
        f"<td>{_format_percent(float(row['test_accuracy']))}</td>"
        "</tr>"
        for row in results
    )

    cards = []
    for model_name, rows in histories.items():
        loss_chart = _chart(
            {
                "Train loss": [row["train_loss"] for row in rows],
                "Test loss": [row["test_loss"] for row in rows],
            },
            f"Loss - {model_name}",
        )
        accuracy_chart = _chart(
            {
                "Train acc": [row["train_accuracy"] for row in rows],
                "Test acc": [row["test_accuracy"] for row in rows],
            },
            f"Accuracy - {model_name}",
            y_min=0,
            y_max=1,
        )
        cards.append(f"<section class='chart-card'>{loss_chart}{accuracy_chart}</section>")

    html_content = f"""<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Bao cao huan luyen ANN - Chat luong nuoc</title>
  <style>
    body {{
      margin: 0;
      background: #edf4f2;
      color: #0f172a;
      font-family: Georgia, "Times New Roman", serif;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 36px 24px 52px;
    }}
    header {{
      border-left: 8px solid #0f766e;
      padding-left: 18px;
      margin-bottom: 28px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: clamp(30px, 5vw, 54px);
      letter-spacing: 0;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
      margin-bottom: 24px;
    }}
    .tile, table, .chart-card {{
      background: #fffdf7;
      border: 1px solid #c8d9d3;
      border-radius: 8px;
      box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
    }}
    .tile {{ padding: 18px; }}
    .tile strong {{
      display: block;
      font-size: 26px;
      color: #0f766e;
      margin-top: 6px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      overflow: hidden;
      margin: 18px 0 24px;
    }}
    th, td {{
      padding: 13px 14px;
      border-bottom: 1px solid #dbe7e3;
      text-align: left;
    }}
    th {{
      background: #0f766e;
      color: white;
      font-weight: 700;
    }}
    .chart-card {{
      margin: 16px 0;
      padding: 16px;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }}
    svg {{
      width: 100%;
      height: auto;
      display: block;
    }}
    @media (max-width: 860px) {{
      .summary, .chart-card {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Phan loai chat luong nuoc bang ANN</h1>
      <p>So sanh 3 kien truc mang voi Softmax va categorical cross-entropy.</p>
    </header>
    <section class="summary">
      <div class="tile">Mo hinh tot nhat<strong>{html.escape(str(best["model"]))}</strong></div>
      <div class="tile">Do chinh xac test<strong>{_format_percent(float(best["test_accuracy"]))}</strong></div>
      <div class="tile">So lop dau ra<strong>{len(LABELS)} lop</strong></div>
    </section>
    <h2>Bang so sanh</h2>
    <table>
      <thead>
        <tr><th>Mo hinh</th><th>Kien truc</th><th>Test loss</th><th>Test accuracy</th></tr>
      </thead>
      <tbody>{result_rows}</tbody>
    </table>
    <h2>Do thi loss va accuracy</h2>
    {"".join(cards)}
  </main>
</body>
</html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_content, encoding="utf-8")


def run_training(args: argparse.Namespace) -> None:
    data_path = ROOT / "data" / "water_quality.csv"
    if args.regenerate or not data_path.exists():
        save_dataset_csv(data_path, samples_per_class=args.samples_per_class, seed=args.seed)

    x, y = load_dataset_csv(data_path)
    x_train, x_test, y_train_labels, y_test_labels = stratified_split(x, y, test_size=0.2, seed=args.seed)
    scaler = StandardScaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    y_train = one_hot_many(y_train_labels)
    y_test = one_hot_many(y_test_labels)

    _save_json(ROOT / "models" / "scaler.json", scaler.to_dict())

    results: list[ResultRow] = []
    histories: dict[str, list[dict[str, float]]] = {}

    print("Dataset:", data_path)
    print("Train label counts:", label_counts(y_train_labels))
    print("Test label counts:", label_counts(y_test_labels))
    print()

    for index, (model_name, hidden_layers) in enumerate(ARCHITECTURES, start=1):
        print(f"Training {model_name}: hidden_layers={hidden_layers}")
        model = SimpleANN(
            input_size=len(FEATURES),
            hidden_layers=hidden_layers,
            output_size=len(LABELS),
            learning_rate=args.learning_rate,
            seed=args.seed + index,
        )
        history = model.fit(x_train_scaled, y_train, x_test_scaled, y_test, epochs=args.epochs, seed=args.seed + index * 10)
        test_loss, test_accuracy = model.evaluate(x_test_scaled, y_test)
        histories[model_name] = history
        results.append(
            {
                "model": model_name,
                "architecture": " -> ".join(str(size) for size in [len(FEATURES), *hidden_layers, len(LABELS)]),
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        )
        model.save(ROOT / "models" / f"{model_name}.json")
        print(f"  test_loss={test_loss:.4f}, test_accuracy={_format_percent(test_accuracy)}")

    best_result = max(results, key=lambda row: float(row["test_accuracy"]))
    best_model_name = str(best_result["model"])
    best_model = SimpleANN.load(ROOT / "models" / f"{best_model_name}.json")
    best_model.save(ROOT / "models" / "best_model.json")

    _save_metrics_csv(ROOT / "outputs" / "training_metrics.csv", histories)
    _save_comparison_csv(ROOT / "outputs" / "model_comparison.csv", results)
    _render_report(ROOT / "outputs" / "training_report.html", results, histories)

    print()
    print(f"Best model: {best_model_name} ({_format_percent(float(best_result['test_accuracy']))})")
    print("Saved:")
    print(f"  {ROOT / 'models' / 'best_model.json'}")
    print(f"  {ROOT / 'outputs' / 'training_report.html'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ANN models for domestic water quality classification.")
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--samples-per-class", type=int, default=2500)
    parser.add_argument("--learning-rate", type=float, default=0.018)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--regenerate", action="store_true", help="Regenerate the synthetic dataset before training.")
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
