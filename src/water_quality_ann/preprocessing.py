from __future__ import annotations

import math
from dataclasses import dataclass

from .data import FEATURES, LABELS


def one_hot(label: str) -> list[float]:
    return [1.0 if item == label else 0.0 for item in LABELS]


def one_hot_many(labels: list[str]) -> list[list[float]]:
    return [one_hot(label) for label in labels]


def argmax(values: list[float]) -> int:
    best_index = 0
    best_value = values[0]
    for index, value in enumerate(values[1:], start=1):
        if value > best_value:
            best_index = index
            best_value = value
    return best_index


@dataclass
class StandardScaler:
    means: list[float]
    stds: list[float]

    @classmethod
    def fit(cls, rows: list[list[float]]) -> "StandardScaler":
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
            stds.append(std if std > 1e-12 else 1.0)
        return cls(means=means, stds=stds)

    def transform_row(self, row: list[float]) -> list[float]:
        return [(value - mean) / std for value, mean, std in zip(row, self.means, self.stds)]

    def transform(self, rows: list[list[float]]) -> list[list[float]]:
        return [self.transform_row(row) for row in rows]

    def to_dict(self) -> dict[str, list[float] | list[str]]:
        return {"features": FEATURES, "means": self.means, "stds": self.stds}

    @classmethod
    def from_dict(cls, payload: dict[str, list[float]]) -> "StandardScaler":
        return cls(means=list(payload["means"]), stds=list(payload["stds"]))
