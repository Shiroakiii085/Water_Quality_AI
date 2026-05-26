from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import TypedDict

from .data import FEATURES, LABELS
from .preprocessing import argmax


class ModelPayload(TypedDict):
    features: list[str]
    labels: list[str]
    input_size: int
    hidden_layers: list[int]
    output_size: int
    learning_rate: float
    layer_sizes: list[int]
    weights: list[list[list[float]]]
    biases: list[list[float]]


class SimpleANN:
    """A compact multilayer neural network for multiclass classification."""

    def __init__(
        self,
        input_size: int,
        hidden_layers: list[int],
        output_size: int,
        learning_rate: float = 0.02,
        seed: int = 13,
    ) -> None:
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.layer_sizes = [input_size, *hidden_layers, output_size]
        self.weights: list[list[list[float]]] = []
        self.biases: list[list[float]] = []

        rng = random.Random(seed)
        for layer_index in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[layer_index]
            fan_out = self.layer_sizes[layer_index + 1]
            scale = math.sqrt(2.0 / fan_in) if layer_index < len(self.layer_sizes) - 2 else math.sqrt(1.0 / fan_in)
            self.weights.append([[rng.gauss(0.0, scale) for _ in range(fan_out)] for _ in range(fan_in)])
            self.biases.append([0.0 for _ in range(fan_out)])

    @staticmethod
    def _relu(values: list[float]) -> list[float]:
        return [value if value > 0.0 else 0.0 for value in values]

    @staticmethod
    def _softmax(values: list[float]) -> list[float]:
        max_value = max(values)
        exps = [math.exp(value - max_value) for value in values]
        total = sum(exps)
        return [value / total for value in exps]

    @staticmethod
    def _cross_entropy(probabilities: list[float], target: list[float]) -> float:
        return -sum(expected * math.log(max(predicted, 1e-12)) for predicted, expected in zip(probabilities, target))

    def forward(self, row: list[float]) -> tuple[list[list[float]], list[list[float]]]:
        activations = [row]
        pre_activations: list[list[float]] = []
        current = row

        for layer_index, (weights, bias) in enumerate(zip(self.weights, self.biases)):
            output_size = len(bias)
            z_values = []
            for output_index in range(output_size):
                total = bias[output_index]
                for input_index, value in enumerate(current):
                    total += value * weights[input_index][output_index]
                z_values.append(total)

            pre_activations.append(z_values)
            if layer_index == len(self.weights) - 1:
                current = self._softmax(z_values)
            else:
                current = self._relu(z_values)
            activations.append(current)

        return activations, pre_activations

    def predict_proba(self, row: list[float]) -> list[float]:
        return self.forward(row)[0][-1]

    def predict(self, row: list[float]) -> str:
        return LABELS[argmax(self.predict_proba(row))]

    def _train_sample(self, row: list[float], target: list[float]) -> float:
        activations, pre_activations = self.forward(row)
        probabilities = activations[-1]
        loss = self._cross_entropy(probabilities, target)

        deltas: list[list[float]] = [[] for _ in self.weights]
        deltas[-1] = [probability - expected for probability, expected in zip(probabilities, target)]

        for layer_index in range(len(self.weights) - 2, -1, -1):
            next_delta = deltas[layer_index + 1]
            next_weights = self.weights[layer_index + 1]
            z_values = pre_activations[layer_index]
            current_delta = []
            for unit_index, z_value in enumerate(z_values):
                propagated = 0.0
                for next_index, delta_value in enumerate(next_delta):
                    propagated += next_weights[unit_index][next_index] * delta_value
                current_delta.append(propagated if z_value > 0.0 else 0.0)
            deltas[layer_index] = current_delta

        for layer_index, delta in enumerate(deltas):
            previous_activation = activations[layer_index]
            weights = self.weights[layer_index]
            bias = self.biases[layer_index]
            for input_index, activation_value in enumerate(previous_activation):
                if activation_value == 0.0:
                    continue
                row_weights = weights[input_index]
                for output_index, delta_value in enumerate(delta):
                    row_weights[output_index] -= self.learning_rate * activation_value * delta_value
            for output_index, delta_value in enumerate(delta):
                bias[output_index] -= self.learning_rate * delta_value

        return loss

    def evaluate(self, x: list[list[float]], y: list[list[float]]) -> tuple[float, float]:
        if not x:
            return 0.0, 0.0

        total_loss = 0.0
        correct = 0
        for row, target in zip(x, y):
            probabilities = self.predict_proba(row)
            total_loss += self._cross_entropy(probabilities, target)
            if argmax(probabilities) == argmax(target):
                correct += 1
        return total_loss / len(x), correct / len(x)

    def fit(
        self,
        x_train: list[list[float]],
        y_train: list[list[float]],
        x_test: list[list[float]],
        y_test: list[list[float]],
        epochs: int = 35,
        seed: int = 13,
    ) -> list[dict[str, float]]:
        rng = random.Random(seed)
        indices = list(range(len(x_train)))
        history: list[dict[str, float]] = []

        for epoch in range(1, epochs + 1):
            rng.shuffle(indices)
            for index in indices:
                self._train_sample(x_train[index], y_train[index])

            train_loss, train_accuracy = self.evaluate(x_train, y_train)
            test_loss, test_accuracy = self.evaluate(x_test, y_test)
            history.append(
                {
                    "epoch": float(epoch),
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy,
                }
            )
        return history

    def to_dict(self) -> ModelPayload:
        return {
            "features": FEATURES,
            "labels": LABELS,
            "input_size": self.input_size,
            "hidden_layers": self.hidden_layers,
            "output_size": self.output_size,
            "learning_rate": self.learning_rate,
            "layer_sizes": self.layer_sizes,
            "weights": self.weights,
            "biases": self.biases,
        }

    @classmethod
    def from_dict(cls, payload: ModelPayload) -> "SimpleANN":
        model = cls(
            input_size=payload["input_size"],
            hidden_layers=payload["hidden_layers"],
            output_size=payload["output_size"],
            learning_rate=payload["learning_rate"],
            seed=13,
        )
        model.weights = payload["weights"]
        model.biases = payload["biases"]
        model.layer_sizes = payload["layer_sizes"]
        return model

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "SimpleANN":
        with Path(path).open("r", encoding="utf-8") as file:
            return cls.from_dict(json.load(file))
