"""Deterministic Iris MLP training helpers for before/after visualizations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class MLPWeights:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray


@dataclass(frozen=True)
class IrisTrainingResult:
    x: np.ndarray
    y: np.ndarray
    target_names: tuple[str, ...]
    initial_weights: MLPWeights
    trained_weights: MLPWeights
    initial_forward: dict[str, np.ndarray]
    trained_forward: dict[str, np.ndarray]
    input_projection: np.ndarray
    hidden_before_projection: np.ndarray
    hidden_after_projection: np.ndarray
    logits_before_projection: np.ndarray
    logits_after_projection: np.ndarray
    initial_accuracy: float
    trained_accuracy: float
    initial_loss: float
    trained_loss: float


def load_standardized_iris() -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Load the full Iris dataset and standardize all four input features."""
    iris = load_iris()
    x = StandardScaler().fit_transform(iris.data.astype(float))
    y = iris.target.astype(int)
    return x, y, tuple(str(name) for name in iris.target_names)


def initialize_weights(seed: int = 7) -> MLPWeights:
    """Return deterministic initial weights for a 4 -> 8 -> 3 MLP."""
    rng = np.random.default_rng(seed)
    return MLPWeights(
        w1=rng.normal(0.0, 0.35, size=(8, 4)),
        b1=np.zeros(8),
        w2=rng.normal(0.0, 0.35, size=(3, 8)),
        b2=np.zeros(3),
    )


def forward(x: np.ndarray, weights: MLPWeights) -> dict[str, np.ndarray]:
    """Run a forward pass and return all layer values used by the scene."""
    z1 = np.asarray(x, dtype=float) @ weights.w1.T + weights.b1
    a1 = np.maximum(0.0, z1)
    logits = a1 @ weights.w2.T + weights.b2
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return {
        "z1": z1,
        "a1": a1,
        "logits": logits,
        "probabilities": probabilities,
    }


def cross_entropy_loss(probabilities: np.ndarray, y: np.ndarray) -> float:
    """Return mean sparse softmax cross-entropy."""
    rows = np.arange(len(y))
    clipped = np.clip(probabilities[rows, y], 1e-12, 1.0)
    return float(-np.mean(np.log(clipped)))


def accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    """Return classification accuracy from raw logits."""
    return float(np.mean(np.argmax(logits, axis=1) == y))


def train_mlp(
    x: np.ndarray,
    y: np.ndarray,
    initial: MLPWeights,
    epochs: int = 1200,
    learning_rate: float = 0.05,
) -> MLPWeights:
    """Train the MLP with full-batch NumPy backpropagation."""
    weights = MLPWeights(
        w1=initial.w1.copy(),
        b1=initial.b1.copy(),
        w2=initial.w2.copy(),
        b2=initial.b2.copy(),
    )
    n_samples = x.shape[0]
    one_hot = np.eye(3)[y]

    for _ in range(epochs):
        values = forward(x, weights)
        probabilities = values["probabilities"]
        a1 = values["a1"]
        z1 = values["z1"]

        dlogits = (probabilities - one_hot) / n_samples
        dw2 = dlogits.T @ a1
        db2 = dlogits.sum(axis=0)
        da1 = dlogits @ weights.w2
        dz1 = da1 * (z1 > 0.0)
        dw1 = dz1.T @ x
        db1 = dz1.sum(axis=0)

        weights = MLPWeights(
            w1=weights.w1 - learning_rate * dw1,
            b1=weights.b1 - learning_rate * db1,
            w2=weights.w2 - learning_rate * dw2,
            b2=weights.b2 - learning_rate * db2,
        )

    return weights


def logit_contrast_projection(logits: np.ndarray) -> np.ndarray:
    """Project 3D logits into a fixed 2D contrast plane."""
    logits = np.asarray(logits, dtype=float)
    u = (logits[:, 0] - logits[:, 1]) / np.sqrt(2.0)
    v = (logits[:, 0] + logits[:, 1] - 2.0 * logits[:, 2]) / np.sqrt(6.0)
    return np.column_stack([u, v])


def build_iris_training_result() -> IrisTrainingResult:
    """Build deterministic Iris training data and projections for Manim."""
    x, y, target_names = load_standardized_iris()
    initial_weights = initialize_weights()
    trained_weights = train_mlp(x, y, initial_weights)
    initial_forward = forward(x, initial_weights)
    trained_forward = forward(x, trained_weights)

    input_projection = PCA(n_components=2, random_state=0).fit_transform(x)

    hidden_stack = np.vstack([initial_forward["a1"], trained_forward["a1"]])
    hidden_pca = PCA(n_components=2, random_state=0).fit(hidden_stack)
    hidden_before_projection = hidden_pca.transform(initial_forward["a1"])
    hidden_after_projection = hidden_pca.transform(trained_forward["a1"])

    logits_before_projection = logit_contrast_projection(initial_forward["logits"])
    logits_after_projection = logit_contrast_projection(trained_forward["logits"])

    initial_loss = cross_entropy_loss(initial_forward["probabilities"], y)
    trained_loss = cross_entropy_loss(trained_forward["probabilities"], y)

    return IrisTrainingResult(
        x=x,
        y=y,
        target_names=target_names,
        initial_weights=initial_weights,
        trained_weights=trained_weights,
        initial_forward=initial_forward,
        trained_forward=trained_forward,
        input_projection=input_projection,
        hidden_before_projection=hidden_before_projection,
        hidden_after_projection=hidden_after_projection,
        logits_before_projection=logits_before_projection,
        logits_after_projection=logits_after_projection,
        initial_accuracy=accuracy(initial_forward["logits"], y),
        trained_accuracy=accuracy(trained_forward["logits"], y),
        initial_loss=initial_loss,
        trained_loss=trained_loss,
    )
