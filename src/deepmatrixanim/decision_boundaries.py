"""Deterministic helpers for decision-boundary representation demos."""

from __future__ import annotations

import numpy as np


def first_layer_matrix() -> np.ndarray:
    """Return the W1 matrix that expands 2D XOR points into four scores."""
    return np.array(
        [
            [1.0, 1.0],
            [-1.0, -1.0],
            [1.0, -1.0],
            [-1.0, 1.0],
        ]
    )


def second_layer_matrix() -> np.ndarray:
    """Return the W2 matrix that groups hidden scores into two coordinates."""
    return np.array(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0],
        ]
    )


def demo_decision_boundary_points() -> tuple[np.ndarray, np.ndarray]:
    """Return XOR-like 2D points and labels for a representation-space demo."""
    points = np.array(
        [
            [1.2, 1.0],
            [1.6, 0.8],
            [-1.1, -1.3],
            [-1.6, -0.7],
            [1.2, -1.0],
            [1.6, -0.8],
            [-1.1, 1.3],
            [-1.6, 0.7],
        ]
    )
    labels = (points[:, 0] * points[:, 1] > 0).astype(int)
    return points, labels


def hidden_preactivation(points: np.ndarray) -> np.ndarray:
    """Apply the first linear map, producing four hidden pre-activation scores."""
    points = np.asarray(points, dtype=float)
    return points @ first_layer_matrix().T


def hidden_activation(points: np.ndarray) -> np.ndarray:
    """Apply ReLU to the four hidden pre-activation scores."""
    return np.maximum(0.0, hidden_preactivation(points))


def representation_space_transform(points: np.ndarray) -> np.ndarray:
    """Map XOR-like input points into a 2D space where a linear split works."""
    return hidden_activation(points) @ second_layer_matrix().T
