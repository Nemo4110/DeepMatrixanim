"""Deterministic 2D linear algebra helpers for vector-space demos."""

from __future__ import annotations

import numpy as np


def demo_input_vector() -> np.ndarray:
    """Return the highlighted input vector for the affine-layer demo."""
    return np.array([1.5, 0.8])


def demo_linear_matrix() -> np.ndarray:
    """Return a 2x2 matrix with visible stretch, rotation, and shear."""
    return np.array([[1.2, 0.6], [-0.4, 1.1]])


def demo_bias_vector() -> np.ndarray:
    """Return the translation vector used after the linear transform."""
    return np.array([0.8, -0.35])


def demo_point_cloud() -> np.ndarray:
    """Return fixed 2D points spanning multiple quadrants."""
    return np.array(
        [
            [-1.5, 1.0],
            [-0.8, -0.6],
            [0.0, 0.0],
            [0.7, 1.2],
            [1.2, -0.9],
            [1.7, 0.4],
            [-1.2, -1.3],
        ]
    )


def affine_transform(values: np.ndarray, matrix: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Apply `matrix @ x + bias` to one vector or to row-wise point data."""
    values = np.asarray(values, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    bias = np.asarray(bias, dtype=float)
    if values.ndim == 1:
        return matrix @ values + bias
    return values @ matrix.T + bias


def relu_transform(values: np.ndarray) -> np.ndarray:
    """Apply coordinate-wise ReLU to a vector or row-wise point data."""
    return np.maximum(0.0, np.asarray(values, dtype=float))
