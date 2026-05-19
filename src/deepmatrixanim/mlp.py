"""Numeric helpers for the first DeepMatrixanim demo."""

import numpy as np


def demo_forward_pass() -> dict[str, np.ndarray]:
    """Return intermediate values for a fixed two-layer MLP demo."""
    x = np.array([[2.0], [-1.0], [0.5]])
    w1 = np.array(
        [
            [1.0, -2.0, 0.0],
            [-1.0, 1.0, 2.0],
            [0.5, 0.0, -1.0],
            [2.0, 1.0, 1.0],
        ]
    )
    b1 = np.array([[0.5], [-0.5], [1.0], [0.0]])
    z1 = w1 @ x + b1
    a1 = np.maximum(0.0, z1)
    w2 = np.array([[1.0, -1.0, 0.5, 2.0], [-0.5, 1.0, 1.0, -1.0]])
    b2 = np.array([[0.25], [-0.75]])
    y = w2 @ a1 + b2

    return {
        "x": x,
        "w1": w1,
        "b1": b1,
        "z1": z1,
        "a1": a1,
        "w2": w2,
        "b2": b2,
        "y": y,
    }

