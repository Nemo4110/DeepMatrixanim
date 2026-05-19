import numpy as np

from deepmatrixanim.decision_boundaries import (
    demo_decision_boundary_points,
    hidden_activation,
    hidden_preactivation,
    representation_space_transform,
)


def test_demo_decision_boundary_points_are_xor_labeled():
    points, labels = demo_decision_boundary_points()

    assert points.shape == (8, 2)
    np.testing.assert_array_equal(labels, np.array([1, 1, 1, 1, 0, 0, 0, 0]))
    np.testing.assert_array_equal(labels, (points[:, 0] * points[:, 1] > 0).astype(int))


def test_representation_space_transform_maps_xor_to_linear_split():
    points, labels = demo_decision_boundary_points()

    represented = representation_space_transform(points)

    assert represented.shape == (8, 2)
    np.testing.assert_allclose(
        represented,
        np.array(
            [
                [2.2, 0.2],
                [2.4, 0.8],
                [2.4, 0.2],
                [2.3, 0.9],
                [0.2, 2.2],
                [0.8, 2.4],
                [0.2, 2.4],
                [0.9, 2.3],
            ]
        ),
    )
    assert np.all(represented[labels == 1, 0] > represented[labels == 1, 1])
    assert np.all(represented[labels == 0, 1] > represented[labels == 0, 0])


def test_representation_space_transform_exposes_w1_relu_w2_steps():
    points, _ = demo_decision_boundary_points()

    preactivation = hidden_preactivation(points)
    activation = hidden_activation(points)
    represented = representation_space_transform(points)

    np.testing.assert_allclose(
        preactivation[:2],
        np.array(
            [
                [2.2, -2.2, 0.2, -0.2],
                [2.4, -2.4, 0.8, -0.8],
            ]
        ),
    )
    np.testing.assert_allclose(
        activation[:2],
        np.array(
            [
                [2.2, 0.0, 0.2, 0.0],
                [2.4, 0.0, 0.8, 0.0],
            ]
        ),
    )
    np.testing.assert_allclose(activation, np.maximum(0.0, preactivation))
    np.testing.assert_allclose(represented[:, 0], activation[:, 0] + activation[:, 1])
    np.testing.assert_allclose(represented[:, 1], activation[:, 2] + activation[:, 3])
