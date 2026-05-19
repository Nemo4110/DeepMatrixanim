import numpy as np

from deepmatrixanim.linear_maps import (
    affine_transform,
    demo_bias_vector,
    demo_input_vector,
    demo_linear_matrix,
    demo_point_cloud,
    relu_transform,
)


def test_demo_linear_layer_values_are_deterministic():
    np.testing.assert_allclose(demo_input_vector(), np.array([1.5, 0.8]))
    np.testing.assert_allclose(demo_linear_matrix(), np.array([[1.2, 0.6], [-0.4, 1.1]]))
    np.testing.assert_allclose(demo_bias_vector(), np.array([0.8, -0.35]))


def test_affine_transform_applies_matrix_then_bias_to_vector():
    result = affine_transform(demo_input_vector(), demo_linear_matrix(), demo_bias_vector())

    np.testing.assert_allclose(result, np.array([3.08, -0.07]))


def test_affine_transform_applies_to_point_cloud_rows():
    points = np.array([[-1.0, 1.0], [0.0, 0.0], [1.0, -0.5]])
    result = affine_transform(points, demo_linear_matrix(), demo_bias_vector())

    expected = np.array([[0.2, 1.15], [0.8, -0.35], [1.7, -1.3]])
    np.testing.assert_allclose(result, expected)


def test_demo_point_cloud_contains_points_in_multiple_quadrants():
    points = demo_point_cloud()

    assert points.shape == (7, 2)
    assert np.any(points[:, 0] < 0)
    assert np.any(points[:, 1] < 0)
    assert np.any((points[:, 0] > 0) & (points[:, 1] > 0))


def test_relu_transform_clamps_negative_coordinates():
    points = np.array([[-1.0, 2.0], [2.0, -3.0], [-2.0, -1.0], [1.0, 1.0]])

    result = relu_transform(points)

    expected = np.array([[0.0, 2.0], [2.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    np.testing.assert_allclose(result, expected)
