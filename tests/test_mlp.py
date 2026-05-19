import numpy as np

from deepmatrixanim.mlp import demo_forward_pass


def test_demo_forward_pass_tracks_each_mlp_stage():
    result = demo_forward_pass()

    np.testing.assert_allclose(result["x"], np.array([[2.0], [-1.0], [0.5]]))
    np.testing.assert_allclose(
        result["w1"],
        np.array(
            [
                [1.0, -2.0, 0.0],
                [-1.0, 1.0, 2.0],
                [0.5, 0.0, -1.0],
                [2.0, 1.0, 1.0],
            ]
        ),
    )
    np.testing.assert_allclose(result["b1"], np.array([[0.5], [-0.5], [1.0], [0.0]]))
    np.testing.assert_allclose(result["z1"], np.array([[4.5], [-2.5], [1.5], [3.5]]))
    np.testing.assert_allclose(result["a1"], np.array([[4.5], [0.0], [1.5], [3.5]]))
    np.testing.assert_allclose(result["w2"], np.array([[1.0, -1.0, 0.5, 2.0], [-0.5, 1.0, 1.0, -1.0]]))
    np.testing.assert_allclose(result["b2"], np.array([[0.25], [-0.75]]))
    np.testing.assert_allclose(result["y"], np.array([[12.5], [-5.0]]))
