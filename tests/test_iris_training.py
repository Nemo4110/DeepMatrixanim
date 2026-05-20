import numpy as np

from deepmatrixanim.iris_training import (
    accuracy,
    build_iris_training_result,
    cross_entropy_loss,
    forward,
    initialize_weights,
    load_standardized_iris,
    logit_contrast_projection,
    train_mlp,
)


def test_load_standardized_iris_uses_full_dataset():
    x, y, target_names = load_standardized_iris()

    assert x.shape == (150, 4)
    assert y.shape == (150,)
    assert target_names == ("setosa", "versicolor", "virginica")
    np.testing.assert_allclose(x.mean(axis=0), np.zeros(4), atol=1e-12)
    np.testing.assert_allclose(x.std(axis=0), np.ones(4), atol=1e-12)
    np.testing.assert_array_equal(np.bincount(y), np.array([50, 50, 50]))


def test_initialize_weights_and_forward_shapes_are_deterministic():
    x, y, _ = load_standardized_iris()
    weights = initialize_weights(seed=7)

    assert weights.w1.shape == (8, 4)
    assert weights.b1.shape == (8,)
    assert weights.w2.shape == (3, 8)
    assert weights.b2.shape == (3,)

    repeated = initialize_weights(seed=7)
    np.testing.assert_allclose(weights.w1, repeated.w1)
    np.testing.assert_allclose(weights.w2, repeated.w2)

    result = forward(x, weights)

    assert result["z1"].shape == (150, 8)
    assert result["a1"].shape == (150, 8)
    assert result["logits"].shape == (150, 3)
    assert result["probabilities"].shape == (150, 3)
    np.testing.assert_allclose(result["probabilities"].sum(axis=1), np.ones(150))
    assert cross_entropy_loss(result["probabilities"], y) > 0.0
    assert 0.0 <= accuracy(result["logits"], y) <= 1.0


def test_train_mlp_lowers_loss_and_improves_accuracy():
    x, y, _ = load_standardized_iris()
    initial = initialize_weights(seed=7)
    initial_forward = forward(x, initial)

    trained = train_mlp(x, y, initial)
    trained_forward = forward(x, trained)

    initial_loss = cross_entropy_loss(initial_forward["probabilities"], y)
    trained_loss = cross_entropy_loss(trained_forward["probabilities"], y)

    assert trained_loss < initial_loss
    assert accuracy(trained_forward["logits"], y) >= 0.9
    assert accuracy(trained_forward["logits"], y) > accuracy(initial_forward["logits"], y)


def test_logit_contrast_projection_is_fixed_two_dimensional_map():
    logits = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    projected = logit_contrast_projection(logits)

    assert projected.shape == (3, 2)
    np.testing.assert_allclose(projected[:, 0], np.array([1.0, -1.0, 0.0]) / np.sqrt(2.0))
    np.testing.assert_allclose(projected[:, 1], np.array([1.0, 1.0, -2.0]) / np.sqrt(6.0))


def test_build_iris_training_result_contains_shared_projection_views():
    result = build_iris_training_result()

    assert result.x.shape == (150, 4)
    assert result.y.shape == (150,)
    assert result.target_names == ("setosa", "versicolor", "virginica")
    assert result.input_projection.shape == (150, 2)
    assert result.hidden_before_projection.shape == (150, 2)
    assert result.hidden_after_projection.shape == (150, 2)
    assert result.logits_before_projection.shape == (150, 2)
    assert result.logits_after_projection.shape == (150, 2)
    assert result.trained_loss < result.initial_loss
    assert result.trained_accuracy >= 0.9
    assert result.trained_accuracy > result.initial_accuracy
