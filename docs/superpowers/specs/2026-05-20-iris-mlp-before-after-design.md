# Iris MLP Before/After Training Design

## Goal

Add a Manim demo that uses the complete classic Iris dataset to show how training changes an MLP's vector-space transformations. The scene should compare the same 150 samples before and after training as they move through:

```text
x -> W1x + b1 -> ReLU hidden space -> W2a + b2 -> logits
```

The viewer should see that training does not change the samples. Training changes the learned matrices, which reshape four-dimensional measurements into class-separable representations.

## Dataset Scope

Use the full Iris dataset through `sklearn.datasets.load_iris()`:

- 150 samples.
- 4 input features: sepal length, sepal width, petal length, petal width.
- 3 target classes: setosa, versicolor, virginica.

Do not reduce the dataset to two classes or two features. The demo may use two-dimensional projections for visualization, but the model must train on all four standardized features and all three classes.

## Model Scope

Use a small transparent NumPy MLP instead of `sklearn.neural_network.MLPClassifier`.

Recommended architecture:

```text
4 -> 8 -> 3
```

with:

- ReLU hidden activation.
- Softmax cross-entropy loss.
- Full-batch gradient descent.
- Fixed seed for deterministic initial weights.
- Fixed epoch count and learning rate chosen to reliably improve Iris classification.

The implementation should expose both the initial and trained weights so the scene can compare the exact same network architecture before and after training.

## Projection Strategy

Because Iris input, hidden activations, and logits are not all two-dimensional, the scene must explicitly use projections rather than pretending the underlying spaces are 2D.

### Input Projection

Standardize the 4D Iris features with `StandardScaler`. Fit PCA on the standardized input data and project to 2D.

The scene should label this as:

```text
PCA of standardized 4D Iris features
```

### Hidden Projection

Hidden activations are 8D.

Compute:

```text
a_before = ReLU(W1_initial x + b1_initial)
a_after = ReLU(W1_trained x + b1_trained)
```

Fit one shared PCA basis on:

```text
vstack([a_before, a_after])
```

Then project `a_before` and `a_after` into the same 2D hidden projection space. This keeps the before/after panels visually comparable.

### Logit Projection

Logits are 3D. Use a fixed three-class contrast projection rather than PCA:

```text
u = (logit_0 - logit_1) / sqrt(2)
v = (logit_0 + logit_1 - 2 * logit_2) / sqrt(6)
```

This projection is stable across before/after states and preserves the idea that each class has its own logit direction.

## Scene Design

Scene class:

```python
IrisMLPBeforeAfterTraining
```

The scene should use the existing DeepMatrixanim style: dark background, sparse text, colored points, formulas as anchors, and clear geometric motion.

### Visual Layout

Use three conceptual spaces:

1. Input PCA space.
2. Hidden activation PCA space.
3. Logit contrast space.

Use side-by-side before/after comparison, but avoid showing too many panels at once. Prefer time-stepped animation:

1. Show the complete Iris input PCA scatter once.
2. Show the random-weight path into hidden and logit projection spaces.
3. Show the trained-weight path into hidden and logit projection spaces.
4. End with before/after logit projection panels side by side.

The final side-by-side frame should compare:

```text
Before training: random geometry
After training: learned class separation
```

Show class colors consistently:

- setosa: blue.
- versicolor: orange.
- virginica: green.

### Animation Flow

1. Title:

   ```text
   Training changes how Iris moves through vector spaces
   ```

2. Formula:

   ```text
   x -> W1x+b1 -> ReLU -> W2a+b2 -> logits
   ```

3. Input:

   Show all 150 standardized Iris samples in 2D PCA. Label the projection.

4. Before training:

   Copy the input points into the hidden projection generated from initial weights, then into the initial logit contrast projection. Label this path as random weights.

5. After training:

   Copy the same input points into the hidden projection generated from trained weights, then into the trained logit contrast projection. Label this path as trained weights.

6. Final comparison:

   Keep the before and after logit projections side by side. Show initial and trained accuracy values. Draw or label the class decision rule:

   ```text
   predict argmax(logits)
   ```

7. Closing message:

   ```text
   Learned matrices reshape 4D measurements into class-separable logits.
   ```

## Code Architecture

Add a new training module:

```text
src/deepmatrixanim/iris_training.py
```

Responsibilities:

- Load Iris with scikit-learn.
- Standardize features.
- Initialize deterministic MLP weights.
- Run NumPy forward propagation.
- Train with NumPy backpropagation.
- Build all projections needed by the scene.
- Return a single deterministic result object for tests and Manim.

Suggested dataclasses:

```python
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
```

Suggested functions and responsibilities:

- `load_standardized_iris() -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]`: load full Iris data and return standardized features, integer labels, and target names.
- `initialize_weights(seed: int = 7) -> MLPWeights`: create deterministic initial weights for the `4 -> 8 -> 3` MLP.
- `forward(x: np.ndarray, weights: MLPWeights) -> dict[str, np.ndarray]`: return `z1`, `a1`, `logits`, and `probabilities`.
- `train_mlp(x: np.ndarray, y: np.ndarray, initial: MLPWeights, epochs: int = 1200, learning_rate: float = 0.05) -> MLPWeights`: train with full-batch NumPy backpropagation and return trained weights.
- `build_iris_training_result() -> IrisTrainingResult`: assemble the deterministic before/after forward passes, projections, losses, and accuracies used by tests and Manim.

Add a new scene:

```text
scenes/iris_mlp_training_demo.py
```

The scene should consume `build_iris_training_result()` and avoid doing training math inside Manim-specific drawing helpers.

## Dependencies

Add scikit-learn to project dependencies:

```toml
scikit-learn>=1.5
```

The implementation may use:

- `sklearn.datasets.load_iris`
- `sklearn.preprocessing.StandardScaler`
- `sklearn.decomposition.PCA`

Do not add PyTorch or TensorFlow for this demo.

## Testing

Add:

```text
tests/test_iris_training.py
```

Required coverage:

- `load_standardized_iris()` returns `x.shape == (150, 4)`, `y.shape == (150,)`, and exactly three target names.
- Standardized features have approximately zero mean and unit variance.
- `initialize_weights()` returns deterministic shapes: `w1 == (8, 4)`, `b1 == (8,)`, `w2 == (3, 8)`, `b2 == (3,)`.
- `forward()` returns `z1 == (150, 8)`, `a1 == (150, 8)`, `logits == (150, 3)`, and `probabilities == (150, 3)`.
- Softmax probabilities sum to 1 row-wise.
- Training lowers loss from initial to trained weights.
- Training reaches a deterministic accuracy threshold of at least `0.9`.
- All 2D projections in `IrisTrainingResult` have shape `(150, 2)`.
- Hidden before/after projections are produced from a shared PCA fit.
- Logit before/after projections use the fixed contrast projection and have shape `(150, 2)`.

## Verification

After implementation, run:

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) ".uv-cache"
uv run pytest -v
uv run manim -ql -s scenes/iris_mlp_training_demo.py IrisMLPBeforeAfterTraining
uv run manim -ql scenes/iris_mlp_training_demo.py IrisMLPBeforeAfterTraining
```

The static render should be checked visually for overlapping text, cropped labels, and excessive point clutter.

## Out Of Scope

Do not animate every training epoch in this first version. Do not add an interactive UI. Do not replace the existing XOR-style representation scene. Do not introduce 3D rendering. Do not add PyTorch, TensorFlow, or a scikit-learn MLP classifier.
