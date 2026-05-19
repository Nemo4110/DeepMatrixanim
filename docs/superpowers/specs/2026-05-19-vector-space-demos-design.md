# Vector Space Demos Design

## Goal

The next DeepMatrixanim demos should move from numeric tracing toward a vector-space view of deep learning. The audience should see a layer as an operation on a space, not only as a table of numbers being multiplied.

This design adds two demos after the first MLP numeric trace:

1. `LinearLayerAsSpaceTransform`: show `z = Wx + b` as a linear transformation followed by a translation.
2. `ReLUAsSpaceFolding`: show `a = ReLU(z)` as a nonlinear folding/clamping operation that changes the geometry of the space.

## Audience And Teaching Style

The intended viewer knows basic vectors and matrices, but may not yet feel what a neural network layer does geometrically. The demos should use 2D space first, with visible grids, arrows, points, and motion. Formulas stay on screen as anchors, but the main explanation comes from spatial change.

The style should be closer to a visual proof than a calculation walkthrough: few numbers, clear geometric motion, and a direct connection back to deep learning terminology.

## Demo 2: Linear Layer As Space Transform

`LinearLayerAsSpaceTransform` explains the affine layer:

```text
z = Wx + b
```

The scene should show a 2D coordinate grid with one highlighted input vector `x` and a small set of reference points. A matrix `W` then transforms the grid by stretching, rotating, or shearing it. The highlighted vector follows the grid to become `Wx`. After that, a bias vector `b` shifts the transformed vector and reference points to `Wx + b`.

The final message should be:

```text
Linear layer = reshape directions + shift origin
```

This scene should prioritize readability over visual cleverness. A simple 2x2 matrix is enough if the deformation is obvious.

## Demo 3: ReLU As Space Folding

`ReLUAsSpaceFolding` explains the coordinate-wise activation:

```text
a = ReLU(z) = max(0, z)
```

The scene should reuse the idea of a 2D plane, then show points or vectors moving through coordinate-wise ReLU. Points with negative x or y components should clamp onto the corresponding axis, while points in the first quadrant remain unchanged.

The key visual claim is that ReLU is not another linear transformation. It folds or collapses regions of the space, which is why repeated `Linear + ReLU` layers can form piecewise-linear geometry.

The final message should be:

```text
MLP = repeated reshape + fold
```

## Architecture

Keep math and scene construction separate.

- `src/deepmatrixanim/linear_maps.py`: small deterministic helpers for 2D matrices, vectors, affine transforms, point clouds, and coordinate-wise ReLU.
- `tests/test_linear_maps.py`: pytest coverage for affine and ReLU helper functions.
- `scenes/linear_layer_space_demo.py`: Manim scene for `LinearLayerAsSpaceTransform`.
- `scenes/relu_space_folding_demo.py`: Manim scene for `ReLUAsSpaceFolding`.

The existing `scenes/mlp_forward_demo.py` remains as the numeric tracing demo. These new scenes form a second line of explanation called the space view.

## Manim Approach

For the linear layer scene, prefer Manim primitives that make the grid transformation clear. `LinearTransformationScene` is a good candidate for the `W` step, but the implementation should verify that its lifecycle works cleanly with the project existing render command. If it makes composition awkward, use `NumberPlane`, `Vector`, `Dot`, and manually animated point transforms instead.

For the ReLU scene, use manually animated point/vector transforms. ReLU is not linear, so `LinearTransformationScene` is the wrong abstraction for the core transformation.

## Error Handling And Constraints

The demos should use fixed matrices and point sets. They do not need runtime configuration, CLI arguments, or random data. Deterministic input keeps tests simple and rendered output stable.

Scenes should fit in Manim low-quality 16:9 output without cropped text. The previous layout issue should be avoided by grouping main content and applying a single safe scale factor with maximum width and height constraints.

## Testing And Verification

Test math helpers with pytest:

```powershell
uv run pytest -v
```

Verify scene import and basic render with low-quality static renders before full video renders:

```powershell
uv run manim -ql -s scenes/linear_layer_space_demo.py LinearLayerAsSpaceTransform
uv run manim -ql -s scenes/relu_space_folding_demo.py ReLUAsSpaceFolding
```

Then render videos:

```powershell
uv run manim -ql scenes/linear_layer_space_demo.py LinearLayerAsSpaceTransform
uv run manim -ql scenes/relu_space_folding_demo.py ReLUAsSpaceFolding
```

## Out Of Scope

Do not introduce 3D scenes yet. Do not make the demos configurable from JSON or YAML in this iteration. Do not replace the existing numeric MLP demo. Do not attempt attention, embeddings, normalization, convolution, or backprop in this spec; those belong to later specs once the space-view visual language is working.
