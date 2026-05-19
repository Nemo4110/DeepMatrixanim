# Vector Space Demos Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two vector-space Manim demos that show a linear layer as `Wx + b` space transformation and ReLU as coordinate-wise space folding.

**Architecture:** Keep deterministic 2D math helpers in `src/deepmatrixanim/linear_maps.py`, with pytest coverage in `tests/test_linear_maps.py`. Add one Manim scene for the affine space transform and one for ReLU folding, reusing the existing project style of low-quality render verification before full video output.

**Tech Stack:** Python 3.12, uv, NumPy, pytest, Manim Community, MiKTeX, FFmpeg.

---

## File Structure

- Create `src/deepmatrixanim/linear_maps.py`: deterministic matrices, vectors, point clouds, affine transform helper, and coordinate-wise ReLU helper.
- Create `tests/test_linear_maps.py`: numeric tests for every helper used by the scenes.
- Create `scenes/linear_layer_space_demo.py`: `LinearLayerAsSpaceTransform` Manim scene.
- Create `scenes/relu_space_folding_demo.py`: `ReLUAsSpaceFolding` Manim scene.
- Modify `README.md`: add commands for the two new scenes.

### Task 1: Linear Map Math Helpers

**Files:**
- Create: `tests/test_linear_maps.py`
- Create: `src/deepmatrixanim/linear_maps.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_linear_maps.py` with:

```python
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
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) ".uv-cache"
uv run pytest tests/test_linear_maps.py -v
```

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'deepmatrixanim.linear_maps'`.

- [ ] **Step 3: Implement the minimal helpers**

Create `src/deepmatrixanim/linear_maps.py` with:

```python
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
```

- [ ] **Step 4: Run the tests and verify GREEN**

Run:

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) ".uv-cache"
uv run pytest tests/test_linear_maps.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 5: Run the full test suite**

Run:

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) ".uv-cache"
uv run pytest -v
```

Expected: existing MLP test and new linear map tests pass.

- [ ] **Step 6: Commit math helpers**

Run:

```powershell
git add src/deepmatrixanim/linear_maps.py tests/test_linear_maps.py
git commit -m "feat: add vector space math helpers"
```

### Task 2: Linear Layer Space Transform Scene

**Files:**
- Create: `scenes/linear_layer_space_demo.py`
- Use: `src/deepmatrixanim/linear_maps.py`

- [ ] **Step 1: Create the scene file**

Create `scenes/linear_layer_space_demo.py` with:

```python
from __future__ import annotations

import numpy as np
from manim import (
    BLUE,
    DOWN,
    GREEN,
    GREY_B,
    ORANGE,
    PURPLE,
    RIGHT,
    UP,
    YELLOW,
    Arrow,
    Create,
    Dot,
    FadeIn,
    MathTex,
    NumberPlane,
    Scene,
    Tex,
    Transform,
    VGroup,
    Write,
)

from deepmatrixanim.linear_maps import (
    affine_transform,
    demo_bias_vector,
    demo_input_vector,
    demo_linear_matrix,
    demo_point_cloud,
)


def point_to_scene(point: np.ndarray) -> np.ndarray:
    return np.array([point[0], point[1], 0.0])


def vector_arrow(point: np.ndarray, color=BLUE) -> Arrow:
    return Arrow(start=np.zeros(3), end=point_to_scene(point), buff=0, color=color)


def dots_for(points: np.ndarray, color=YELLOW) -> VGroup:
    return VGroup(*[Dot(point_to_scene(point), radius=0.055, color=color) for point in points])


class LinearLayerAsSpaceTransform(Scene):
    def construct(self):
        matrix = demo_linear_matrix()
        bias = demo_bias_vector()
        x = demo_input_vector()
        wx = affine_transform(x, matrix, np.zeros(2))
        z = affine_transform(x, matrix, bias)
        points = demo_point_cloud()
        linear_points = affine_transform(points, matrix, np.zeros(2))
        affine_points = affine_transform(points, matrix, bias)

        title = Tex("Linear layer as a space transform").scale(0.75).to_edge(UP)
        formula = MathTex(r"z = Wx + b").scale(0.85).next_to(title, DOWN, buff=0.16)
        note = Tex("reshape directions, then shift origin").scale(0.45).next_to(formula, DOWN, buff=0.12)

        plane = NumberPlane(
            x_range=[-3.5, 4.5, 1],
            y_range=[-3.0, 3.0, 1],
            x_length=8.0,
            y_length=5.2,
            background_line_style={"stroke_color": GREY_B, "stroke_width": 1, "stroke_opacity": 0.45},
        ).shift(DOWN * 0.45)

        plane_center = plane.get_center()
        original_vector = vector_arrow(x, BLUE).shift(plane_center)
        linear_vector = vector_arrow(wx, GREEN).shift(plane_center)
        affine_vector = Arrow(
            start=point_to_scene(bias) + plane_center,
            end=point_to_scene(z) + plane_center,
            buff=0,
            color=PURPLE,
        )
        bias_arrow = vector_arrow(bias, ORANGE).shift(plane_center)

        original_dots = dots_for(points, YELLOW).shift(plane_center)
        linear_dots = dots_for(linear_points, GREEN).shift(plane_center)
        affine_dots = dots_for(affine_points, PURPLE).shift(plane_center)

        w_label = MathTex(r"W = \begin{bmatrix}1.2 & 0.6 \\ -0.4 & 1.1\end{bmatrix}").scale(0.48)
        b_label = MathTex(r"b = \begin{bmatrix}0.8 \\ -0.35\end{bmatrix}").scale(0.48)
        VGroup(w_label, b_label).arrange(RIGHT, buff=0.55).to_edge(DOWN)

        self.play(Write(title), Write(formula), FadeIn(note))
        self.play(Create(plane), FadeIn(original_dots), Create(original_vector), FadeIn(w_label))
        self.wait(0.4)
        self.play(Transform(original_dots, linear_dots), Transform(original_vector, linear_vector), run_time=1.6)
        self.wait(0.4)
        self.play(Create(bias_arrow), FadeIn(b_label))
        self.play(Transform(original_dots, affine_dots), Transform(original_vector, affine_vector), run_time=1.3)
        self.wait(0.6)
```

- [ ] **Step 2: Run a static render smoke test**

Run:

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) ".uv-cache"
$env:Path = "C:\Users\Administrator\AppData\Local\Programs\MiKTeX\miktex\bin\x64;C:\Users\Administrator\AppData\Local\Microsoft\WinGet\Links;" + $env:Path
uv run manim -ql -s scenes/linear_layer_space_demo.py LinearLayerAsSpaceTransform
```

Expected: exit code 0 and a PNG under `media/images/linear_layer_space_demo/`.

- [ ] **Step 3: Render the video**

Run:

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) ".uv-cache"
$env:Path = "C:\Users\Administrator\AppData\Local\Programs\MiKTeX\miktex\bin\x64;C:\Users\Administrator\AppData\Local\Microsoft\WinGet\Links;" + $env:Path
uv run manim -ql scenes/linear_layer_space_demo.py LinearLayerAsSpaceTransform
```

Expected: exit code 0 and video at `media/videos/linear_layer_space_demo/480p15/LinearLayerAsSpaceTransform.mp4`.

- [ ] **Step 4: Commit the linear layer scene**

Run:

```powershell
git add scenes/linear_layer_space_demo.py
git commit -m "feat: add linear layer space transform scene"
```

### Task 3: ReLU Space Folding Scene

**Files:**
- Create: `scenes/relu_space_folding_demo.py`
- Use: `src/deepmatrixanim/linear_maps.py`

- [ ] **Step 1: Create the scene file**

Create `scenes/relu_space_folding_demo.py` with:

```python
from __future__ import annotations

import numpy as np
from manim import (
    BLUE,
    DOWN,
    GREEN,
    GREY_B,
    LEFT,
    ORANGE,
    RED,
    UP,
    YELLOW,
    Arrow,
    Create,
    Dot,
    FadeIn,
    MathTex,
    NumberPlane,
    Scene,
    Tex,
    Transform,
    VGroup,
    Write,
)

from deepmatrixanim.linear_maps import demo_point_cloud, relu_transform


def point_to_scene(point: np.ndarray) -> np.ndarray:
    return np.array([point[0], point[1], 0.0])


def dots_for(points: np.ndarray, color=YELLOW) -> VGroup:
    return VGroup(*[Dot(point_to_scene(point), radius=0.06, color=color) for point in points])


class ReLUAsSpaceFolding(Scene):
    def construct(self):
        points = demo_point_cloud()
        relu_points = relu_transform(points)

        title = Tex("ReLU as space folding").scale(0.78).to_edge(UP)
        formula = MathTex(r"a = \mathrm{ReLU}(z) = \max(0, z)").scale(0.72).next_to(title, DOWN, buff=0.16)
        message = Tex("negative coordinates collapse onto the axes").scale(0.45).next_to(formula, DOWN, buff=0.12)

        plane = NumberPlane(
            x_range=[-3.0, 3.8, 1],
            y_range=[-2.8, 3.2, 1],
            x_length=7.6,
            y_length=5.1,
            background_line_style={"stroke_color": GREY_B, "stroke_width": 1, "stroke_opacity": 0.45},
        ).shift(DOWN * 0.45)

        plane_center = plane.get_center()
        original_dots = dots_for(points, YELLOW).shift(plane_center)
        folded_dots = dots_for(relu_points, GREEN).shift(plane_center)

        x_axis_arrow = Arrow(
            start=point_to_scene([-2.6, 0.0]) + plane_center,
            end=point_to_scene([3.3, 0.0]) + plane_center,
            buff=0,
            color=BLUE,
        )
        y_axis_arrow = Arrow(
            start=point_to_scene([0.0, -2.4]) + plane_center,
            end=point_to_scene([0.0, 2.8]) + plane_center,
            buff=0,
            color=BLUE,
        )
        clamp_x = MathTex(r"x < 0 \Rightarrow x = 0").set_color(RED).scale(0.48).to_corner(LEFT + DOWN)
        clamp_y = MathTex(r"y < 0 \Rightarrow y = 0").set_color(ORANGE).scale(0.48).next_to(clamp_x, UP, buff=0.18)
        final = Tex("MLP = repeated reshape + fold").scale(0.55).to_edge(DOWN)

        self.play(Write(title), Write(formula), FadeIn(message))
        self.play(Create(plane), Create(x_axis_arrow), Create(y_axis_arrow), FadeIn(original_dots))
        self.wait(0.5)
        self.play(FadeIn(clamp_x), FadeIn(clamp_y))
        self.play(Transform(original_dots, folded_dots), run_time=1.8)
        self.wait(0.5)
        self.play(Write(final))
        self.wait(1.0)
```

- [ ] **Step 2: Run a static render smoke test**

Run:

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) ".uv-cache"
$env:Path = "C:\Users\Administrator\AppData\Local\Programs\MiKTeX\miktex\bin\x64;C:\Users\Administrator\AppData\Local\Microsoft\WinGet\Links;" + $env:Path
uv run manim -ql -s scenes/relu_space_folding_demo.py ReLUAsSpaceFolding
```

Expected: exit code 0 and a PNG under `media/images/relu_space_folding_demo/`.

- [ ] **Step 3: Render the video**

Run:

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) ".uv-cache"
$env:Path = "C:\Users\Administrator\AppData\Local\Programs\MiKTeX\miktex\bin\x64;C:\Users\Administrator\AppData\Local\Microsoft\WinGet\Links;" + $env:Path
uv run manim -ql scenes/relu_space_folding_demo.py ReLUAsSpaceFolding
```

Expected: exit code 0 and video at `media/videos/relu_space_folding_demo/480p15/ReLUAsSpaceFolding.mp4`.

- [ ] **Step 4: Commit the ReLU scene**

Run:

```powershell
git add scenes/relu_space_folding_demo.py
git commit -m "feat: add relu space folding scene"
```

### Task 4: README And Final Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README render commands**

Modify `README.md` so the render section contains commands for all three scenes:

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) ".uv-cache"
uv run manim -ql scenes/mlp_forward_demo.py MLPForwardDemo
uv run manim -ql scenes/linear_layer_space_demo.py LinearLayerAsSpaceTransform
uv run manim -ql scenes/relu_space_folding_demo.py ReLUAsSpaceFolding
```

Keep the note that Manim writes rendered videos under `media/videos/`.

- [ ] **Step 2: Run full tests**

Run:

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) ".uv-cache"
uv run pytest -v
```

Expected: all tests pass.

- [ ] **Step 3: Verify Manim can import all scenes with static renders**

Run:

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) ".uv-cache"
$env:Path = "C:\Users\Administrator\AppData\Local\Programs\MiKTeX\miktex\bin\x64;C:\Users\Administrator\AppData\Local\Microsoft\WinGet\Links;" + $env:Path
uv run manim -ql -s scenes/mlp_forward_demo.py MLPForwardDemo
uv run manim -ql -s scenes/linear_layer_space_demo.py LinearLayerAsSpaceTransform
uv run manim -ql -s scenes/relu_space_folding_demo.py ReLUAsSpaceFolding
```

Expected: all three static render commands exit 0.

- [ ] **Step 4: Commit documentation**

Run:

```powershell
git add README.md
git commit -m "docs: add vector space demo render commands"
```

- [ ] **Step 5: Check final git status**

Run:

```powershell
git status --short
```

Expected: no uncommitted tracked changes. Ignored `media/`, `.venv/`, `.uv-cache/`, `.pytest_cache/`, and `__pycache__/` may exist locally.

