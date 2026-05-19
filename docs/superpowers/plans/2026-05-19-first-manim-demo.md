# First Manim Demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first runnable DeepMatrixanim demo: a Manim Community scene that traces one fixed two-layer MLP forward pass with LaTeX labels.

**Architecture:** Keep numeric computation in `src/deepmatrixanim/mlp.py` and visual presentation in `scenes/mlp_forward_demo.py`. Use `uv` for the project environment, Manim Community for rendering, MiKTeX for LaTeX, and FFmpeg for video encoding.

**Tech Stack:** Python 3.12, uv, Manim Community, NumPy, pytest, MiKTeX, FFmpeg.

---

### Task 1: Project Environment

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`

- [ ] Create project metadata with Python 3.12, Manim, NumPy, and pytest dependencies.
- [ ] Run `uv sync --group dev` with `UV_CACHE_DIR=.uv-cache`.
- [ ] Verify `uv run python --version`, `uv run manim --version`, and `uv run pytest --version`.

### Task 2: MLP Math Core

**Files:**
- Create: `tests/test_mlp.py`
- Create: `src/deepmatrixanim/mlp.py`

- [ ] Write a failing test that asserts fixed matrices produce `z1`, `a1`, and `y`.
- [ ] Run `uv run pytest tests/test_mlp.py -v` and confirm failure due to missing implementation.
- [ ] Implement `demo_forward_pass()` with NumPy.
- [ ] Run `uv run pytest tests/test_mlp.py -v` and confirm it passes.

### Task 3: Manim Scene

**Files:**
- Create: `scenes/mlp_forward_demo.py`

- [ ] Create `MLPForwardDemo(Scene)` that imports `demo_forward_pass()`.
- [ ] Show formula `x \rightarrow z_1 = W_1x + b_1 \rightarrow a_1 = \mathrm{ReLU}(z_1) \rightarrow y = W_2a_1 + b_2`.
- [ ] Show the fixed numeric matrices and vectors.
- [ ] Highlight the negative hidden pre-activation and its ReLU output becoming zero.
- [ ] Render with `uv run manim -ql scenes/mlp_forward_demo.py MLPForwardDemo`.

### Task 4: Documentation

**Files:**
- Modify: `README.md`

- [ ] Document prerequisites, setup command, test command, and render command.
- [ ] Mention where Manim writes rendered media.
