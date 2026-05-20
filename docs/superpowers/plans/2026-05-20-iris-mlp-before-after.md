# Iris MLP Before/After Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Manim demo that trains a transparent NumPy MLP on the full Iris dataset and visualizes before/after vector-space transformations with explicit 2D projections.

**Architecture:** Add `src/deepmatrixanim/iris_training.py` for scikit-learn Iris loading, NumPy MLP training, and projection assembly. Add `tests/test_iris_training.py` for data, forward-pass, training, and projection behavior. Add `scenes/iris_mlp_training_demo.py` for the Manim scene and update README render commands.

**Tech Stack:** Python 3.12, NumPy, scikit-learn, pytest, Manim Community, uv.

---

## File Structure

- Modify `pyproject.toml` and `uv.lock`: add `scikit-learn`.
- Create `tests/test_iris_training.py`: TDD coverage for full Iris loading, MLP shapes, training improvement, and projections.
- Create `src/deepmatrixanim/iris_training.py`: deterministic data, training, metrics, and projections.
- Create `scenes/iris_mlp_training_demo.py`: before/after training visualization.
- Modify `README.md`: add Manim render command.

## Task 1: Dependency And Failing Iris Tests

- [ ] Add tests in `tests/test_iris_training.py` that import the intended API:
  - `load_standardized_iris`
  - `initialize_weights`
  - `forward`
  - `cross_entropy_loss`
  - `accuracy`
  - `train_mlp`
  - `logit_contrast_projection`
  - `build_iris_training_result`
- [ ] Run `uv run pytest tests/test_iris_training.py -v` and verify RED from missing `deepmatrixanim.iris_training`.
- [ ] Add `scikit-learn>=1.5` with `uv add scikit-learn`.

## Task 2: Iris Data And Forward Pass

- [ ] Implement dataclasses `MLPWeights` and `IrisTrainingResult`.
- [ ] Implement `load_standardized_iris()` using `load_iris()` and `StandardScaler`.
- [ ] Implement deterministic `initialize_weights(seed=7)` for a `4 -> 8 -> 3` MLP.
- [ ] Implement `forward()` with ReLU and stable softmax.
- [ ] Implement `cross_entropy_loss()` and `accuracy()`.
- [ ] Run `uv run pytest tests/test_iris_training.py -v` and verify the data/forward tests pass while training/projection tests still drive remaining work.

## Task 3: Training And Projections

- [ ] Implement full-batch NumPy backprop in `train_mlp()`.
- [ ] Implement input PCA, shared hidden PCA, and fixed logit contrast projection.
- [ ] Implement `build_iris_training_result()`.
- [ ] Run `uv run pytest tests/test_iris_training.py -v` and verify all Iris tests pass.
- [ ] Run `uv run pytest -v` and verify the full suite passes.

## Task 4: Manim Scene

- [ ] Create `scenes/iris_mlp_training_demo.py` with scene class `IrisMLPBeforeAfterTraining`.
- [ ] Consume `build_iris_training_result()` rather than training inside drawing helpers.
- [ ] Show full Iris input PCA, random-weight hidden/logit path, trained-weight hidden/logit path, and final before/after logit comparison with accuracies.
- [ ] Use three stable class colors: blue, orange, green.
- [ ] Run `uv run manim -ql -s scenes/iris_mlp_training_demo.py IrisMLPBeforeAfterTraining` and inspect the PNG for layout issues.

## Task 5: Documentation And Verification

- [ ] Add README render command:
  `uv run manim -ql scenes/iris_mlp_training_demo.py IrisMLPBeforeAfterTraining`
- [ ] Run `uv run pytest -v`.
- [ ] Run `uv run manim -ql scenes/iris_mlp_training_demo.py IrisMLPBeforeAfterTraining`.
- [ ] Check `git status --short` and prepare a concise commit.
