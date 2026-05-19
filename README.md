# DeepMatrixanim

Animations for understanding deep learning through matrix and vector operations.

## Prerequisites

- Windows 11
- [uv](https://docs.astral.sh/uv/)
- MiKTeX with `latex`, `pdflatex`, and `dvisvgm` available on `PATH`
- FFmpeg with `ffmpeg` available on `PATH`

## Setup

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) ".uv-cache"
uv sync --group dev
```

## Verify

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) ".uv-cache"
uv run pytest -v
uv run manim --version
```

## Render The Demos

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) ".uv-cache"
uv run manim -ql scenes/mlp_forward_demo.py MLPForwardDemo
uv run manim -ql scenes/linear_layer_space_demo.py LinearLayerAsSpaceTransform
uv run manim -ql scenes/relu_space_folding_demo.py ReLUAsSpaceFolding
uv run manim -ql scenes/representation_space_demo.py RepresentationSpaceDecisionBoundary
```

Manim writes rendered videos under `media/videos/`.
