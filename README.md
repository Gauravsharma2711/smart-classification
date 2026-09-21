# Smart Waste Classification: Software-Only Camera and Upload App

An agent-first, software-only machine learning application for smart waste classification using device cameras and image uploads.

## Project Architecture & Methodology
- **Source of Truth**: `PRD_software_only.md`
- **Method**: Two-phase transfer learning with pretrained CNN backbones (`efficientnet_b0`, `mobilenetv3_large_100`, `resnet50`).
- **Input Modes**:
  1. Image upload (P0)
  2. Camera snapshot (P0)
  3. Live camera streaming with frame smoothing and quality gates (P1)
- **Key Rigor Rules**:
  - Leak-free splits with perceptual hash deduplication (`imagehash`).
  - Camera-realistic augmentation on train split only; deterministic validation and test transforms.
  - Held-out field set evaluation for camera-domain testing.
  - Parity test verifying inference preprocessing matches evaluation pipeline.
  - Zero storage of camera frames or uploaded images by default.

## Setup & Quick Start

```bash
# Sync dependencies and lockfile
uv sync --all-extras

# Run linter and formatter checks
uv run ruff check .
uv run ruff format --check .

# Run test suite
uv run pytest -q
```
