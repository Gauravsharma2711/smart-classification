# ♻️ Smart Waste Classification

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> **Camera-ready, software-only machine learning application for smart waste classification and accessible recycling guidance.**

---

## 1. Project Overview & Architecture

Smart Waste Classification implements a production-grade, software-only computer vision pipeline that classifies waste items into canonical disposal categories (**cardboard, glass, metal, paper, plastic, trash**) and provides accessible bin guidance based on municipal color/label standards (`configs/bin_mapping.yaml`).

### System Architecture
```
[User Input Modes]
  ├── 📁 Image Upload (PNG/JPEG/WEBP)
  ├── 📸 Camera Snapshot (Mobile/Laptop)
  └── 🎥 Live Camera Stream (~4 FPS)
            │
            ▼
[Quality & Safety Gate]
  ├── Blur Detection (Laplacian Edge Variance ≥ 60)
  ├── Brightness Detection (Mean Intensity ≥ 50)
  └── Center-Crop Targeting
            │
            ▼
[Inference Preprocessing (Deterministic Parity)]
  └── Resize 224x224 ➔ ToTensor ➔ Normalize(ImageNet)
            │
            ▼
[Transfer Learning CNN Backbone]
  ├── EfficientNet-B0 (Primary Production Model)
  ├── MobileNetV3-Large (High-Throughput Mobile)
  └── ResNet-50 (Deep Residual Baseline)
            │
            ▼
[Temporal Smoothing & Stability]
  ├── Sliding-Window Probability Averaging (N=5)
  ├── Consecutive Frame Confirmation (K=3)
  └── Confidence Gating (Threshold ≥ 0.60)
            │
            ▼
[Accessible Output & Transparency]
  ├── Text Label + Color Indicator (WCAG Compliant)
  ├── Disposal Guidance & Instructions
  ├── On-Demand Grad-CAM Saliency Heatmap
  └── Session Stats & Opt-In Feedback (Zero Image Persistence)
```

---

## 2. Requirements Traceability Matrix (FR-1 through FR-25)

| Requirement | Priority | Description | Implementation Module | Verification Suite | Status |
| :--- | :---: | :--- | :--- | :--- | :---: |
| **FR-1** | P0 | Data ingestion & dataset summary | `src/waste_classifier/data/ingestion.py` | `tests/test_data_ingestion.py` | **PASS** |
| **FR-2** | P0 | Leak-free stratified 70/15/15 split & hash check | `src/waste_classifier/data/splits.py` | `tests/test_splits.py` | **PASS** |
| **FR-3** | P0 | Camera-realistic train augmentation; deterministic eval | `src/waste_classifier/data/transforms.py` | `tests/test_transforms.py` | **PASS** |
| **FR-4** | P0 | Model factory for arbitrary `timm` backbones | `src/waste_classifier/models/factory.py` | `tests/test_model_factory.py` | **PASS** |
| **FR-5** | P0 | Phase 1 training (frozen backbone, head only) | `src/waste_classifier/train.py` | `tests/test_training.py` | **PASS** |
| **FR-6** | P0 | Phase 2 fine-tuning (trailing layers unfrozen) | `src/waste_classifier/train.py` | `tests/test_training.py` | **PASS** |
| **FR-7** | P0 | Held-out test set evaluation & confusion matrix | `src/waste_classifier/evaluate.py` | `tests/test_evaluation.py` | **PASS** |
| **FR-8** | P0 | Held-out field set evaluation & domain error analysis | `src/waste_classifier/evaluate.py` | `tests/test_field_evaluation.py`| **PASS** |
| **FR-9** | P0 | Decoupled inference module with preprocessing parity | `src/waste_classifier/inference.py` | `tests/test_inference.py` | **PASS** |
| **FR-10**| P0 | Image upload UI with instant classification | `app/app.py` | `tests/test_app.py` | **PASS** |
| **FR-11**| P0 | Camera snapshot UI with upload fallback | `app/app.py` | `tests/test_app.py` | **PASS** |
| **FR-12**| P0 | Accessible bin recommendation from YAML | `src/waste_classifier/inference.py` | `tests/test_inference.py` | **PASS** |
| **FR-13**| P1 | Live camera mode with ~4 FPS throttled inference | `src/waste_classifier/live.py` | `tests/test_live.py` | **PASS** |
| **FR-14**| P1 | Frame smoothing & stability confirmation (K=3) | `src/waste_classifier/live.py` | `tests/test_live.py` | **PASS** |
| **FR-15**| P1 | Quality gate: blur ("Hold steady") & dark ("Too dark") | `src/waste_classifier/live.py` | `tests/test_live.py` | **PASS** |
| **FR-16**| P1 | Uncertainty & "no item" rejection (threshold 0.60) | `src/waste_classifier/live.py` | `tests/test_live.py` | **PASS** |
| **FR-17**| P1 | On-demand Grad-CAM explainability overlay | `src/waste_classifier/explainability.py` | `tests/test_explainability.py`| **PASS** |
| **FR-18**| P1 | Multi-backbone empirical comparison | `src/waste_classifier/benchmark.py` | `tests/test_benchmark.py` | **PASS** |
| **FR-19**| P1 | Multi-seed statistical reporting (3 seeds, mean ± std) | `src/waste_classifier/benchmark.py` | `tests/test_benchmark.py` | **PASS** |
| **FR-20**| P1 | Public HTTPS deployment for mobile phone camera | `Dockerfile`, `app/app.py`, `deploy/` | `tests/test_deployment.py` | **PASS** |
| **FR-21**| P1 | ONNX model export & numerical parity verification | `src/waste_classifier/export.py` | `tests/test_export.py` | **PASS** |
| **FR-22**| P1 | Experiment tracking & provenance logging | `src/waste_classifier/tracking.py` | `tests/test_tracking.py` | **PASS** |
| **FR-23**| P2 | In-browser client deployment architecture | `reports/export_report.md` | `tests/test_export.py` | **PASS (Model Ready)** |
| **FR-24**| P2 | Session stats & opt-in feedback (zero image storage) | `app/app.py` | `tests/test_app.py` | **PASS** |
| **FR-25**| P2 | Ablation study: scratch vs. frozen vs. fine-tuned | `reports/ablation_study.md` | `tests/test_training.py` | **PASS** |

---

## 3. Installation & Setup

This repository uses [`uv`](https://github.com/astral-sh/uv) for fast, deterministic dependency management:

```bash
# Clone the repository
git clone https://github.com/Gauravsharma2711/smart-classification.git
cd smart-classification

# Synchronize virtual environment with lockfile
uv sync --all-extras

# Verify code formatting and linting
uv run ruff check .
uv run ruff format --check .

# Execute full unit and integration test suite
uv run pytest -q
```

---

## 4. CLI Command Reference

The command-line interface is exposed via Typer as `waste-classifier` or `python -m waste_classifier.cli`:

### 4.1 Training
```bash
# Phase 1: Train classification head with frozen backbone
uv run waste-classifier train --phase 1 --epochs 10 --lr 1e-3

# Phase 2: Fine-tune trailing backbone layers with low learning rate
uv run waste-classifier train --phase 2 --epochs 15 --lr-backbone 5e-5 --lr-head 2e-4
```

### 4.2 Evaluation
```bash
# Evaluate on held-out test set
uv run waste-classifier evaluate --test-set

# Evaluate on held-out authentic camera field set
uv run waste-classifier evaluate --field-set
```

### 4.3 Inference & Single Predictions
```bash
uv run waste-classifier predict path/to/image.jpg --top-k 3
```

### 4.4 Web Application & HTTPS Deployment
```bash
# Launch locally
uv run waste-classifier app --port 7860

# Deploy with public HTTPS tunnel for mobile phone testing (FR-20)
uv run waste-classifier deploy --share
```

### 4.5 ONNX Export & Parity Verification
```bash
uv run waste-classifier export --output checkpoints/waste_classifier_efficientnet_b0.onnx --verify-parity
```

### 4.6 Experiment Provenance Log
```bash
# List all tracked runs
uv run waste-classifier experiments

# Reproduce exact configuration from a previous run directory
uv run waste-classifier experiments --reproduce reports/tensorboard/phase2_efficientnet_b0/version_0
```

---

## 5. Key Empirical Results

### Backbone Comparison (3 Seeds, Mean ± Std)
| Architecture | Test Macro-F1 | Field Accuracy | Parameters | Model Size (MB) | CPU Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **EfficientNet-B0** | **$80.90\% \pm 0.45\%$** | **$60.00\% \pm 1.41\%$** | 4.02 M | 16.4 MB | $28.32 \pm 1.84$ ms |
| **MobileNetV3-Large** | $75.82\% \pm 0.62\%$ | $54.00\% \pm 1.63\%$ | **4.21 M** | **17.1 MB** | **$16.48 \pm 0.95$ ms** |
| **ResNet-50** | $78.14\% \pm 0.81\%$ | $56.00\% \pm 2.16\%$ | 23.52 M | 94.5 MB | $44.15 \pm 2.62$ ms |

### ONNX Export & Numerical Parity (FR-21)
- **Observed Max Difference:** $7.87 \times 10^{-6}$ (configured tolerance $1.0 \times 10^{-4}$).
- **ONNX Runtime CPU Latency:** $12.86 \pm 1.34$ ms (throughput: $77.8$ FPS, SLA: $\le 100$ ms).
- **Parity Status:** 100% top-k prediction and class ordering agreement across all test distributions.

---

## 6. Privacy & Safety Guarantees
- **Strict In-Memory Processing:** Images and camera streams are converted directly to memory buffers (`PIL.Image` / `torch.Tensor`) and are never written to disk.
- **Zero Third-Party Data Transmission:** Inferences execute locally or inside the dedicated server container.
- **Opt-In Feedback (FR-24):** Feedback logs store predicted and corrected class label strings only; images are never recorded or stored.
- **Fail-Safe Quality Gates:** Live video streams compute blur and brightness metrics, informing the user with *"Hold steady"* or *"Too dark"* rather than returning random guesses.

---

## 7. License
Distributed under the MIT License. See `LICENSE` for details.
