# Smart Waste Classification: Software-Only Camera and Upload App
## Product Requirements Document (PRD) v2.0

| | |
|---|---|
| **Version** | 2.0 (draft). Supersedes `PRD.md` v1.0 |
| **Type** | Machine Learning course project (third-year, Data Science) |
| **Approach** | Two-phase fine-tuning of a pretrained CNN (feature extraction, then partial fine-tuning) |
| **Delivery** | 100% software: a web app that classifies waste from a **device camera** or an **uploaded image** |
| **Build tool** | Google Antigravity (agent-first IDE), driven by this PRD |
| **Status** | Draft. Revise targets after the first baseline run. |

**What changed from v1:** no hardware, edge-device, or Raspberry Pi work anywhere. The "smart" part is now the software experience: live camera classification, frame smoothing, quality hints, uncertainty handling, and instant bin advice. Camera input also makes real-world image quality a first-class requirement, so there is a dedicated camera-domain evaluation (the "field set").

---

## 1. Overview

### 1.1 Problem
People often do not know which bin an item belongs in, so recyclables end up in landfill. Anyone with a phone or laptop already has a camera, so no special hardware should be needed to help them.

### 1.2 Product vision
A fast, private, browser-based app that:
1. Lets the user **point a camera at an item** (snapshot or live mode) or **upload a photo**.
2. Predicts the waste category with a confidence score.
3. Tells the user which bin it goes in, and explains the decision with a Grad-CAM heatmap.
4. Says "not sure" when the image is unclear, instead of guessing.
5. Works on a laptop or a phone, with no hardware beyond the device's own camera.

### 1.3 Goals
- G1. A strong transfer-learning classifier using **two-phase fine-tuning**.
- G2. A robust model for **real camera images**, not just clean dataset photos (augmentation designed for camera conditions, plus an independent field-set evaluation).
- G3. A polished, fast UI with three input paths: upload, camera snapshot, live camera.
- G4. Rigorous methodology: leak-free splits, fixed seeds, per-class metrics, multi-seed results, error analysis.
- G5. Professional, tested, config-driven code, built with Antigravity under a written spec.
- G6. A publicly reachable HTTPS demo that works with a phone camera.

### 1.4 Non-goals
- Any hardware: Raspberry Pi, microcontrollers, servo motors, or a physical smart bin.
- Multi-object detection or segmentation.
- User accounts, databases, or a persistent cloud backend.
- Training a model from scratch (ablation only).

---

## 2. Users and use cases

| User | Need | Use case |
|---|---|---|
| **Everyday user (phone)** | "Which bin does this go in?" | Open the link, allow the camera, point at the item, read the answer |
| **Everyday user (laptop)** | Same, without a phone | Upload a saved photo or use the webcam |
| **Examiner / reviewer** | Evidence the method is sound | Read metrics, the confusion matrix, Grad-CAM, and the field-set results; reproduce a run |
| **Developer (you)** | Fast, safe iteration | One command to train, evaluate, export, or run the app |

---

## 3. Scope and priorities

- **P0 (must have):** data pipeline, leak-free split, Phase 1 and Phase 2 training, test-set evaluation, field-set evaluation, inference module, upload UI, camera snapshot UI, bin recommendation.
- **P1 (should have):** live camera mode with smoothing, image-quality hints, uncertainty and "no item" handling, Grad-CAM, backbone comparison, multi-seed results, ONNX export, experiment tracking, public HTTPS deployment.
- **P2 (nice to have):** React front end running the model fully in the browser (ONNX Runtime Web), session statistics, opt-in feedback button, ablation study, hyperparameter search.

---

## 4. Input modes

| Mode | Behavior | Priority |
|---|---|---|
| **Upload** | Drag and drop or pick an image; classify on submit | P0 |
| **Camera snapshot** | Capture one frame from the device camera; classify it | P0 |
| **Live camera** | Continuously classify camera frames, show a smoothed, stable result | P1 |

**Camera requirements**
- Request camera permission; if denied or no camera exists, show a clear message and fall back to upload.
- On phones, prefer the rear camera when the platform allows.
- Show a **guide box** so the user centers one item in the frame.
- Browsers only allow camera access on **HTTPS or localhost**. Testing on a phone over a plain local network address will fail, so deploy to an HTTPS host (see FR-20) to test the phone camera.

---

## 5. Functional requirements

| ID | Priority | Requirement | Acceptance criteria |
|---|---|---|---|
| FR-1 | P0 | **Data ingestion**: download or unpack the dataset, remove unreadable images, print class counts | Folder-per-class dataset plus a class-count table |
| FR-2 | P0 | **Leak-free split**: stratified 70/15/15 with a fixed seed; manifest CSV saved | A test asserts zero file overlap between splits; manifest committed |
| FR-3 | P0 | **Camera-realistic augmentation** on the train split only (see 8.3); resize to 224×224 with backbone-appropriate normalization | Val and test pipelines contain no random transforms (unit-tested) |
| FR-4 | P0 | **Model factory**: any `timm` backbone by name with a new head | Changing `model.backbone` in the config swaps the model with no code change |
| FR-5 | P0 | **Phase 1 training** (frozen backbone, head only) | Per-epoch loss and accuracy logged; best checkpoint saved |
| FR-6 | P0 | **Phase 2 fine-tuning** (last blocks unfrozen, low LR, early stopping) | Improves val macro-F1 over Phase 1, or is reported as a negative result |
| FR-7 | P0 | **Test-set evaluation**: accuracy, macro-F1, per-class precision, recall, F1, confusion matrix | `reports/` holds a metrics JSON, confusion-matrix PNG, and classification report |
| FR-8 | P0 | **Field-set evaluation**: evaluate on 100+ of your own camera photos (see 8.5) that are never used for training or tuning | Separate metrics table and a short error analysis in `reports/` |
| FR-9 | P0 | **Inference module**: `predict(image) -> top-k classes with probabilities` | Works on a PIL image; unit test; a parity test confirms inference preprocessing equals eval preprocessing |
| FR-10 | P0 | **Upload UI**: upload an image, see class, confidence, and bin | Runs locally with one command |
| FR-11 | P0 | **Camera snapshot UI**: capture a frame and classify it | Works on laptop webcam; permission-denied path falls back to upload |
| FR-12 | P0 | **Bin recommendation** from `configs/bin_mapping.yaml` | Each class maps to a bin label and color, with a text label (never color alone) |
| FR-13 | P1 | **Live camera mode** with throttled inference | Result updates at a usable rate without freezing the UI (see 9) |
| FR-14 | P1 | **Frame smoothing and stability**: temporal averaging and a "confirmed" state | Label does not flicker between frames (unit-tested on synthetic sequences) |
| FR-15 | P1 | **Quality hints**: detect blur and low light before predicting | User sees "Hold steady" or "Too dark" instead of a bad guess |
| FR-16 | P1 | **Uncertainty and "no item" handling**: confidence threshold, optional `background` class | Empty scenes, hands, or faces do not produce confident waste labels |
| FR-17 | P1 | **Grad-CAM**: overlay for a prediction | Shown on demand in the UI and saved for the report (correct and wrong examples) |
| FR-18 | P1 | **Backbone comparison**: EfficientNet-B0, MobileNetV3, ResNet50 under the same protocol | One table: macro-F1, field-set accuracy, params, model size, CPU latency |
| FR-19 | P1 | **Multi-seed reporting** (3 seeds) | Table reports mean ± std |
| FR-20 | P1 | **Public deployment**: a hosted HTTPS demo | The link works with a phone camera and with upload |
| FR-21 | P1 | **ONNX export** with a parity check against PyTorch | Max output difference below a set tolerance; latency measured with ONNX Runtime |
| FR-22 | P1 | **Experiment tracking**: config, metrics, and git commit logged per run | Runs are comparable in TensorBoard (or MLflow) |
| FR-23 | P2 | **Client-side app** (React) running the ONNX model in the browser | Works offline after load; no image ever leaves the device |
| FR-24 | P2 | **Session stats and feedback**: items classified per category; optional "was this correct?" | Feedback stores labels only, never images, unless the user opts in |
| FR-25 | P2 | **Ablation**: from scratch vs frozen vs fine-tuned | One table and a short analysis |

---

## 6. Non-functional requirements

- **Reproducibility:** seeds everywhere; dependencies locked with `uv.lock`; every run logs config and git commit.
- **Performance (proposed targets, measure and report):** model-only inference ≤ 100 ms on a laptop CPU with ONNX Runtime; live mode shows a stable label within about 1 second of the user holding an item steady.
- **Privacy:** camera frames are processed in memory and **not saved** by default; no third-party analytics; any stored feedback is opt-in (see section 10).
- **Code quality:** type hints, docstrings, `ruff` clean, tests for all non-training logic.
- **Portability:** training runs on Google Colab and locally with the same commands; the app runs on a laptop and is reachable from a phone browser once deployed.
- **Accessibility:** bin advice always includes text, not only color; camera failure never blocks use because upload always works.
- **Licensing:** record dataset sources and licenses in the README and report.

---

## 7. Data requirements

### 7.1 Candidate datasets (verify availability and license at download time)
| Dataset | Approx. size and classes | Notes |
|---|---|---|
| **TrashNet** | ~2.5k images, 6 classes (glass, paper, cardboard, plastic, metal, trash) | Small and classic. Good for fast iteration. |
| **Kaggle "Garbage Classification"** | ~15k images, 12 classes | More data and classes, more confusion between similar items. |
| **Organic vs recyclable (Kaggle)** | ~25k images, 2 classes | Simplest fallback for a binary version. |

**Decision for v1:** start with TrashNet, move to the 12-class set if time allows, and report both.

### 7.2 Data rules
- Split **before** augmentation; check for near-duplicates across splits with perceptual hashing.
- Handle imbalance with class weights or a weighted sampler; report **macro-F1** alongside accuracy.
- Never commit raw data or checkpoints. Commit only the split manifest and scripts.
- The **field set** (8.5) is a separate, held-out evaluation set. It is never used for training or tuning in the main results.

---

## 8. ML approach

### 8.1 Method: two-phase transfer learning
1. **Phase 1, feature extraction:** ImageNet-pretrained backbone frozen; train only the new head.
2. **Phase 2, fine-tuning:** unfreeze the last blocks (roughly the last 20-30% of the network) and continue at a much lower learning rate.

### 8.2 Proposed defaults (all in `configs/`; tune on the validation set)
| Setting | Value |
|---|---|
| Primary backbone | `efficientnet_b0` (via `timm`) |
| Comparison backbones | `mobilenetv3_large_100` (the lightest, best for browser use), `resnet50` |
| Input size | 224×224 |
| Head | Global average pooling, dropout 0.3, linear layer |
| Loss | Cross-entropy, label smoothing 0.1, class weights |
| Optimizer | AdamW |
| Phase 1 | LR 1e-3, ~10 epochs |
| Phase 2 | backbone LR ~1e-5, head LR ~1e-4, cosine schedule, early stopping (patience 5), up to ~15 epochs |
| BatchNorm | Eval mode during Phase 2 (small batches, small dataset) |
| Precision | Mixed precision on GPU |
| Seeds | 3 seeds for final comparisons |

### 8.3 Camera-realistic augmentation (train split only)
Dataset photos are clean; camera frames are not. Simulate the difference with Albumentations:
- Random resized crop (scale about 0.6-1.0), rotation (about ±25°), and perspective.
- Brightness, contrast, and color jitter (lighting changes, white-balance shifts).
- Motion blur and Gaussian blur; Gaussian noise; JPEG compression artifacts.
- Random erasing or coarse dropout (partial occlusion by hands or other objects).
- Background variation where feasible.

### 8.4 Experiments
- **E1:** Phase 1 baseline.
- **E2:** Phase 1 + Phase 2 (main model).
- **E3:** Backbone comparison under an identical protocol.
- **E4:** Effect of camera-realistic augmentation (with vs without), evaluated on the **field set**. This is the key experiment for a camera-based product.
- **E5 (P2):** Ablation: from scratch vs frozen vs fine-tuned.

### 8.5 The field set (camera-domain test)
- Photograph **100+ items** (about 15-20 per class) with your phone or laptop camera in varied lighting and backgrounds, including messy real-world scenes.
- Label them, store them under `data/field_set/` (kept out of git if they contain anything personal), and never train or tune on them in the main results.
- Report accuracy, macro-F1, and a confusion matrix separately from the test set, and discuss the gap.
- Optional follow-up: in a later, clearly labeled experiment, split the field set into two halves and use one half for extra fine-tuning and the other half for evaluation.
- Include some **negative images** (empty table, hands, faces, unrelated objects) to test the "no item" behavior.

### 8.6 Evaluation and success metrics (proposed targets; revise after E1)
| Metric | Target |
|---|---|
| Test macro-F1 | ≥ 0.90 |
| Phase 2 gain over Phase 1 | ≥ +3 percentage points macro-F1 (a hypothesis to test) |
| Field-set top-1 accuracy | Reported honestly; proposed ≥ 0.80 |
| Weakest class recall | Reported and discussed with a confusion analysis |
| Negative images | Most are rejected as "not sure" or "no item" (report the rate) |
| CPU latency (ONNX Runtime) | ≤ 100 ms model-only (measure and report) |
| Public demo | Works from a phone camera over HTTPS |

---

## 9. Live camera pipeline (P1)

**Flow:** capture frame → center-crop to the guide box → quality gate → resize and normalize → model → smoothing → display.

1. **Frame rate:** classify at a throttled rate (target about 2-5 frames per second), not every video frame.
2. **Quality gate:** compute a blur score (variance of the Laplacian) and a mean-brightness check. If poor, show "Hold steady" or "Too dark" and skip inference.
3. **Smoothing:** keep a sliding window (default N = 5) and average the probability vectors, or apply an exponential moving average.
4. **Confirmed state:** mark a result "confirmed" when the top class has been the same for K consecutive frames (default K = 3) and smoothed confidence is at least the threshold (default 0.6).
5. **No item:** if confidence stays below the threshold, or the `background` class wins, show "Point the camera at an item".
6. **Display:** label, confidence, bin, and a short hint. Show Grad-CAM only on demand.

The smoothing and stability logic lives in a **pure-Python class** (for example `FrameSmoother` in `live.py`) with no UI dependency, so it can be unit-tested with synthetic probability sequences and reused by any front end.

---

## 10. UI and UX requirements

### 10.1 Layout
1. **Input area:** tabs for **Upload**, **Camera snapshot**, and **Live camera**; sample images for a quick try.
2. **Result card:** predicted class, confidence, top-3 bar chart, and a colored bin badge with a text label.
3. **Guidance line:** contextual hints ("Hold steady", "Too dark", "Not sure, try another angle").
4. **Explain toggle:** Grad-CAM overlay, computed lazily.
5. **About tab:** model, dataset, test and field-set metrics, and honest limitations.
6. (P2) Session stats: items classified per category.

### 10.2 Privacy and permissions
- Frames are processed in memory only; nothing is written to disk by default.
- If opt-in feedback is added (P2), store the predicted label and the user's correction only; store an image only with explicit consent.
- Explain in the UI what the camera is used for.

### 10.3 UI performance rules
- Load the model once at startup.
- Use the ONNX Runtime model when available.
- Downscale frames before inference.
- Compute Grad-CAM only on demand.
- Never block the UI thread while inferring.

---

## 11. Technical stack (open-source libraries)

### 11.1 Machine learning
| Need | Library | Why |
|---|---|---|
| Framework | **PyTorch**, **torchvision** | Standard, huge ecosystem |
| Pretrained backbones | **timm** | Many pretrained models; one line to create with a new head |
| Training structure | **PyTorch Lightning** | Checkpointing, early stopping, mixed precision, logging |
| Augmentation | **Albumentations** | Fast; has blur, noise, JPEG, perspective, and dropout transforms |
| Image quality checks | **opencv-python-headless** | Laplacian blur score, brightness, crops |
| Metrics | **torchmetrics**, **scikit-learn** | F1, confusion matrix, reports, class weights |
| Explainability | **pytorch-grad-cam** | Grad-CAM for CNNs |
| Duplicate check | **imagehash** | Perceptual hashing to detect leakage |
| Experiment tracking | **TensorBoard** (default), **MLflow** (optional) | Compare runs |
| Hyperparameter search (P2) | **Optuna** | Lightweight |
| Config | **OmegaConf** + YAML | Simple, typed |
| Data and plots | **NumPy**, **pandas**, **Pillow**, **matplotlib**, **seaborn** | Standard |

### 11.2 Engineering and deployment
| Need | Library or service | Why |
|---|---|---|
| Environment | **uv** | Fast installs, lockfile |
| Lint and format | **ruff** | One tool |
| Tests | **pytest** | Standard |
| Git hooks | **pre-commit** | Lint and test before commits |
| CLI | **Typer** | Clean `train`, `evaluate`, `export` commands |
| Fast inference and export | **ONNX**, **ONNX Runtime** | Small, fast CPU inference |
| Hosting (free tier) | **Hugging Face Spaces** (Gradio) | HTTPS out of the box, so the phone camera works |
| Containerization (optional) | **Docker** | Reproducible deployment |

### 11.3 UI options
| Option | Best for | Role |
|---|---|---|
| **Gradio** | Fastest ML demo: image upload, webcam, streaming, label with confidences, examples, themes | **Primary UI (P0/P1).** Upload, snapshot, and live camera. |
| **Streamlit** + **streamlit-webrtc** | Dashboard-style layouts and live video | Alternative if you prefer it |
| **NiceGUI** | Python-only UI with a custom look (Tailwind-based) | Middle ground for design control |
| **React + Vite + Tailwind CSS + shadcn/ui + Recharts + ONNX Runtime Web** | A premium interface where the model runs **entirely in the browser** (no server, works offline, images never leave the device) | **P2 stretch**, and the best-looking option. Requires ONNX export (FR-21). |

**Recommendation:** build the Gradio app first so a working camera and upload demo exists early. If you want the "really good UI", add the React client-side app afterwards, reusing the same exported ONNX model and the same bin-mapping config. The MobileNetV3 model is the most practical choice for in-browser inference; size is roughly 20 MB in FP32 and can be reduced with INT8 quantization.

---

## 12. Architecture and repository layout

```
smart-waste-classifier/
├── AGENTS.md                  # rules for Antigravity (see section 13)
├── PRD.md                     # v1 (superseded)
├── PRD_software_only.md       # this document, the source of truth
├── .agents/
│   ├── rules/
│   ├── skills/                # reusable how-to packages (SKILL.md each)
│   └── workflows/             # slash commands: /train, /evaluate, /export
├── configs/
│   ├── base.yaml
│   ├── backbones/             # efficientnet_b0.yaml, mobilenetv3.yaml, resnet50.yaml
│   ├── augmentation.yaml
│   ├── live.yaml              # fps, window size, stability K, thresholds
│   └── bin_mapping.yaml
├── data/                      # gitignored: raw data, field_set, checkpoints
├── notebooks/
│   ├── 01_eda.ipynb
│   └── colab_train.ipynb      # thin wrapper that calls the CLI
├── src/waste_classifier/
│   ├── data/                  # download.py, splits.py, datamodule.py, transforms.py
│   ├── models/                # classifier.py (LightningModule)
│   ├── train.py
│   ├── evaluate.py            # test set and field set
│   ├── explain.py             # Grad-CAM
│   ├── quality.py             # blur and brightness checks
│   ├── live.py                # FrameSmoother (pure Python, unit-tested)
│   ├── export.py              # ONNX export and parity check
│   ├── inference.py           # predict() used by every UI
│   └── cli.py                 # Typer commands
├── app/
│   ├── app.py                 # Gradio UI
│   └── assets/
├── web/                       # (P2) React + ONNX Runtime Web client
├── tests/
├── reports/                   # metrics JSON, figures, tables
├── pyproject.toml
├── uv.lock
└── README.md
```

**Data flow:** `download` → `splits` (manifest) → `DataModule` → `train` (Phase 1 → Phase 2) → checkpoint → `evaluate` (test set and field set) → `reports/` → `export` (ONNX) → `inference.predict()` → `app/app.py` (upload, snapshot, live) → deploy.

---

## 13. Building this with Google Antigravity

Antigravity is an agent-first IDE. Treat it as a fast junior engineer that follows written rules, and keep yourself in control of every decision.

> **Note:** Antigravity's conventions (folder names such as `.agents/`, and where rules, skills, and workflows live) have changed between versions. Check the current Antigravity docs when setting up and adjust paths if needed.

### 13.1 Principles
1. **Spec-driven:** this PRD and `AGENTS.md` are the source of truth. Every task cites a requirement ID and its acceptance criteria.
2. **Plan first:** read and correct the agent's plan before any code is written.
3. **Small vertical slices:** one requirement at a time, never "build the whole project".
4. **Verify everything:** run lint and tests yourself, and read the tests the agent wrote.
5. **Explain every line:** if you cannot explain it in a viva, ask the agent to simplify it.
6. **Use established libraries** from section 11; no new dependencies without approval.

### 13.2 Setup checklist
- [ ] Create the repo and folder layout from section 12.
- [ ] Add `PRD_software_only.md` and `AGENTS.md` to the root.
- [ ] Create `.agents/skills/` and `.agents/workflows/`.
- [ ] Run `uv init`, add dependencies, commit `uv.lock`.
- [ ] Set up `ruff` and `pre-commit`.

### 13.3 Sample `AGENTS.md`
```markdown
# AGENTS.md: Smart Waste Classifier (software-only, camera and upload)

## Project
Transfer-learning waste classifier delivered as a web app (upload, camera snapshot,
live camera). Source of truth: PRD_software_only.md. Read the relevant section
and requirement ID before planning any task.

## Stack (do not add dependencies without asking)
Python 3.11+, PyTorch, timm, PyTorch Lightning, Albumentations, opencv-python-headless,
torchmetrics, scikit-learn, pytorch-grad-cam, OmegaConf, ONNX Runtime, Gradio,
pytest, ruff. Environment managed with `uv`.

## Commands
- Install: `uv sync`
- Lint and format: `uv run ruff check . && uv run ruff format .`
- Test: `uv run pytest -q`
- Train: `uv run python -m waste_classifier.cli train --config configs/base.yaml`
- App: `uv run python app/app.py`

## Rules
1. Plan first: propose a plan and wait for approval before editing more than 2 files.
2. Type hints and docstrings on public functions. Reusable logic lives in src/, not notebooks.
3. Config-driven: no hardcoded paths, hyperparameters, thresholds, or class names.
4. Reproducibility: seed everything; log config and git commit with every run.
5. Data hygiene: split once, save the manifest. Never tune on the test set or the field
   set; only `evaluate` may read them.
6. Privacy: never write camera frames or uploaded images to disk or logs by default.
7. UI-independent logic (inference, smoothing, quality checks) lives in src/ with tests;
   the UI only calls it.
8. Inference preprocessing must match evaluation preprocessing (parity test required).
9. Every new module ships with at least one pytest test. Run lint and tests before
   saying a task is done.
10. Small commits, one concern each. Never commit data, checkpoints, or secrets.
```

### 13.4 Sample skill: `.agents/skills/ml-experiment-hygiene/SKILL.md`
```markdown
---
name: ml-experiment-hygiene
description: Checklist for any change to data, training, evaluation, or live-inference code in this project.
---

# ML experiment hygiene
Before finishing a change:
1. Splits are created once, saved as a manifest, and disjoint (a test exists).
2. Augmentation is applied to the train split only.
3. Seeds are set; config and git commit are logged.
4. Metrics include macro-F1 and per-class recall, not only accuracy.
5. The test set and field set were not used for tuning.
6. Inference preprocessing equals evaluation preprocessing.
7. No camera frames or uploads are persisted.
8. Run `ruff` and `pytest` and report the results.
```

### 13.5 Sample workflow: `.agents/workflows/train.md` (invoked as `/train`)
```markdown
---
description: Run a full two-phase training run and record results
---
1. Run lint and tests; stop if they fail.
2. Run Phase 1 with the given config; save the best checkpoint.
3. Run Phase 2 from the Phase 1 checkpoint.
4. Compute validation metrics and append a row to reports/runs.csv
   (config name, seed, val macro-F1, git commit).
5. Summarize results and flag anything unusual (overfitting, a class with very low recall).
```
Create similar workflows for `/evaluate` (test set and field set, run deliberately), `/export` (ONNX plus parity check), and `/review` (review the current diff against AGENTS.md).

### 13.6 Working loop and prompt template
1. Start a task in the Agent Manager with the template below.
2. Review the plan; edit or reject it if it deviates from the PRD.
3. Let the agent implement one slice.
4. Run `ruff` and `pytest` yourself; read the diff and the tests.
5. Run the relevant command and inspect the output.
6. Commit with a clear message.

```
Implement <FR-ID> from PRD_software_only.md only.
Acceptance criteria: <copy from the PRD table>.
Constraints: follow AGENTS.md; use existing libraries; no new dependencies.
First produce a plan and wait for my approval before editing files.
Also write pytest tests for the acceptance criteria.
```

### 13.7 Parallel agents
Once `inference.predict()` has a fixed signature (PIL image in; top-k list of class name and probability out), one agent can build the Gradio UI while another works on evaluation or export. Agree on the interface first so they do not collide.

### 13.8 Anti-patterns
- One giant prompt such as "build the whole project".
- Accepting data-splitting, metric, or smoothing code without reading it.
- Letting the agent evaluate on the test set or field set while tuning.
- Letting the agent add libraries or rewrite working code silently.
- Treating "tests pass" as proof of correctness when the agent wrote both the code and the tests.

If your Antigravity version has a browser agent, use it to smoke-test the UI (upload an image, check the result renders), but still verify the camera flow yourself on a real device.

---

## 14. Milestones (adjust to your deadline)

| Week | Focus | Exit criteria |
|---|---|---|
| 1 | Setup, data, EDA, split, tests | Repo, PRD, AGENTS.md in place; FR-1 to FR-3 done; class-count and duplicate report; start collecting the field set |
| 2 | Baseline and evaluation | FR-4, FR-5, FR-7, FR-9 done; E1 baseline saved |
| 3 | Fine-tuning and comparison | FR-6, FR-18, FR-19 done; E2 and E3 results; field set complete |
| 4 | Camera app | FR-8, FR-10 to FR-12 done; E4 (augmentation vs field set); Gradio app working with upload and snapshot |
| 5 | Live mode, polish, deploy | FR-13 to FR-17, FR-20, FR-21 done; deployed HTTPS demo tested on a phone; report, slides, README |
| Stretch | React client-side app | FR-23, if time remains |

---

## 15. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Clean dataset photos vs messy camera frames (domain shift) | Camera-realistic augmentation; field-set evaluation; E4 experiment; uncertainty threshold |
| Model confidently labels empty scenes, hands, or faces | Confidence threshold; optional `background` class; negative test images |
| Live mode flickers or lags | Throttled inference, smoothing window, tested `FrameSmoother` |
| Camera blocked (permission denied, no HTTPS, no camera) | Upload fallback; deploy to an HTTPS host for phone testing |
| Small or imbalanced dataset | Augmentation, class weights, macro-F1, low LR in Phase 2 |
| Data leakage or near-duplicates | Split first; perceptual-hash check; disjoint-split test |
| Agent-written code that looks right but is wrong | Plan review, reading diffs, tests, running commands yourself |
| Colab session timeouts | Save checkpoints to Drive; make training resumable |
| Scope creep | Finish all P0 before P1 and P2 |
| Dependency drift | `uv.lock`; no new libraries without approval |

---

## 16. Deliverables

- Source repo with README, tests, and lockfile.
- Trained checkpoint and ONNX export.
- `reports/`: metrics JSON, confusion matrices (test set and field set), backbone comparison, augmentation experiment (E4), multi-seed results, Grad-CAM figures.
- Deployed demo (HTTPS) supporting upload, camera snapshot, and live camera, plus a short screen recording as a backup.
- Project report (problem, method, experiments, results, error analysis, limitations).
- Presentation slides.

---

## 17. Definition of done

- [ ] All P0 requirements met and acceptance criteria verified.
- [ ] `uv run ruff check .` and `uv run pytest -q` pass.
- [ ] The test set and field set were used only for final evaluation.
- [ ] Results are reproducible from the config, seed, and commit in the logs.
- [ ] Per-class metrics, confusion matrices, and error analysis are in the report.
- [ ] Upload and camera snapshot work on a laptop; the deployed demo works on a phone camera.
- [ ] Low-confidence and "no item" cases are handled gracefully.
- [ ] No camera frames or uploads are stored by default.
- [ ] The README covers setup, training, evaluation, and running the app.
- [ ] You can explain the code, the two-phase method, and every result in the report.

---

## Appendix A: sample `configs/base.yaml`

```yaml
seed: 42
data:
  root: data/raw
  manifest: data/splits.csv
  field_set: data/field_set
  split: {train: 0.70, val: 0.15, test: 0.15}
  image_size: 224
  batch_size: 32
  num_workers: 2
model:
  backbone: efficientnet_b0
  pretrained: true
  dropout: 0.3
loss:
  label_smoothing: 0.1
  use_class_weights: true
phase1:
  epochs: 10
  lr: 1.0e-3
phase2:
  epochs: 15
  unfreeze_fraction: 0.25
  lr_backbone: 1.0e-5
  lr_head: 1.0e-4
  early_stopping_patience: 5
inference:
  top_k: 3
  confidence_threshold: 0.6
```

## Appendix B: sample `configs/live.yaml`

```yaml
live:
  target_fps: 4
  smoothing_window: 5
  stable_frames: 3
  confidence_threshold: 0.6
  quality:
    min_blur_score: 60      # variance of Laplacian; tune on your own frames
    min_brightness: 50      # mean pixel value 0-255; tune on your own frames
```

## Appendix C: viva talking points

- Why transfer learning works on small datasets.
- Why freeze first, then unfreeze at a low learning rate.
- Why macro-F1 and per-class recall, not accuracy alone.
- Why the split happens before augmentation.
- Why a camera app needs a separate field set, and what E4 showed about the domain gap.
- How frame smoothing reduces flicker, and why inference is throttled.
- What Grad-CAM showed, including one case where the model looked at the wrong thing.
- Limitations: one item per image, dataset backgrounds, lighting, no detection.
