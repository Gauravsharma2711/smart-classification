---
title: Smart Waste Classifier
emoji: ♻️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.28.0
app_file: app/app.py
pinned: false
license: mit
---

# ♻️ Smart Waste Classifier - Hosted HTTPS Demo (FR-20)

Automated waste classification and bin recommendations powered by fine-tuned deep learning CNNs (`efficientnet_b0`, `mobilenetv3_large_100`).

## Features
- **📁 Image Upload:** Select or drop photo with instant top-3 predictions and bin disposal advice.
- **📸 Camera Snapshot:** Mobile-optimized single-frame capture with rear camera support.
- **🎥 Live Camera Mode:** Real-time throttled streaming (~4 FPS) with automated blur/brightness quality gates and temporal stabilization.
- **🔍 Grad-CAM Explainability:** On-demand visual convolutional saliency overlays.
- **🔒 Privacy Guarantee:** 100% ephemeral in-memory processing. Zero images or video frames persisted to disk or cloud.

## Deployment Options

### 1. Instant HTTPS Tunnel (Gradio Share)
Run locally on your machine with public TLS tunnel:
```bash
waste-classifier deploy --share
```
or
```bash
waste-classifier app --share
```
This outputs a secure `https://xxxx.gradio.live` link instantly accessible on any mobile phone (iOS Safari / Android Chrome) with active camera permissions.

### 2. Hugging Face Spaces Deployment
1. Create a new Space at [Hugging Face Spaces](https://huggingface.co/new-space) (SDK: Gradio, Python 3.12).
2. Push repository files or set repository as a Git remote:
```bash
git remote add space https://huggingface.co/spaces/<username>/<space-name>
git push space main
```

### 3. Docker Container Deployment
Build and run anywhere with Docker:
```bash
docker build -t smart-waste-classifier .
docker run -p 7860:7860 smart-waste-classifier
```
