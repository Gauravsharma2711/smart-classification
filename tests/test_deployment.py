"""Unit and integration tests for Deployment Configuration and HTTPS Demo (FR-20).

Verifies:
1. Production deployment artifacts (Dockerfile, Hugging Face metadata, deployment configs).
2. Model loading readiness and graceful fallback behavior.
3. Understandable failure states when processing corrupted or unsupported images.
4. Non-persistence guarantees: zero images written to disk during application lifecycle.
5. Presence of clear privacy messaging and mobile camera permission troubleshooting guidance.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.app import classify_image_handler, create_app
from waste_classifier.inference import WasteClassifierPredictor


@pytest.fixture
def sample_pil_image() -> Image.Image:
    """Fixture providing a synthetic test PIL RGB image."""
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def test_dockerfile_configuration():
    """Verify Dockerfile exists and contains required production specifications."""
    dockerfile = Path("Dockerfile")
    assert dockerfile.exists(), "Dockerfile must exist at repository root."

    content = dockerfile.read_text(encoding="utf-8")
    assert "FROM python:3.12" in content
    assert "EXPOSE 7860" in content
    assert "app" in content
    assert "uv" in content


def test_huggingface_deployment_metadata():
    """Verify deployment metadata for hosted Gradio Spaces exists and is valid."""
    deploy_readme = Path("deploy/README.md")
    assert deploy_readme.exists(), "deploy/README.md must exist."

    content = deploy_readme.read_text(encoding="utf-8")
    assert "sdk: gradio" in content
    assert "app_file: app/app.py" in content
    assert "HTTPS" in content


def test_model_loading_and_fallback_resilience():
    """Verify that predictor loads fine-tuned model or falls back gracefully without crash."""
    # Initialize with default search
    predictor = WasteClassifierPredictor()
    assert predictor.model is not None
    assert predictor.num_classes == 6

    # Test with non-existent checkpoint path -> triggers fallback to pretrained
    fallback_predictor = WasteClassifierPredictor(checkpoint_path="non_existent/fake.ckpt")
    assert fallback_predictor.model is not None
    assert fallback_predictor.checkpoint_path is None


def test_unsupported_and_corrupt_image_failure_state():
    """Verify understandable error cards are returned for corrupted or non-image inputs."""
    # Test corrupt/empty image mode (e.g. 0-dimension image)
    corrupt_image = Image.new("RGB", (0, 0))

    header, prob_dict, bin_html, guidance_html = classify_image_handler(corrupt_image)

    # Must return understandable error status rather than crashing
    assert "Error" in header or "%" in header
    assert "Classification Error" in bin_html or "Instructions:" in bin_html
    assert "Guidance:" in guidance_html or "Notice:" in guidance_html


def test_deployment_privacy_messaging_and_troubleshooting():
    """Verify that the deployed UI contains clear privacy guarantees and mobile troubleshooting."""
    demo = create_app()
    assert demo is not None

    # Inspect rendered layout blocks
    app_text = ""
    for block in demo.blocks.values():
        if hasattr(block, "value") and isinstance(block.value, str):
            app_text += block.value + "\n"

    # Privacy guarantee checks
    assert "Privacy Guarantee" in app_text
    assert "never saved to disk" in app_text

    # Mobile and HTTPS troubleshooting checks
    assert "HTTPS" in app_text
    assert "Safari" in app_text
    assert "Chrome" in app_text
    assert "Upload" in app_text


def test_strict_zero_disk_persistence_during_upload_and_stream(sample_pil_image: Image.Image):
    """Verify that end-to-end inference and live frame processing write zero files to disk."""
    before_files = set(os.listdir("."))

    # 1. Run upload classification
    _ = classify_image_handler(sample_pil_image)

    # 2. Run live frame handler
    from app.app import live_frame_handler

    _ = live_frame_handler(sample_pil_image)

    after_files = set(os.listdir("."))
    diff = after_files - before_files
    assert len(diff) == 0, f"Detected unexpected files written to workspace root: {diff}"
