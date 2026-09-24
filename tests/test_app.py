"""Unit and integration tests for Gradio web application (FR-10, FR-11, FR-12).

Verifies:
1. Gradio Blocks app assembly without configuration or build errors.
2. Classification handler on valid PIL image input.
3. Graceful handling of None/empty input (no crash).
4. Accessibility compliance: mandatory text labels accompany all color indicators.
5. In-memory privacy preservation: zero images written to disk.
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import numpy as np
import pytest
from PIL import Image

from app.app import classify_image_handler, create_app, format_bin_card
from waste_classifier.inference import PredictionItem, PredictionResult


@pytest.fixture
def sample_pil_image() -> Image.Image:
    """Fixture providing a synthetic test PIL RGB image."""
    arr = np.random.randint(0, 255, (250, 250, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def test_create_app_structure() -> None:
    """Verify that create_app() initializes a valid Gradio Blocks instance."""
    app = create_app()
    assert isinstance(app, gr.Blocks)
    assert app.title == "Smart Waste Classifier"


def test_classify_image_handler_valid(sample_pil_image: Image.Image) -> None:
    """Verify classify_image_handler returns populated results for a valid image."""
    header, prob_dict, bin_html, guidance_html = classify_image_handler(sample_pil_image)

    # 1. Header validation
    assert isinstance(header, str)
    assert "%" in header

    # 2. Probability dictionary for gr.Label
    assert isinstance(prob_dict, dict)
    assert len(prob_dict) > 0
    for cls_name, prob in prob_dict.items():
        assert isinstance(cls_name, str)
        assert 0.0 <= prob <= 1.0

    # 3. Accessible HTML bin card
    assert isinstance(bin_html, str)
    assert "🗑️" in bin_html
    assert "Instructions:" in bin_html

    # 4. Contextual guidance card
    assert isinstance(guidance_html, str)
    assert "Guidance:" in guidance_html


def test_classify_image_handler_none() -> None:
    """Verify classify_image_handler handles None gracefully without raising exceptions."""
    header, prob_dict, bin_html, guidance_html = classify_image_handler(None)

    assert header == "No Image"
    assert prob_dict == {}
    assert "No image provided" in bin_html
    assert "Awaiting image input" in guidance_html


def test_accessibility_compliance_in_bin_card() -> None:
    """Verify accessibility mandate: explicit text label accompanies color in bin recommendation."""
    fake_item = PredictionItem(
        class_name="glass",
        probability=0.88,
        index=1,
        bin_name="Glass Recycling",
        bin_color="#00897B",
        bin_label="Teal/Green Bin - Glass",
    )

    fake_res = PredictionResult(
        top_class="glass",
        confidence=0.88,
        is_confident=True,
        predictions=[fake_item],
        all_probabilities={"glass": 0.88},
        guidance="Identified as Glass with 88.0% confidence.",
        bin_name="Glass Recycling",
        bin_color="#00897B",
        bin_label="Teal/Green Bin - Glass",
        bin_instructions="Rinse bottles and jars thoroughly.",
    )

    card_html = format_bin_card(fake_res)

    # Color indicator present
    assert "#00897B" in card_html
    # Text label present (accessibility compliance)
    assert "Teal/Green Bin - Glass" in card_html
    assert "Glass Recycling" in card_html


def test_in_memory_privacy_preservation(sample_pil_image: Image.Image, tmp_path: Path) -> None:
    """Verify that classifying an image does not persist files to disk."""
    # Count files before
    files_before = set(os.listdir("."))

    # Run classification
    _ = classify_image_handler(sample_pil_image)

    # Count files after
    files_after = set(os.listdir("."))

    # No new files created in root directory
    diff = files_after - files_before
    assert len(diff) == 0, f"Unexpected files written to disk during inference: {diff}"


def test_live_frame_handler_none() -> None:
    """Verify live_frame_handler handles None gracefully without errors."""
    from app.app import live_frame_handler

    status, prob_dict, bin_html, guidance_html = live_frame_handler(None)
    assert status == "Awaiting Camera..."
    assert prob_dict == {}
    assert "STANDBY" in guidance_html
    assert "Instructions:" in bin_html


def test_live_frame_handler_valid(sample_pil_image: Image.Image) -> None:
    """Verify live_frame_handler processes incoming frame and returns live badges."""
    from app.app import live_frame_handler

    status, prob_dict, bin_html, guidance_html = live_frame_handler(sample_pil_image)
    assert isinstance(status, str)
    assert isinstance(prob_dict, dict)
    assert "Instructions:" in bin_html
    assert "Target ~4 FPS" in guidance_html


def test_reset_live_handler() -> None:
    """Verify reset_live_handler clears live pipeline state."""
    from app.app import reset_live_handler

    status, prob_dict, bin_html, guidance_html = reset_live_handler()
    assert "Stabilizer Reset" in status
    assert prob_dict == {}
    assert "STANDBY" in guidance_html


def test_explain_image_handler_none() -> None:
    """Verify explain_image_handler handles empty inputs without crashing."""
    from app.app import explain_image_handler

    cam_img, info_text = explain_image_handler(None, None, "Top Prediction")
    assert cam_img is None
    assert "No image provided" in info_text


def test_explain_image_handler_valid(sample_pil_image: Image.Image) -> None:
    """Verify explain_image_handler returns overlay PIL image and interpretability disclaimer."""
    from app.app import explain_image_handler

    cam_img, info_text = explain_image_handler(sample_pil_image, None, "Top Prediction")
    assert isinstance(cam_img, Image.Image)
    assert cam_img.size == sample_pil_image.size
    assert "Explained Target:" in info_text
    assert "disclaimer" in info_text.lower() or "causal" in info_text.lower()


def test_format_session_stats():
    """Verify session stats formatting (FR-24)."""
    from app.app import format_session_stats

    stats = {"cardboard": 2, "plastic": 5, "metal": 1}
    card_html = format_session_stats(stats)
    assert "Items Sorted This Session:" in card_html
    assert "8" in card_html
    assert "Plastic:" in card_html


def test_record_feedback_preserves_privacy(tmp_path: Path):
    """Verify feedback logging records labels only, zero images (FR-24)."""
    import json

    from app.app import record_feedback

    log_file = tmp_path / "feedback.jsonl"

    # Test positive feedback
    msg1 = record_feedback("PLASTIC (94.2%)", "CORRECT", "", log_path=log_file)
    assert "correct" in msg1.lower()
    assert "no image" in msg1.lower()

    # Test correction feedback
    msg2 = record_feedback("TRASH (51.0%)", "INCORRECT", "Cardboard", log_path=log_file)
    assert "corrected" in msg2.lower()

    # Verify JSONL lines contain strictly metadata and labels
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    entry1 = json.loads(lines[0])
    assert entry1["predicted_class"] == "plastic"
    assert entry1["feedback"] == "CORRECT"
    assert "image" not in entry1

    entry2 = json.loads(lines[1])
    assert entry2["predicted_class"] == "trash"
    assert entry2["suggested_label"] == "cardboard"
    assert "image" not in entry2
