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
