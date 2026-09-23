"""Unit and integration tests for inference module (FR-9).

Verifies:
1. Preprocessing parity: inference preprocessing strictly equals evaluation preprocessing.
2. PIL Image input support across color modes (RGB, RGBA, Grayscale).
3. Defensive error handling on malformed or invalid inputs.
4. Deterministic inference across repeated calls.
5. Strict probability normalization (sum == 1.0) and top-k descending ordering.
6. Confidence threshold gating and contextual UI guidance.
7. Serialization of PredictionResult.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from waste_classifier.data.transforms import build_eval_transforms
from waste_classifier.inference import (
    CANONICAL_CLASSES,
    PredictionResult,
    WasteClassifierPredictor,
    get_predictor,
    predict,
)


@pytest.fixture(scope="module")
def predictor() -> WasteClassifierPredictor:
    """Fixture providing initialized WasteClassifierPredictor instance."""
    return get_predictor(device="cpu", force_reload=True)


@pytest.fixture
def sample_pil_image() -> Image.Image:
    """Fixture providing a standard test PIL RGB image."""
    arr = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def test_preprocessing_parity(
    predictor: WasteClassifierPredictor, sample_pil_image: Image.Image
) -> None:
    """Verify that inference preprocessing strictly equals evaluation preprocessing.

    Acceptance criterion (FR-9):
    A parity test confirms inference preprocessing equals eval preprocessing down to exact numerical tensors.
    """
    # 1. Inference preprocessing tensor
    infer_tensor = predictor.preprocess_image(sample_pil_image)

    # 2. Evaluation preprocessing tensor
    eval_transform = build_eval_transforms(image_size=predictor.image_size)
    rgb_arr = np.array(sample_pil_image.convert("RGB"))
    eval_tensor = eval_transform(image=rgb_arr)["image"].unsqueeze(0)

    # Assert exact numerical match
    assert infer_tensor.shape == eval_tensor.shape
    assert torch.equal(infer_tensor, eval_tensor), (
        "Inference preprocessing tensor does NOT match evaluation preprocessing tensor!"
    )


def test_pil_image_modes(predictor: WasteClassifierPredictor) -> None:
    """Verify inference accepts and correctly processes RGB, RGBA, and Grayscale PIL images."""
    # RGB
    rgb_img = Image.new("RGB", (250, 250), color=(200, 100, 50))
    res_rgb = predictor.predict(rgb_img)
    assert isinstance(res_rgb, PredictionResult)
    assert res_rgb.top_class in CANONICAL_CLASSES

    # RGBA
    rgba_img = Image.new("RGBA", (200, 300), color=(50, 150, 200, 255))
    res_rgba = predictor.predict(rgba_img)
    assert isinstance(res_rgba, PredictionResult)
    assert res_rgba.top_class in CANONICAL_CLASSES

    # Grayscale
    gray_img = Image.new("L", (180, 180), color=128)
    res_gray = predictor.predict(gray_img)
    assert isinstance(res_gray, PredictionResult)
    assert res_gray.top_class in CANONICAL_CLASSES


def test_malformed_inputs_fail_loudly(predictor: WasteClassifierPredictor) -> None:
    """Verify malformed or invalid inputs raise appropriate exceptions with clear messages."""
    # None input
    with pytest.raises(ValueError, match="Input image cannot be None"):
        predictor.predict(None)  # type: ignore

    # Non-PIL types
    with pytest.raises(TypeError, match="Expected image of type PIL.Image.Image"):
        predictor.predict("path/to/image.jpg")  # type: ignore

    with pytest.raises(TypeError, match="Expected image of type PIL.Image.Image"):
        predictor.predict(np.zeros((224, 224, 3), dtype=np.uint8))  # type: ignore

    with pytest.raises(TypeError, match="Expected image of type PIL.Image.Image"):
        predictor.predict([1, 2, 3])  # type: ignore

    # Zero-dimension image
    empty_img = Image.new("RGB", (0, 0))
    with pytest.raises(ValueError, match="Image has invalid dimensions"):
        predictor.predict(empty_img)


def test_deterministic_inference(
    predictor: WasteClassifierPredictor, sample_pil_image: Image.Image
) -> None:
    """Verify that multiple inference calls on the exact same input yield identical results."""
    res1 = predictor.predict(sample_pil_image)
    res2 = predictor.predict(sample_pil_image)

    assert res1.top_class == res2.top_class
    assert pytest.approx(res1.confidence, rel=1e-5) == res2.confidence
    assert res1.all_probabilities == res2.all_probabilities
    for item1, item2 in zip(res1.predictions, res2.predictions, strict=True):
        assert item1.class_name == item2.class_name
        assert item1.probability == item2.probability


def test_probability_normalization_and_ordering(
    predictor: WasteClassifierPredictor, sample_pil_image: Image.Image
) -> None:
    """Verify probabilities are strictly normalized and top-k list is sorted descending."""
    res = predictor.predict(sample_pil_image)

    # 1. Normalization
    prob_sum = sum(res.all_probabilities.values())
    assert pytest.approx(prob_sum, abs=0.01) == 1.0

    for prob in res.all_probabilities.values():
        assert 0.0 <= prob <= 1.0

    # 2. Descending ordering
    probs = [item.probability for item in res.predictions]
    assert probs == sorted(probs, reverse=True), "Top-k predictions must be ordered descending!"

    # 3. Top-k length
    assert len(res.predictions) == predictor.top_k
    assert res.predictions[0].class_name == res.top_class


def test_configurable_top_k(
    predictor: WasteClassifierPredictor, sample_pil_image: Image.Image
) -> None:
    """Verify top-k parameter alters number of returned predictions."""
    # Top-1
    res1 = predict(sample_pil_image, top_k=1)
    assert len(res1.predictions) == 1

    # Top-5
    res5 = predict(sample_pil_image, top_k=5)
    assert len(res5.predictions) == 5

    # Top-10 (capped at number of classes = 6)
    res_max = predict(sample_pil_image, top_k=10)
    assert len(res_max.predictions) == len(CANONICAL_CLASSES)


def test_confidence_threshold_and_guidance(sample_pil_image: Image.Image) -> None:
    """Verify confidence threshold behavior, certainty flags, and UI guidance messages."""
    # Ultra-low threshold: must be confident
    res_confident = predict(sample_pil_image, confidence_threshold=0.001)
    assert res_confident.is_confident is True
    assert "identified as" in res_confident.guidance.lower()
    assert res_confident.bin_name != "Uncertain / Verification Needed"

    # Ultra-high threshold: must be unconfident
    res_unconfident = predict(sample_pil_image, confidence_threshold=0.999)
    assert res_unconfident.is_confident is False
    assert "Low confidence" in res_unconfident.guidance
    assert res_unconfident.bin_name == "Uncertain / Verification Needed"
    assert "Inspect Item" in res_unconfident.bin_label


def test_prediction_result_serialization(
    predictor: WasteClassifierPredictor, sample_pil_image: Image.Image
) -> None:
    """Verify PredictionResult converts cleanly to a JSON-compatible dictionary."""
    res = predictor.predict(sample_pil_image)
    data = res.to_dict()

    assert isinstance(data, dict)
    assert "top_class" in data
    assert "confidence" in data
    assert "is_confident" in data
    assert "predictions" in data
    assert "all_probabilities" in data
    assert "bin_name" in data
    assert "guidance" in data
    assert len(data["predictions"]) == predictor.top_k
