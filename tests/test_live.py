"""Comprehensive unit tests for Live Camera pipeline and stability engine (FR-13 to FR-16).

Tests:
1. Quality checks with synthetic images (blur and brightness detection).
2. Temporal probability smoothing with synthetic sequences.
3. Anti-flicker rejection under alternating noisy predictions.
4. Confidence threshold and uncertainty gating ("Point the camera at an item").
5. Background / no-item handling.
6. Stability confirmation over K consecutive frames.
7. Frame throttling against target FPS.
8. Center-crop geometry.
9. End-to-end LiveCameraPipeline integration with mocked predictor.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from PIL import Image

from waste_classifier.inference import CANONICAL_CLASSES, PredictionItem, PredictionResult
from waste_classifier.live import (
    FrameSmoother,
    LiveCameraPipeline,
    QualityGate,
    Throttler,
    center_crop,
    compute_blur_score,
    compute_brightness_score,
)

# ============================================================================
# 1. Quality Checks with Synthetic Images (FR-15)
# ============================================================================


def test_quality_dark_image_detection():
    """Verify that images below the minimum brightness threshold fail with 'Too dark'."""
    gate = QualityGate(min_blur_score=60.0, min_brightness=50.0)

    # Completely black image (brightness = 0)
    black_img = Image.new("RGB", (224, 224), color=(0, 0, 0))
    res = gate.check(black_img)
    assert not res.passed
    assert res.hint == "Too dark"
    assert res.brightness == 0.0

    # Low light image (brightness = 30 < 50)
    dim_img = Image.new("RGB", (224, 224), color=(30, 30, 30))
    res_dim = gate.check(dim_img)
    assert not res_dim.passed
    assert res_dim.hint == "Too dark"
    assert res_dim.brightness == 30.0


def test_quality_blurred_image_detection():
    """Verify that blurry images fail with 'Hold steady'."""
    gate = QualityGate(min_blur_score=60.0, min_brightness=50.0)

    # 1. Create a sharp synthetic checkerboard image with high edge variance and brightness > 50
    arr = np.zeros((224, 224), dtype=np.uint8)
    for i in range(0, 224, 16):
        for j in range(0, 224, 16):
            arr[i : i + 16, j : j + 16] = 200 if ((i // 16 + j // 16) % 2 == 0) else 60
    arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)

    sharp_blur = compute_blur_score(arr)
    sharp_bright = compute_brightness_score(arr)
    assert sharp_bright >= 50.0
    assert sharp_blur > 200.0  # High sharpness

    # Sharp image passes quality check
    res_sharp = gate.check(arr)
    assert res_sharp.passed
    assert res_sharp.hint is None

    # 2. Apply heavy Gaussian blur to eliminate high-frequency edges
    blurred_arr = cv2.GaussianBlur(arr, (31, 31), sigmaX=15.0)
    blur_score = compute_blur_score(blurred_arr)
    assert blur_score < 60.0  # Laplacian variance collapsed

    res_blur = gate.check(blurred_arr)
    assert not res_blur.passed
    assert res_blur.hint == "Hold steady"


def test_quality_overexposed_image():
    """Verify that washed-out images fail with 'Too bright'."""
    gate = QualityGate(min_blur_score=60.0, min_brightness=50.0, max_brightness=250.0)
    bright_img = Image.new("RGB", (224, 224), color=(255, 255, 255))
    res = gate.check(bright_img)
    assert not res.passed
    assert res.hint == "Too bright"


# ============================================================================
# 2. Center-Crop Target Geometry
# ============================================================================


def test_center_crop_pil():
    """Verify center-cropping on PIL images preserves centering and scale."""
    img = Image.new("RGB", (100, 200), color=(100, 150, 200))
    cropped = center_crop(img, crop_ratio=0.8)

    assert cropped.size == (80, 160)
    # Check bounds
    with pytest.raises(ValueError):
        center_crop(img, crop_ratio=0.0)
    with pytest.raises(ValueError):
        center_crop(img, crop_ratio=1.5)


def test_center_crop_numpy():
    """Verify center-cropping on NumPy array images."""
    arr = np.zeros((300, 400, 3), dtype=np.uint8)
    cropped = center_crop(arr, crop_ratio=0.5)

    assert cropped.shape == (150, 200, 3)


# ============================================================================
# 3. Throttler Rate Limiter (FR-13)
# ============================================================================


def test_throttler_target_fps():
    """Verify inference throttler enforces minimum inter-frame interval."""
    # Target 4 FPS -> 0.25s interval
    throttler = Throttler(target_fps=4.0)
    assert throttler.min_interval == 0.25

    t0 = 1000.0
    # First frame at t0: passes
    assert throttler.should_process(current_time=t0)

    # Frame at t0 + 0.1s (100ms later): throttled
    assert not throttler.should_process(current_time=t0 + 0.10)

    # Frame at t0 + 0.20s: throttled
    assert not throttler.should_process(current_time=t0 + 0.20)

    # Frame at t0 + 0.26s: passes
    assert throttler.should_process(current_time=t0 + 0.26)

    # Immediate next frame at t0 + 0.27s: throttled
    assert not throttler.should_process(current_time=t0 + 0.27)

    # Reset allows immediate execution
    throttler.reset()
    assert throttler.should_process(current_time=t0 + 0.27)


# ============================================================================
# 4. Frame Smoothing & Stability Confirmation (FR-14)
# ============================================================================


def test_smoother_sliding_window_averaging():
    """Verify mathematical correctness of rolling probability smoothing."""
    smoother = FrameSmoother(
        smoothing_window=3,
        stable_frames=2,
        confidence_threshold=0.60,
        class_names=["glass", "plastic"],
    )

    # Frame 1: plastic = 0.90, glass = 0.10
    r1 = smoother.update({"plastic": 0.90, "glass": 0.10})
    assert r1.top_class == "plastic"
    assert pytest.approx(r1.confidence, 0.01) == 0.90
    assert not r1.is_confirmed  # needs 2 frames

    # Frame 2: plastic = 0.60, glass = 0.40
    # Average: plastic = (0.90 + 0.60) / 2 = 0.75, glass = 0.25
    r2 = smoother.update({"plastic": 0.60, "glass": 0.40})
    assert r2.top_class == "plastic"
    assert pytest.approx(r2.confidence, 0.01) == 0.75
    assert r2.is_confirmed  # 2 consecutive frames >= 0.60 threshold!

    # Frame 3: plastic = 0.30, glass = 0.70
    # Window (3): [(0.9, 0.1), (0.6, 0.4), (0.3, 0.7)]
    # Mean: plastic = 0.60, glass = 0.40
    r3 = smoother.update({"plastic": 0.30, "glass": 0.70})
    assert r3.top_class == "plastic"
    assert pytest.approx(r3.confidence, 0.01) == 0.60
    assert r3.is_confirmed  # consecutive count = 3


def test_smoother_stability_confirmation():
    """Verify K consecutive frames requirement before marking confirmed."""
    smoother = FrameSmoother(
        smoothing_window=5,
        stable_frames=3,
        confidence_threshold=0.60,
        class_names=CANONICAL_CLASSES,
    )

    cardboard_frame = {c: 0.02 for c in CANONICAL_CLASSES}
    cardboard_frame["cardboard"] = 0.90

    # Frame 1: consecutive = 1 -> Not confirmed
    s1 = smoother.update(cardboard_frame)
    assert s1.consecutive_count == 1
    assert not s1.is_confirmed

    # Frame 2: consecutive = 2 -> Not confirmed
    s2 = smoother.update(cardboard_frame)
    assert s2.consecutive_count == 2
    assert not s2.is_confirmed

    # Frame 3: consecutive = 3 >= K -> CONFIRMED
    s3 = smoother.update(cardboard_frame)
    assert s3.consecutive_count == 3
    assert s3.is_confirmed
    assert "Item confirmed as Cardboard" in s3.guidance


def test_smoother_label_flicker_rejection():
    """Verify alternating noisy predictions do not cause UI flicker or confirmed state."""
    smoother = FrameSmoother(
        smoothing_window=5,
        stable_frames=3,
        confidence_threshold=0.60,
        class_names=["metal", "plastic"],
    )

    frame_metal = {"metal": 0.55, "plastic": 0.45}
    frame_plastic = {"metal": 0.45, "plastic": 0.55}

    # Alternate between metal and plastic
    for _ in range(6):
        r_metal = smoother.update(frame_metal)
        assert not r_metal.is_confirmed  # Cannot confirm oscillating label

        r_plastic = smoother.update(frame_plastic)
        assert not r_plastic.is_confirmed  # Cannot confirm oscillating label


# ============================================================================
# 5. Uncertainty & No-Item Handling (FR-16)
# ============================================================================


def test_smoother_uncertainty_threshold():
    """Verify that low-confidence predictions trigger 'Point the camera at an item'."""
    smoother = FrameSmoother(
        smoothing_window=5,
        stable_frames=2,
        confidence_threshold=0.60,
        class_names=CANONICAL_CLASSES,
    )

    # Flat probability distribution across all 6 classes (1/6 = ~0.166)
    uniform_frame = {c: 1.0 / len(CANONICAL_CLASSES) for c in CANONICAL_CLASSES}

    for _ in range(4):
        res = smoother.update(uniform_frame)
        assert not res.is_confirmed
        assert res.is_uncertain
        assert res.guidance == "Point the camera at an item"


def test_smoother_background_class_handling():
    """Verify that an explicit background class suppresses waste classification."""
    smoother = FrameSmoother(
        smoothing_window=3,
        stable_frames=2,
        confidence_threshold=0.60,
        class_names=["background", "plastic", "metal"],
        background_class="background",
    )

    bg_frame = {"background": 0.95, "plastic": 0.03, "metal": 0.02}

    # 3 consecutive background frames
    for _ in range(3):
        res = smoother.update(bg_frame)
        assert not res.is_confirmed
        assert res.is_uncertain
        assert res.guidance == "Point the camera at an item"


def test_smoother_reset():
    """Verify reset clears history and counters."""
    smoother = FrameSmoother(
        smoothing_window=3,
        stable_frames=2,
        confidence_threshold=0.60,
        class_names=CANONICAL_CLASSES,
    )

    sample_frame = {c: 0.02 for c in CANONICAL_CLASSES}
    sample_frame["glass"] = 0.90

    smoother.update(sample_frame)
    smoother.update(sample_frame)
    assert smoother._consecutive_count == 2

    smoother.reset()
    assert smoother._consecutive_count == 0
    assert len(smoother._window) == 0
    assert smoother._last_top_class is None


# ============================================================================
# 6. End-to-End LiveCameraPipeline Integration
# ============================================================================


def test_live_pipeline_full_cycle():
    """Test full integration with a mocked predictor."""
    mock_predictor = MagicMock()

    # Create dummy PredictionResult
    dummy_pred = PredictionResult(
        top_class="metal",
        confidence=0.88,
        is_confident=True,
        predictions=[
            PredictionItem(
                "metal", 0.88, 0, "Metals & Cans Recycling", "#FBC02D", "Yellow Bin - Metals"
            ),
            PredictionItem(
                "plastic", 0.08, 1, "Plastics Recycling", "#FB8C00", "Orange Bin - Plastics"
            ),
            PredictionItem(
                "trash", 0.04, 2, "General Waste", "#616161", "Gray Bin - General Waste"
            ),
        ],
        all_probabilities={
            "metal": 0.88,
            "plastic": 0.08,
            "trash": 0.04,
            "glass": 0.0,
            "paper": 0.0,
            "cardboard": 0.0,
        },
        guidance="Item identified as Metal",
        bin_name="Metals & Cans Recycling",
        bin_color="#FBC02D",
        bin_label="Yellow Bin - Metals",
        bin_instructions="Empty and rinse aluminum cans.",
    )
    mock_predictor.predict.return_value = dummy_pred

    pipeline = LiveCameraPipeline(predictor=mock_predictor)
    pipeline.smoother.stable_frames = 2  # Set to 2 frames for fast test

    # 1. Null image -> Standby
    null_res = pipeline.process_frame(None)
    assert null_res.badge_label == "⚪ STANDBY"
    assert not null_res.is_confirmed

    # 2. Dark image -> Quality Alert (Too dark)
    dark_img = Image.new("RGB", (224, 224), color=(10, 10, 10))
    dark_res = pipeline.process_frame(dark_img, enforce_throttle=False)
    assert not dark_res.quality_passed
    assert dark_res.quality_hint == "Too dark"
    assert "🟠 TOO DARK" in dark_res.badge_label

    # 3. High quality sharp image -> Frame 1 (Stabilizing)
    arr = np.zeros((224, 224), dtype=np.uint8)
    for i in range(0, 224, 16):
        for j in range(0, 224, 16):
            arr[i : i + 16, j : j + 16] = 200 if ((i // 16 + j // 16) % 2 == 0) else 60
    sharp_img = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB))

    frame1_res = pipeline.process_frame(sharp_img, enforce_throttle=False)
    assert frame1_res.quality_passed
    assert not frame1_res.is_confirmed
    assert frame1_res.badge_label == "🟡 STABILIZING"

    # Frame 2 -> Confirmed (reaches stable_frames = 2)
    frame2_res = pipeline.process_frame(sharp_img, enforce_throttle=False)
    assert frame2_res.is_confirmed
    assert frame2_res.badge_label == "🟢 CONFIRMED"
    assert frame2_res.bin_label == "Yellow Bin - Metals"
    assert "CONFIRMED: METAL" in frame2_res.status
