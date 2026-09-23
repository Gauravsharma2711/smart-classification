"""Live camera pipeline with quality gating, temporal smoothing, and stability confirmation.

Implements:
- FR-13: Throttled live camera mode (~2-5 FPS).
- FR-14: Temporal probability smoothing and stability confirmation (K consecutive frames).
- FR-15: Quality gating (Laplacian blur variance and brightness check).
- FR-16: Uncertainty and no-item handling ("Point the camera at an item").

Architecture:
camera frame -> center crop -> quality gate -> preprocessing -> inference
-> probability smoothing -> stability confirmation -> uncertainty/no-item handling -> UI advice
"""

from __future__ import annotations

import collections
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PIL import Image

from waste_classifier.inference import (
    CANONICAL_CLASSES,
    DEFAULT_BIN_MAPPING_PATH,
    PredictionItem,
    WasteClassifierPredictor,
    get_predictor,
    load_bin_mapping,
)

logger = logging.getLogger(__name__)

DEFAULT_LIVE_CONFIG_PATH = Path("configs/live.yaml")


def to_grayscale_array(image: Image.Image | np.ndarray) -> np.ndarray:
    """Convert a PIL Image or NumPy array to a 2D uint8 grayscale NumPy array."""
    if isinstance(image, Image.Image):
        return np.array(image.convert("L"), dtype=np.uint8)
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return image.astype(np.uint8)
        if image.ndim == 3:
            # Check channel count
            if image.shape[2] == 3:
                return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            if image.shape[2] == 4:
                return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGBA2GRAY)
    raise TypeError(f"Unsupported image type or shape: {type(image)}")


def center_crop(
    image: Image.Image | np.ndarray, crop_ratio: float = 0.8
) -> Image.Image | np.ndarray:
    """Extract a centered crop box representing the camera guide target area.

    Args:
        image: PIL Image or NumPy array [H, W, C].
        crop_ratio: Fraction of width and height to retain (0 < crop_ratio <= 1.0).

    Returns:
        Center-cropped image of the same type as input.
    """
    if not (0.0 < crop_ratio <= 1.0):
        raise ValueError(f"crop_ratio must be in (0, 1.0], got {crop_ratio}")

    if isinstance(image, Image.Image):
        w, h = image.size
        new_w = max(1, int(w * crop_ratio))
        new_h = max(1, int(h * crop_ratio))
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        return image.crop((left, top, left + new_w, top + new_h))

    if isinstance(image, np.ndarray):
        h, w = image.shape[:2]
        new_w = max(1, int(w * crop_ratio))
        new_h = max(1, int(h * crop_ratio))
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        return image[top : top + new_h, left : left + new_w]

    raise TypeError(f"Unsupported image type: {type(image)}")


def compute_blur_score(image: Image.Image | np.ndarray) -> float:
    """Compute blur metric as the variance of the Laplacian (higher = sharper).

    Args:
        image: PIL Image or NumPy image array.

    Returns:
        Non-negative float variance of the Laplacian.
    """
    gray = to_grayscale_array(image)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def compute_brightness_score(image: Image.Image | np.ndarray) -> float:
    """Compute average brightness as the mean pixel intensity across grayscale channels.

    Args:
        image: PIL Image or NumPy image array.

    Returns:
        Float value in range [0.0, 255.0].
    """
    gray = to_grayscale_array(image)
    return float(np.mean(gray))


@dataclass
class QualityResult:
    """Quality gate inspection result."""

    passed: bool
    blur_score: float
    brightness: float
    hint: str | None = None


class QualityGate:
    """Quality gate validating frame readiness before expensive model inference."""

    def __init__(
        self,
        min_blur_score: float = 60.0,
        min_brightness: float = 50.0,
        max_brightness: float = 250.0,
    ) -> None:
        self.min_blur_score = float(min_blur_score)
        self.min_brightness = float(min_brightness)
        self.max_brightness = float(max_brightness)

    def check(self, image: Image.Image | np.ndarray) -> QualityResult:
        """Inspect image for sufficient lighting and sharpness.

        Args:
            image: PIL Image or NumPy array.

        Returns:
            QualityResult indicating pass/fail with actionable UI hint.
        """
        brightness = compute_brightness_score(image)
        if brightness < self.min_brightness:
            return QualityResult(
                passed=False,
                blur_score=0.0,
                brightness=round(brightness, 2),
                hint="Too dark",
            )
        if brightness > self.max_brightness:
            return QualityResult(
                passed=False,
                blur_score=0.0,
                brightness=round(brightness, 2),
                hint="Too bright",
            )

        blur_score = compute_blur_score(image)
        if blur_score < self.min_blur_score:
            return QualityResult(
                passed=False,
                blur_score=round(blur_score, 2),
                brightness=round(brightness, 2),
                hint="Hold steady",
            )

        return QualityResult(
            passed=True,
            blur_score=round(blur_score, 2),
            brightness=round(brightness, 2),
            hint=None,
        )


class Throttler:
    """Rate limiter ensuring live inference does not exceed target frame rates."""

    def __init__(self, target_fps: float = 4.0) -> None:
        self.target_fps = max(0.1, float(target_fps))
        self.min_interval = 1.0 / self.target_fps
        self._last_processed_time = 0.0

    def should_process(self, current_time: float | None = None) -> bool:
        """Determine whether enough time has elapsed to process a new frame."""
        now = time.time() if current_time is None else current_time
        if (now - self._last_processed_time) >= self.min_interval:
            self._last_processed_time = now
            return True
        return False

    def reset(self) -> None:
        """Reset internal timestamp."""
        self._last_processed_time = 0.0


@dataclass
class SmoothedResult:
    """Output from the temporal FrameSmoother."""

    top_class: str
    confidence: float
    is_confirmed: bool
    consecutive_count: int
    probabilities: dict[str, float]
    guidance: str
    is_uncertain: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize smoothed result to dictionary."""
        return asdict(self)


class FrameSmoother:
    """Pure-Python sliding window probability smoother and stability confirmation engine.

    Decoupled from all UI and PyTorch dependencies for exhaustive unit testing.
    """

    def __init__(
        self,
        smoothing_window: int = 5,
        stable_frames: int = 3,
        confidence_threshold: float = 0.60,
        class_names: list[str] | None = None,
        background_class: str | None = None,
    ) -> None:
        """Initialize frame smoother.

        Args:
            smoothing_window: Size N of sliding window for rolling probability average.
            stable_frames: Number K of consecutive identical top-class frames required.
            confidence_threshold: Minimum smoothed probability to consider prediction confident.
            class_names: Canonical list of class labels.
            background_class: Optional label designating empty scene or background.
        """
        self.smoothing_window = max(1, int(smoothing_window))
        self.stable_frames = max(1, int(stable_frames))
        self.confidence_threshold = float(confidence_threshold)
        self.class_names = list(class_names) if class_names else list(CANONICAL_CLASSES)
        self.background_class = background_class

        self._window: collections.deque[dict[str, float]] = collections.deque(
            maxlen=self.smoothing_window
        )
        self._last_top_class: str | None = None
        self._consecutive_count: int = 0
        self._confirmed_class: str | None = None

    def update(self, probs: dict[str, float] | np.ndarray) -> SmoothedResult:
        """Ingest single-frame class probabilities, apply smoothing, and verify stability.

        Args:
            probs: Dictionary of {class_name: probability} or 1D array matching class_names.

        Returns:
            SmoothedResult detailing stabilized prediction and confirmation state.
        """
        # Convert array input to dict if necessary
        if isinstance(probs, np.ndarray):
            prob_dict = {
                cls: float(probs[i]) if i < len(probs) else 0.0
                for i, cls in enumerate(self.class_names)
            }
        elif isinstance(probs, dict):
            prob_dict = {cls: float(probs.get(cls, 0.0)) for cls in self.class_names}
        else:
            raise TypeError(f"Expected dict or np.ndarray for probs, got {type(probs)}")

        # 1. Slide window
        self._window.append(prob_dict)
        n = len(self._window)

        # 2. Average probabilities across window
        avg_probs: dict[str, float] = {}
        for cls in self.class_names:
            avg_probs[cls] = sum(frame[cls] for frame in self._window) / n

        # Normalize to ensure strict probability distribution
        total_p = sum(avg_probs.values())
        if total_p > 0:
            avg_probs = {cls: p / total_p for cls, p in avg_probs.items()}

        # 3. Identify smoothed top class
        sorted_classes = sorted(avg_probs.items(), key=lambda item: item[1], reverse=True)
        top_class, top_conf = sorted_classes[0]
        top_conf = round(float(top_conf), 4)

        # 4. Track consecutive identical frames
        if top_class == self._last_top_class:
            self._consecutive_count += 1
        else:
            self._last_top_class = top_class
            self._consecutive_count = 1

        # 5. Determine uncertainty and confirmation
        is_background = self.background_class is not None and top_class == self.background_class
        is_low_confidence = top_conf < self.confidence_threshold
        is_uncertain = is_background or is_low_confidence

        is_confirmed = (not is_uncertain) and (self._consecutive_count >= self.stable_frames)

        if is_confirmed:
            self._confirmed_class = top_class
            guidance = f"Item confirmed as {top_class.title()} ({top_conf * 100:.1f}% confidence)."
        elif is_uncertain:
            guidance = "Point the camera at an item"
        else:
            guidance = (
                f"Analyzing {top_class.title()}... "
                f"({self._consecutive_count}/{self.stable_frames} stable frames)"
            )

        return SmoothedResult(
            top_class=top_class,
            confidence=top_conf,
            is_confirmed=is_confirmed,
            consecutive_count=self._consecutive_count,
            probabilities={k: round(v, 4) for k, v in avg_probs.items()},
            guidance=guidance,
            is_uncertain=is_uncertain,
        )

    def reset(self) -> None:
        """Reset internal window and stability counters."""
        self._window.clear()
        self._last_top_class = None
        self._consecutive_count = 0
        self._confirmed_class = None


@dataclass
class LiveFrameResult:
    """Full end-to-end result of a live frame processing cycle."""

    status: str
    badge_label: str
    badge_color: str
    is_confirmed: bool
    quality_passed: bool
    quality_hint: str | None
    top_class: str
    confidence: float
    top_predictions: list[PredictionItem]
    all_probabilities: dict[str, float]
    guidance: str
    bin_name: str
    bin_color: str
    bin_label: str
    bin_instructions: str
    is_throttled: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert live frame result to dictionary."""
        return asdict(self)


class LiveCameraPipeline:
    """End-to-end orchestrator connecting camera capture, quality checks, predictor, smoother, and bin mapping."""

    def __init__(
        self,
        config_path: Path | str = DEFAULT_LIVE_CONFIG_PATH,
        predictor: WasteClassifierPredictor | None = None,
        bin_mapping_path: Path | str | None = None,
    ) -> None:
        """Initialize live camera pipeline from YAML config."""
        p = Path(config_path)
        live_cfg: dict[str, Any] = {}
        if p.exists():
            try:
                with open(p, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    live_cfg = data.get("live", {})
            except Exception as err:
                logger.warning(f"Could not load live configuration from {p}: {err}")

        # Parameters
        self.target_fps = float(live_cfg.get("target_fps", 4.0))
        self.smoothing_window = int(live_cfg.get("smoothing_window", 5))
        self.stable_frames = int(live_cfg.get("stable_frames", 3))
        self.confidence_threshold = float(live_cfg.get("confidence_threshold", 0.60))
        self.crop_ratio = float(live_cfg.get("crop_ratio", 0.80))

        qual_cfg = live_cfg.get("quality", {})
        min_blur = float(qual_cfg.get("min_blur_score", 60.0))
        min_bright = float(qual_cfg.get("min_brightness", 50.0))

        # Core Components
        self.quality_gate = QualityGate(min_blur_score=min_blur, min_brightness=min_bright)
        self.throttler = Throttler(target_fps=self.target_fps)
        self.smoother = FrameSmoother(
            smoothing_window=self.smoothing_window,
            stable_frames=self.stable_frames,
            confidence_threshold=self.confidence_threshold,
            class_names=CANONICAL_CLASSES,
        )
        self.predictor = predictor if predictor is not None else get_predictor()
        self.bin_mapping = load_bin_mapping(bin_mapping_path or DEFAULT_BIN_MAPPING_PATH)
        self._last_result: LiveFrameResult | None = None

    def reset(self) -> None:
        """Reset internal pipeline states."""
        self.throttler.reset()
        self.smoother.reset()
        self._last_result = None

    def process_frame(
        self,
        image: Image.Image | np.ndarray | None,
        enforce_throttle: bool = True,
    ) -> LiveFrameResult:
        """Execute complete live processing cycle on a single camera frame.

        Args:
            image: Incoming video frame (PIL Image or NumPy array).
            enforce_throttle: If True, drop frames that exceed target FPS.

        Returns:
            LiveFrameResult containing status, UI badges, and disposal advice.
        """
        # 1. Null image handling
        if image is None:
            return self._build_standby_result(
                "No video signal received. Start webcam or hold item in view."
            )

        # 2. Frame throttling
        if enforce_throttle and not self.throttler.should_process():
            if self._last_result is not None:
                # Return cached result with throttled flag
                cached = self._last_result
                cached.is_throttled = True
                return cached

        # Convert numpy to PIL for predictor if needed
        pil_frame = (
            image if isinstance(image, Image.Image) else Image.fromarray(image.astype(np.uint8))
        )

        # 3. Center crop to guide box
        cropped_frame = center_crop(pil_frame, crop_ratio=self.crop_ratio)

        # 4. Quality Gate inspection
        quality = self.quality_gate.check(cropped_frame)
        if not quality.passed:
            # Quality failed: reset stability count to avoid locking on motion blur
            self.smoother._consecutive_count = 0
            res = self._build_quality_alert_result(quality)
            self._last_result = res
            return res

        # 5. Model Inference Forward Pass
        pred_res = self.predictor.predict(cropped_frame)

        # 6. Temporal Probability Smoothing & Stability Confirmation
        smoothed = self.smoother.update(pred_res.all_probabilities)

        # 7. Assemble Live Result
        res = self._build_live_result(smoothed)
        self._last_result = res
        return res

    def _build_standby_result(self, guidance: str) -> LiveFrameResult:
        """Create a default standby result when no video frame is active."""
        uncertain_bin = self.bin_mapping.get("uncertain", {})
        return LiveFrameResult(
            status="Awaiting Camera...",
            badge_label="⚪ STANDBY",
            badge_color="#757575",
            is_confirmed=False,
            quality_passed=True,
            quality_hint=None,
            top_class="none",
            confidence=0.0,
            top_predictions=[],
            all_probabilities={c: 0.0 for c in CANONICAL_CLASSES},
            guidance=guidance,
            bin_name=uncertain_bin.get("bin_name", "Standby"),
            bin_color=uncertain_bin.get("color_hex", "#9E9E9E"),
            bin_label=uncertain_bin.get("text_label", "Gray Bin - Standby"),
            bin_instructions="Position an item steadily within the camera view to identify.",
        )

    def _build_quality_alert_result(self, quality: QualityResult) -> LiveFrameResult:
        """Create result for frames failing quality checks (too dark or blurry)."""
        hint = quality.hint or "Quality Alert"
        badge_label = f"🟠 {hint.upper()}"
        guidance = f"{hint}. Adjust camera or hold steady for automated classification."
        uncertain_bin = self.bin_mapping.get("uncertain", {})

        return LiveFrameResult(
            status=f"Quality Alert: {hint}",
            badge_label=badge_label,
            badge_color="#F57C00",
            is_confirmed=False,
            quality_passed=False,
            quality_hint=quality.hint,
            top_class="none",
            confidence=0.0,
            top_predictions=[],
            all_probabilities={c: 0.0 for c in CANONICAL_CLASSES},
            guidance=guidance,
            bin_name=uncertain_bin.get("bin_name", "Adjust Item"),
            bin_color="#F57C00",
            bin_label="Orange Alert - Adjust Camera",
            bin_instructions=f"{hint}. Ensure good lighting and keep the item still.",
        )

    def _build_live_result(self, smoothed: SmoothedResult) -> LiveFrameResult:
        """Assemble complete result from smoothed prediction."""
        top_cls = smoothed.top_class
        conf = smoothed.confidence

        # Sort top-k items
        sorted_probs = sorted(
            smoothed.probabilities.items(), key=lambda item: item[1], reverse=True
        )
        top_items: list[PredictionItem] = []
        for idx, (cls_name, prob) in enumerate(sorted_probs[:3]):
            b_info = self.bin_mapping.get(cls_name, {})
            top_items.append(
                PredictionItem(
                    class_name=cls_name,
                    probability=prob,
                    index=idx,
                    bin_name=b_info.get("bin_name", cls_name.title()),
                    bin_color=b_info.get("color_hex", "#9E9E9E"),
                    bin_label=b_info.get("text_label", f"{cls_name.title()} Bin"),
                )
            )

        if smoothed.is_confirmed:
            badge_label = "🟢 CONFIRMED"
            badge_color = "#2E7D32"
            status = f"CONFIRMED: {top_cls.upper()} ({conf * 100:.1f}%)"
            bin_info = self.bin_mapping.get(top_cls, self.bin_mapping.get("uncertain", {}))
            bin_name = bin_info.get("bin_name", top_cls.title())
            bin_color = bin_info.get("color_hex", "#2E7D32")
            bin_label = bin_info.get("text_label", f"{top_cls.title()} Bin")
            bin_instructions = bin_info.get("instructions", f"Place item into {bin_name}.")
        elif smoothed.is_uncertain:
            badge_label = "⚪ UNCERTAIN"
            badge_color = "#757575"
            status = "SCANNING: Point camera at item"
            uncertain_bin = self.bin_mapping.get("uncertain", {})
            bin_name = uncertain_bin.get("bin_name", "Uncertain / Verification Needed")
            bin_color = uncertain_bin.get("color_hex", "#9E9E9E")
            bin_label = uncertain_bin.get("text_label", "Gray Bin - Inspect Item")
            bin_instructions = uncertain_bin.get(
                "instructions",
                "Confidence is low. Point camera directly at the waste item.",
            )
        else:
            badge_label = "🟡 STABILIZING"
            badge_color = "#FBC02D"
            status = f"ANALYZING: {top_cls.upper()} ({conf * 100:.1f}%)"
            bin_info = self.bin_mapping.get(top_cls, {})
            bin_name = f"Likely {bin_info.get('bin_name', top_cls.title())}"
            bin_color = bin_info.get("color_hex", "#9E9E9E")
            bin_label = bin_info.get("text_label", f"{top_cls.title()} Bin")
            bin_instructions = f"Stabilizing detection for {top_cls.title()}..."

        return LiveFrameResult(
            status=status,
            badge_label=badge_label,
            badge_color=badge_color,
            is_confirmed=smoothed.is_confirmed,
            quality_passed=True,
            quality_hint=None,
            top_class=top_cls,
            confidence=conf,
            top_predictions=top_items,
            all_probabilities=smoothed.probabilities,
            guidance=smoothed.guidance,
            bin_name=bin_name,
            bin_color=bin_color,
            bin_label=bin_label,
            bin_instructions=bin_instructions,
        )
