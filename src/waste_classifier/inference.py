"""Production inference module for waste classification models.

Implements FR-9:
- predict(image) -> top-k classes with probabilities.
- PIL Image input with defensive validation.
- Guaranteed preprocessing parity with evaluation pipeline.
- Configurable top-k and confidence threshold.
- UI-friendly confidence gating and bin recommendation guidance.
- Strict decoupling: zero UI dependency and zero training logic.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

from waste_classifier.data.transforms import build_eval_transforms
from waste_classifier.evaluate import find_best_evaluation_checkpoint
from waste_classifier.models.factory import create_model

logger = logging.getLogger(__name__)

CANONICAL_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
DEFAULT_BIN_MAPPING_PATH = Path("configs/bin_mapping.yaml")
DEFAULT_CONFIG_PATH = Path("configs/base.yaml")


@dataclass
class PredictionItem:
    """Individual class prediction item within top-k output."""

    class_name: str
    probability: float
    index: int
    bin_name: str
    bin_color: str
    bin_label: str


@dataclass
class PredictionResult:
    """Structured container for model inference outputs, confidence metrics, and UI advice."""

    top_class: str
    confidence: float
    is_confident: bool
    predictions: list[PredictionItem]
    all_probabilities: dict[str, float]
    guidance: str
    bin_name: str
    bin_color: str
    bin_label: str
    bin_instructions: str

    def to_dict(self) -> dict[str, Any]:
        """Convert prediction result to a serializable dictionary."""
        return asdict(self)


def load_bin_mapping(bin_mapping_path: Path | str | None = None) -> dict[str, dict[str, str]]:
    """Load disposal bin mapping information from YAML configuration file."""
    p = Path(bin_mapping_path) if bin_mapping_path else DEFAULT_BIN_MAPPING_PATH
    if p.exists():
        try:
            with open(p, encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data.get("categories", {})
        except Exception as err:
            logger.warning(f"Could not load bin mapping from {p}: {err}")

    # Built-in fallbacks if file not found
    return {
        "cardboard": {
            "bin_name": "Paper & Cardboard Recycling",
            "text_label": "Blue Bin - Recyclables",
            "color_hex": "#1976D2",
            "instructions": "Flatten cardboard boxes before placing into the recycling bin.",
        },
        "glass": {
            "bin_name": "Glass Recycling",
            "text_label": "Teal/Green Bin - Glass",
            "color_hex": "#00897B",
            "instructions": "Rinse bottles and jars thoroughly. Remove metal or plastic lids.",
        },
        "metal": {
            "bin_name": "Metals & Cans Recycling",
            "text_label": "Yellow Bin - Metals",
            "color_hex": "#FBC02D",
            "instructions": "Empty and rinse aluminum cans and food tins.",
        },
        "paper": {
            "bin_name": "Paper Recycling",
            "text_label": "Blue Bin - Paper",
            "color_hex": "#1E88E5",
            "instructions": "Clean office paper, newspapers, magazines.",
        },
        "plastic": {
            "bin_name": "Plastics Recycling",
            "text_label": "Orange Bin - Plastics",
            "color_hex": "#FB8C00",
            "instructions": "Empty liquids and crush plastic bottles.",
        },
        "trash": {
            "bin_name": "General Waste / Landfill",
            "text_label": "Black/Gray Bin - General Waste",
            "color_hex": "#616161",
            "instructions": "Non-recyclable items or composite packaging.",
        },
        "uncertain": {
            "bin_name": "Uncertain / Verification Needed",
            "text_label": "Gray Bin - Inspect Item",
            "color_hex": "#9E9E9E",
            "instructions": "Item could not be identified with high confidence. Try another angle or inspect manually.",
        },
    }


class WasteClassifierPredictor:
    """Production predictor encapsulating model weights, preprocessing, and inference execution."""

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        config_path: Path | str = DEFAULT_CONFIG_PATH,
        device: str = "cpu",
        top_k: int | None = None,
        confidence_threshold: float | None = None,
        bin_mapping_path: Path | str | None = None,
    ) -> None:
        """Initialize and warm up predictor instance.

        Args:
            checkpoint_path: Path to checkpoint file (auto-detected if None).
            config_path: Path to base YAML configuration file.
            device: Computing device ('cpu' or 'cuda').
            top_k: Override number of top predictions to return.
            confidence_threshold: Override confidence threshold for certainty gating.
            bin_mapping_path: Optional path to custom bin_mapping.yaml.
        """
        self.device = torch.device(device)
        self.config_path = Path(config_path)

        # 1. Load Configuration
        if self.config_path.exists():
            with open(self.config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        else:
            logger.warning(f"Config file not found at {self.config_path}, using defaults.")
            cfg = {}

        data_cfg = cfg.get("data", {})
        model_cfg = cfg.get("model", {})
        infer_cfg = cfg.get("inference", {})

        self.backbone_name = model_cfg.get("backbone", "efficientnet_b0")
        self.image_size = int(data_cfg.get("image_size", 224))
        self.top_k = int(top_k if top_k is not None else infer_cfg.get("top_k", 3))
        self.confidence_threshold = float(
            confidence_threshold
            if confidence_threshold is not None
            else infer_cfg.get("confidence_threshold", 0.60)
        )
        self.class_names = list(CANONICAL_CLASSES)
        self.num_classes = len(self.class_names)

        # 2. Setup Deterministic Preprocessing (exact parity with evaluation)
        self.transform = build_eval_transforms(image_size=self.image_size)

        # 3. Load Bin Mapping
        self.bin_mapping = load_bin_mapping(bin_mapping_path)

        # 4. Resolve Checkpoint and Load Model
        # 4. Resolve Checkpoint and Load Model
        try:
            self.checkpoint_path = find_best_evaluation_checkpoint(checkpoint_path)
            logger.info(f"Loading inference model checkpoint from: {self.checkpoint_path}")

            self.model = create_model(
                backbone_name=self.backbone_name,
                num_classes=self.num_classes,
                pretrained=False,
                dropout=float(model_cfg.get("dropout", 0.3)),
            )

            ckpt_data = torch.load(self.checkpoint_path, map_location="cpu")
            model_state = {}
            for k, v in ckpt_data["state_dict"].items():
                if k.startswith("model."):
                    model_state[k[len("model.") :]] = v
                elif (
                    not k.startswith("criterion.")
                    and not k.startswith("train_")
                    and not k.startswith("val_")
                    and k != "class_weights"
                ):
                    model_state[k] = v

            self.model.load_state_dict(model_state)
        except FileNotFoundError as err:
            logger.warning(
                f"No fine-tuned model checkpoint found ({err}). "
                f"Initializing {self.backbone_name} with pretrained backbone weights for deployment demonstration."
            )
            self.checkpoint_path = None
            self.model = create_model(
                backbone_name=self.backbone_name,
                num_classes=self.num_classes,
                pretrained=True,
                dropout=float(model_cfg.get("dropout", 0.3)),
            )

        self.model.to(self.device)
        self.model.eval()

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Validate and preprocess PIL Image into normalized model input tensor.

        Args:
            image: PIL Image instance.

        Returns:
            Preprocessed 4D torch.Tensor with batch dimension [1, 3, H, W].
        """
        if image is None:
            raise ValueError("Input image cannot be None.")

        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected image of type PIL.Image.Image, got {type(image).__name__}.")

        if image.width <= 0 or image.height <= 0:
            raise ValueError(f"Image has invalid dimensions: {image.size}.")

        # Convert to RGB mode safely handling Grayscale, Palette, or RGBA inputs
        rgb_image = image.convert("RGB")
        img_arr = np.array(rgb_image)

        # Apply deterministic evaluation transforms
        transformed = self.transform(image=img_arr)
        tensor: torch.Tensor = transformed["image"]

        # Add batch dimension: [3, 224, 224] -> [1, 3, 224, 224]
        return tensor.unsqueeze(0).to(self.device)

    def predict(
        self,
        image: Image.Image,
        top_k: int | None = None,
        confidence_threshold: float | None = None,
    ) -> PredictionResult:
        """Execute deterministic inference on a PIL image.

        Args:
            image: PIL Image input.
            top_k: Optional per-call override for number of top predictions.
            confidence_threshold: Optional per-call override for confidence threshold.

        Returns:
            Populated PredictionResult with top-k predictions and UI guidance.
        """
        tensor = self.preprocess_image(image)

        effective_top_k = int(top_k if top_k is not None else self.top_k)
        effective_threshold = float(
            confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        )

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        # Ensure strict normalization
        probs = np.clip(probs, 0.0, 1.0)
        prob_sum = float(np.sum(probs))
        if prob_sum > 0:
            probs = probs / prob_sum

        # Build full probability dictionary
        all_probabilities: dict[str, float] = {
            cls_name: round(float(probs[idx]), 4) for idx, cls_name in enumerate(self.class_names)
        }

        # Sort indices in descending order of probability
        sorted_indices = np.argsort(probs)[::-1]
        top_idx = int(sorted_indices[0])
        top_class = self.class_names[top_idx]
        confidence = round(float(probs[top_idx]), 4)
        is_confident = confidence >= effective_threshold

        # Compile top-k PredictionItems
        effective_k = min(effective_top_k, self.num_classes)
        top_items: list[PredictionItem] = []
        for i in range(effective_k):
            idx = int(sorted_indices[i])
            cls_name = self.class_names[idx]
            cls_prob = round(float(probs[idx]), 4)
            b_info = self.bin_mapping.get(cls_name, {})
            top_items.append(
                PredictionItem(
                    class_name=cls_name,
                    probability=cls_prob,
                    index=idx,
                    bin_name=b_info.get("bin_name", cls_name.title()),
                    bin_color=b_info.get("color_hex", "#9E9E9E"),
                    bin_label=b_info.get("text_label", f"{cls_name.title()} Bin"),
                )
            )

        # Guidance and bin recommendation determination
        if is_confident:
            bin_info = self.bin_mapping.get(top_class, self.bin_mapping.get("uncertain", {}))
            bin_name = bin_info.get("bin_name", top_class.title())
            bin_color = bin_info.get("color_hex", "#9E9E9E")
            bin_label = bin_info.get("text_label", f"{top_class.title()} Bin")
            bin_instructions = bin_info.get("instructions", f"Place item in the {bin_name}.")
            guidance = (
                f"Item identified as {top_class.title()} with {confidence * 100:.1f}% confidence."
            )
        else:
            uncertain_info = self.bin_mapping.get("uncertain", {})
            bin_name = uncertain_info.get("bin_name", "Uncertain / Verification Needed")
            bin_color = uncertain_info.get("color_hex", "#9E9E9E")
            bin_label = uncertain_info.get("text_label", "Gray Bin - Inspect Item")
            bin_instructions = uncertain_info.get(
                "instructions",
                f"Confidence ({confidence * 100:.1f}%) is below threshold ({effective_threshold * 100:.0f}%). "
                f"Best guess is {top_class.title()}, but manual verification is recommended.",
            )
            guidance = (
                f"Low confidence ({confidence * 100:.1f}% < {effective_threshold * 100:.0f}%). "
                "Point the camera directly at the item, improve lighting, or adjust the angle."
            )

        return PredictionResult(
            top_class=top_class,
            confidence=confidence,
            is_confident=is_confident,
            predictions=top_items,
            all_probabilities=all_probabilities,
            guidance=guidance,
            bin_name=bin_name,
            bin_color=bin_color,
            bin_label=bin_label,
            bin_instructions=bin_instructions,
        )


# Global singleton instance for efficient single-load serving
_DEFAULT_PREDICTOR: WasteClassifierPredictor | None = None


def get_predictor(
    checkpoint_path: Path | str | None = None,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    device: str = "cpu",
    top_k: int | None = None,
    confidence_threshold: float | None = None,
    bin_mapping_path: Path | str | None = None,
    force_reload: bool = False,
) -> WasteClassifierPredictor:
    """Retrieve or initialize the cached WasteClassifierPredictor instance.

    Args:
        checkpoint_path: Optional explicit model checkpoint path.
        config_path: Path to configuration YAML file.
        device: Device to run inference on.
        top_k: Number of top candidate predictions to return.
        confidence_threshold: Minimum threshold to mark prediction as confident.
        bin_mapping_path: Optional path to bin_mapping.yaml.
        force_reload: Force re-initialization of predictor.

    Returns:
        Configured WasteClassifierPredictor instance.
    """
    global _DEFAULT_PREDICTOR
    if _DEFAULT_PREDICTOR is None or force_reload or checkpoint_path is not None:
        _DEFAULT_PREDICTOR = WasteClassifierPredictor(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device=device,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
            bin_mapping_path=bin_mapping_path,
        )
    return _DEFAULT_PREDICTOR


def predict(
    image: Image.Image,
    checkpoint_path: Path | str | None = None,
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    device: str = "cpu",
    top_k: int | None = None,
    confidence_threshold: float | None = None,
) -> PredictionResult:
    """Predict waste classification top-k classes and probabilities for a PIL image.

    Args:
        image: PIL Image instance.
        checkpoint_path: Optional explicit model checkpoint path.
        config_path: Path to configuration YAML file.
        device: Device to run inference on ('cpu' or 'cuda').
        top_k: Number of top candidate predictions to return.
        confidence_threshold: Minimum threshold to mark prediction as confident.

    Returns:
        Structured PredictionResult instance.
    """
    predictor = get_predictor(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=device,
    )
    return predictor.predict(
        image=image,
        top_k=top_k,
        confidence_threshold=confidence_threshold,
    )
