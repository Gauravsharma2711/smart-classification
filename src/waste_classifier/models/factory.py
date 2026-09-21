"""Model architecture factory module.

Implements FR-4:
- Pretrained backbone selection (efficientnet_b0, mobilenetv3_large_100, resnet50).
- Configurable classifier head with Global Average Pooling (GAP), Dropout (0.3 default), and Linear layer.
- Dynamic number of output classes derived from dataset or configuration (no hardcoding).
- Model selection entirely driven through configuration files without source code modification.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import timm
import torch
import torch.nn as nn
import yaml

from waste_classifier.data.ingestion import discover_classes

logger = logging.getLogger(__name__)

SUPPORTED_BACKBONES: set[str] = {
    "efficientnet_b0",
    "mobilenetv3_large_100",
    "resnet50",
}


class WasteClassifier(nn.Module):
    """Configurable waste classification model architecture.

    Encapsulates a timm backbone with a modular, configurable classifier head
    consisting of global average pooling (maintained by backbone reset),
    dropout, and a final linear projection to the target classes.
    """

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if backbone_name not in SUPPORTED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone '{backbone_name}'. "
                f"Supported backbones are: {sorted(SUPPORTED_BACKBONES)}"
            )

        if num_classes <= 1:
            raise ValueError(
                f"num_classes must be greater than 1 for multi-class classification, got {num_classes}"
            )

        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout rate must be in [0.0, 1.0), got {dropout}")

        self.backbone_name = backbone_name
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.pretrained = pretrained

        # Load backbone
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained)

        # Retrieve feature dimensions before classifier replacement
        orig_classifier = self.backbone.get_classifier()
        if hasattr(orig_classifier, "in_features"):
            self.in_features = orig_classifier.in_features
        elif hasattr(self.backbone, "num_features"):
            self.in_features = self.backbone.num_features
        else:
            raise RuntimeError(
                f"Unable to determine feature dimensions for backbone '{backbone_name}'"
            )

        # Reset backbone classifier to Identity to output pooled feature vector
        self.backbone.reset_classifier(num_classes=0)

        # Custom modular classifier head: Dropout -> Linear
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone feature extractor and classifier head."""
        features = self.backbone(x)
        logits = self.head(features)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract global feature representations prior to the classifier head."""
        return self.backbone(x)

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters for Phase 1 head-only fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info(f"Frozen all backbone parameters for {self.backbone_name}.")

    def unfreeze_backbone(self, unfreeze_fraction: float = 1.0) -> None:
        """Unfreeze top fraction or all backbone parameters for Phase 2 fine-tuning.

        Args:
            unfreeze_fraction: Float in (0.0, 1.0] specifying the fraction of trailing
                backbone parameter tensors to unfreeze.
        """
        if not (0.0 < unfreeze_fraction <= 1.0):
            raise ValueError(f"unfreeze_fraction must be in (0.0, 1.0], got {unfreeze_fraction}")

        params = list(self.backbone.parameters())
        num_to_unfreeze = int(len(params) * unfreeze_fraction)
        if num_to_unfreeze == 0:
            num_to_unfreeze = 1

        cutoff = len(params) - num_to_unfreeze
        for i, param in enumerate(params):
            param.requires_grad = i >= cutoff

        logger.info(
            f"Unfroze {num_to_unfreeze}/{len(params)} parameter tensors "
            f"({unfreeze_fraction * 100:.1f}%) in {self.backbone_name} backbone."
        )


def create_model(
    backbone_name: str,
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> WasteClassifier:
    """Instantiate a WasteClassifier with specified backbone and head parameters.

    Args:
        backbone_name: One of the supported backbones (efficientnet_b0, mobilenetv3_large_100, resnet50).
        num_classes: Number of target output classes.
        pretrained: Whether to load pre-trained ImageNet weights.
        dropout: Dropout probability before the output projection layer.

    Returns:
        Instantiated WasteClassifier module.
    """
    return WasteClassifier(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )


def derive_num_classes(
    manifest_path: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> int:
    """Dynamically determine the number of classes from splits manifest or dataset root.

    Args:
        manifest_path: Path to splits CSV (e.g. data/splits.csv).
        data_dir: Path to raw dataset root (e.g. data/raw).

    Returns:
        Total number of unique classes discovered.
    """
    if manifest_path is not None:
        p = Path(manifest_path)
        if p.exists():
            df = pd.read_csv(p)
            if "class_name" in df.columns:
                classes = df["class_name"].unique().tolist()
                logger.info(
                    f"Derived {len(classes)} classes from manifest '{p}': {sorted(classes)}"
                )
                return len(classes)

    if data_dir is not None:
        p = Path(data_dir)
        if p.exists():
            classes = discover_classes(p)
            logger.info(
                f"Derived {len(classes)} classes from data directory '{p}': {sorted(classes)}"
            )
            return len(classes)

    # Fallback to default locations if available
    default_manifest = Path("data/splits.csv")
    if default_manifest.exists():
        df = pd.read_csv(default_manifest)
        if "class_name" in df.columns:
            return len(df["class_name"].unique())

    default_data = Path("data/raw")
    if default_data.exists():
        classes = discover_classes(default_data)
        return len(classes)

    raise ValueError(
        "Could not automatically derive number of classes: neither valid manifest "
        "nor dataset directory was found."
    )


def create_model_from_config(
    config_source: str | Path | dict[str, Any],
    num_classes: int | None = None,
    pretrained: bool | None = None,
    dropout: float | None = None,
) -> WasteClassifier:
    """Build a WasteClassifier instance directly from configuration without code edits.

    Args:
        config_source: Path to YAML file or configuration dictionary.
        num_classes: Optional explicit class count. If not provided, it is dynamically
            derived from the configuration's data section or local dataset manifest.
        pretrained: Optional override for pretrained weights flag.
        dropout: Optional override for head dropout probability.

    Returns:
        Configured WasteClassifier module.
    """
    if isinstance(config_source, (str, Path)):
        config_path = Path(config_source)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, encoding="utf-8") as f:
            cfg: dict[str, Any] = yaml.safe_load(f)
    elif isinstance(config_source, dict):
        cfg = config_source
    else:
        raise TypeError(
            f"Expected config_source to be str, Path, or dict, got {type(config_source)}"
        )

    # Model parameters can be under 'model' key or at the top level
    model_cfg = cfg.get("model", cfg)
    backbone_name = model_cfg.get("backbone")
    if not backbone_name:
        raise ValueError("Configuration missing required 'backbone' field under model.")

    if pretrained is None:
        pretrained = model_cfg.get("pretrained", True)
    if dropout is None:
        dropout = model_cfg.get("dropout", 0.3)

    # Resolve number of classes dynamically
    target_num_classes = num_classes
    if target_num_classes is None:
        target_num_classes = model_cfg.get("num_classes")

    if target_num_classes is None:
        data_cfg = cfg.get("data", {})
        manifest_path = data_cfg.get("manifest")
        data_root = data_cfg.get("root")
        try:
            target_num_classes = derive_num_classes(
                manifest_path=manifest_path,
                data_dir=data_root,
            )
        except Exception as err:
            raise ValueError(
                "Unable to derive num_classes from config or dataset. "
                "Specify 'num_classes' in model config or pass it explicitly."
            ) from err

    logger.info(
        f"Creating model from config: backbone={backbone_name}, "
        f"num_classes={target_num_classes}, pretrained={pretrained}, dropout={dropout}"
    )

    return create_model(
        backbone_name=backbone_name,
        num_classes=target_num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )
