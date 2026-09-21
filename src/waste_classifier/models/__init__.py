"""Model definitions, architecture factory, and Lightning modules."""

from waste_classifier.models.factory import (
    SUPPORTED_BACKBONES,
    WasteClassifier,
    create_model,
    create_model_from_config,
    derive_num_classes,
)
from waste_classifier.models.module import WasteLightningModule

__all__ = [
    "SUPPORTED_BACKBONES",
    "WasteClassifier",
    "WasteLightningModule",
    "create_model",
    "create_model_from_config",
    "derive_num_classes",
]
