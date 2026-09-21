"""Tests for FR-4: Configurable Model Architecture Factory.

Validates:
- Instantiation of all supported backbones: efficientnet_b0, mobilenetv3_large_100, resnet50.
- Forward pass output shape (B, num_classes) and feature extraction shape (B, in_features).
- Classifier head configuration: Global Average Pooling, Dropout (default 0.3 or custom), Linear.
- Dynamic derivation of number of classes from config or dataset without hardcoding.
- Rejection of unsupported backbones and invalid hyperparameters.
- Backbone freezing and progressive unfreezing for fine-tuning phases.
- Seamless instantiation from external YAML configuration files.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from waste_classifier.models.factory import (
    SUPPORTED_BACKBONES,
    WasteClassifier,
    create_model,
    create_model_from_config,
    derive_num_classes,
)


@pytest.mark.parametrize("backbone", sorted(SUPPORTED_BACKBONES))
def test_supported_backbones_instantiation(backbone: str) -> None:
    """Verify all supported backbones instantiate cleanly with default settings."""
    model = create_model(backbone_name=backbone, num_classes=6, pretrained=False)
    assert isinstance(model, WasteClassifier)
    assert isinstance(model, nn.Module)
    assert model.backbone_name == backbone
    assert model.num_classes == 6
    assert model.dropout_rate == 0.3
    assert model.in_features > 0


@pytest.mark.parametrize("backbone", sorted(SUPPORTED_BACKBONES))
@pytest.mark.parametrize("num_classes", [2, 6, 10])
def test_output_tensor_shape(backbone: str, num_classes: int) -> None:
    """Verify forward pass output tensor matches (batch_size, num_classes)."""
    batch_size = 2
    model = create_model(backbone_name=backbone, num_classes=num_classes, pretrained=False)
    model.eval()

    dummy_input = torch.randn(batch_size, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
        features = model.extract_features(dummy_input)

    assert output.shape == (batch_size, num_classes)
    assert features.shape == (batch_size, model.in_features)


def test_custom_dropout_configuration() -> None:
    """Verify custom dropout probability is set correctly in head."""
    model = create_model(
        backbone_name="efficientnet_b0", num_classes=6, pretrained=False, dropout=0.45
    )
    assert model.dropout_rate == 0.45
    assert isinstance(model.head[0], nn.Dropout)
    assert model.head[0].p == 0.45
    assert isinstance(model.head[1], nn.Linear)
    assert model.head[1].out_features == 6
    assert model.head[1].in_features == model.in_features


def test_unsupported_backbone_raises_value_error() -> None:
    """Verify factory raises ValueError when an unsupported backbone is requested."""
    with pytest.raises(ValueError, match="Unsupported backbone 'vgg16'"):
        create_model(backbone_name="vgg16", num_classes=6, pretrained=False)


@pytest.mark.parametrize("invalid_classes", [0, 1, -5])
def test_invalid_num_classes_raises_value_error(invalid_classes: int) -> None:
    """Verify non-positive or single-class target counts are rejected."""
    with pytest.raises(ValueError, match="num_classes must be greater than 1"):
        create_model(backbone_name="resnet50", num_classes=invalid_classes, pretrained=False)


@pytest.mark.parametrize("invalid_dropout", [-0.1, 1.0, 1.5])
def test_invalid_dropout_raises_value_error(invalid_dropout: float) -> None:
    """Verify dropout outside [0.0, 1.0) is rejected."""
    with pytest.raises(ValueError, match="dropout rate must be in"):
        create_model(
            backbone_name="resnet50", num_classes=6, pretrained=False, dropout=invalid_dropout
        )


def test_backbone_freeze_and_unfreeze() -> None:
    """Verify freezing backbone leaves head trainable, and unfreezing restores gradients."""
    model = create_model(backbone_name="resnet50", num_classes=6, pretrained=False)

    # Freeze backbone (Phase 1)
    model.freeze_backbone()
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.head.parameters())

    # Partial unfreeze (Phase 2)
    model.unfreeze_backbone(unfreeze_fraction=0.25)
    trainable_backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    assert len(trainable_backbone) > 0
    assert len(trainable_backbone) < len(list(model.backbone.parameters()))

    # Full unfreeze
    model.unfreeze_backbone(unfreeze_fraction=1.0)
    assert all(p.requires_grad for p in model.backbone.parameters())

    # Invalid fraction
    with pytest.raises(ValueError, match="unfreeze_fraction must be in"):
        model.unfreeze_backbone(unfreeze_fraction=0.0)


def test_derive_num_classes_from_manifest() -> None:
    """Verify class count derivation from splits manifest and raw data."""
    manifest_path = Path("data/splits.csv")
    if manifest_path.exists():
        num_classes = derive_num_classes(manifest_path=manifest_path)
        assert num_classes == 6


def test_derive_num_classes_from_raw_dir() -> None:
    """Verify class count derivation from raw dataset directory."""
    raw_dir = Path("data/raw")
    if raw_dir.exists():
        num_classes = derive_num_classes(data_dir=raw_dir)
        assert num_classes == 6


def test_create_model_from_yaml_configs() -> None:
    """Verify model can be created directly from all project YAML configurations without code edits."""
    configs_to_test = [
        "configs/base.yaml",
        "configs/backbones/efficientnet_b0.yaml",
        "configs/backbones/mobilenetv3.yaml",
        "configs/backbones/resnet50.yaml",
    ]

    for cfg_file in configs_to_test:
        path = Path(cfg_file)
        if path.exists():
            # Pass pretrained=False or override to avoid internet dependency in automated test
            model = create_model_from_config(
                config_source=path,
                num_classes=6,
                pretrained=False,
            )
            assert isinstance(model, WasteClassifier)
            assert model.num_classes == 6

            # Forward check with dummy input
            dummy = torch.randn(1, 3, 224, 224)
            model.eval()
            with torch.no_grad():
                out = model(dummy)
            assert out.shape == (1, 6)


def test_create_model_from_dict_config() -> None:
    """Verify model instantiation from a plain configuration dictionary."""
    cfg = {
        "model": {
            "backbone": "mobilenetv3_large_100",
            "pretrained": False,
            "dropout": 0.25,
            "num_classes": 4,
        }
    }
    model = create_model_from_config(cfg)
    assert model.backbone_name == "mobilenetv3_large_100"
    assert model.num_classes == 4
    assert model.dropout_rate == 0.25
