"""Unit tests for Grad-CAM Explainability engine (FR-17).

Verifies:
1. Target convolutional layer resolution across supported backbones.
2. Error handling on unsupported or non-convolutional architectures.
3. Grad-CAM heatmap generation, normalization, and overlay geometry.
4. Explaining top predicted class (correct and incorrect predictions).
5. Counterfactual / alternative class targeting for error inspection.
6. Methodological interpretability disclaimer.
7. Report figures generator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch.nn as nn
from PIL import Image

from waste_classifier.explainability import (
    EXPLAINABILITY_DISCLAIMER,
    GradCAMExplainer,
    GradCAMResult,
    generate_gradcam_report_samples,
    resolve_target_layer,
)
from waste_classifier.inference import CANONICAL_CLASSES, get_predictor
from waste_classifier.models.factory import create_model


def test_target_layer_resolution_supported_backbones() -> None:
    """Verify resolve_target_layer resolves the appropriate final conv layer for all backbones."""
    # 1. EfficientNet-B0
    m_eff = create_model("efficientnet_b0", num_classes=6, pretrained=False)
    layers_eff = resolve_target_layer(m_eff, "efficientnet_b0")
    assert len(layers_eff) == 1
    assert layers_eff[0] == m_eff.backbone.conv_head

    # 2. MobileNetV3
    m_mob = create_model("mobilenetv3_large_100", num_classes=6, pretrained=False)
    layers_mob = resolve_target_layer(m_mob, "mobilenetv3_large_100")
    assert len(layers_mob) == 1
    assert layers_mob[0] == m_mob.backbone.conv_head

    # 3. ResNet50
    m_res = create_model("resnet50", num_classes=6, pretrained=False)
    layers_res = resolve_target_layer(m_res, "resnet50")
    assert len(layers_res) == 1
    assert layers_res[0] == m_res.backbone.layer4[-1]


def test_target_layer_resolution_unsupported_model() -> None:
    """Verify resolve_target_layer raises ValueError on architectures without conv layers."""
    linear_model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    with pytest.raises(ValueError, match="Unable to resolve target convolutional layer"):
        resolve_target_layer(linear_model, backbone_name="mlp")


def test_gradcam_on_synthetic_image() -> None:
    """Verify GradCAMExplainer generates normalized heatmaps and matching overlay images."""
    predictor = get_predictor()
    explainer = GradCAMExplainer(predictor=predictor)

    # Synthetic RGB image of arbitrary dimensions (e.g. 300x400)
    arr = np.random.randint(50, 220, (300, 400, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")

    res = explainer.explain(img)

    assert isinstance(res, GradCAMResult)
    # Heatmap is 2D float array normalized in [0, 1]
    assert isinstance(res.heatmap, np.ndarray)
    assert res.heatmap.ndim == 2
    assert res.heatmap.shape == (224, 224)
    assert 0.0 <= res.heatmap.min()
    assert res.heatmap.max() <= 1.0

    # Overlay matches input image size
    assert isinstance(res.overlay_image, Image.Image)
    assert res.overlay_image.size == (400, 300)

    # Class details
    assert res.predicted_class in CANONICAL_CLASSES
    assert res.target_class == res.predicted_class
    assert 0.0 <= res.predicted_prob <= 1.0

    # Disclaimer is present
    assert EXPLAINABILITY_DISCLAIMER in res.disclaimer
    assert "causal" in res.disclaimer


def test_gradcam_counterfactual_target_class() -> None:
    """Verify targeting an alternative class for error diagnosis."""
    predictor = get_predictor()
    explainer = GradCAMExplainer(predictor=predictor)

    arr = np.random.randint(60, 200, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")

    # Target specific class by name
    res_metal = explainer.explain(img, target_class="metal")
    assert res_metal.target_class == "metal"
    assert res_metal.target_idx == CANONICAL_CLASSES.index("metal")

    # Target specific class by index
    plastic_idx = CANONICAL_CLASSES.index("plastic")
    res_plastic = explainer.explain(img, target_class=plastic_idx)
    assert res_plastic.target_class == "plastic"
    assert res_plastic.target_idx == plastic_idx

    # Invalid class name raises ValueError
    with pytest.raises(ValueError, match="Unknown target class"):
        explainer.explain(img, target_class="unobtainium")

    # Out of bounds index raises ValueError
    with pytest.raises(ValueError, match="out of bounds"):
        explainer.explain(img, target_class=999)


def test_gradcam_correctness_ground_truth() -> None:
    """Verify is_correct calculation against ground truth labels."""
    predictor = get_predictor()
    explainer = GradCAMExplainer(predictor=predictor)

    arr = np.random.randint(60, 200, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")

    # Pass ground truth matching predicted class
    res1 = explainer.explain(img)
    res_true = explainer.explain(img, ground_truth=res1.predicted_class)
    assert res_true.is_correct is True

    # Pass ground truth different from predicted class
    other_cls = [c for c in CANONICAL_CLASSES if c != res1.predicted_class][0]
    res_false = explainer.explain(img, ground_truth=other_cls)
    assert res_false.is_correct is False


def test_generate_gradcam_report_samples(tmp_path: Path) -> None:
    """Verify generation and saving of report figures."""
    predictor = get_predictor()
    saved = generate_gradcam_report_samples(output_dir=tmp_path, predictor=predictor)

    # Should create README.md
    readme = tmp_path / "README.md"
    assert readme.exists()
    content = readme.read_text(encoding="utf-8")
    assert "Grad-CAM Visual Explainability Report" in content
    assert "Disclaimer" in content

    # Should have saved at least 1 figure if data/raw samples exist
    for fig_path in saved:
        assert fig_path.exists()
        assert fig_path.suffix == ".png"
        assert fig_path.stat().st_size > 0
