"""Foundation and configuration tests."""

from pathlib import Path

from omegaconf import OmegaConf

import waste_classifier


def test_package_version():
    """Verify package can be imported and has valid version."""
    assert hasattr(waste_classifier, "__version__")
    assert waste_classifier.__version__ == "0.1.0"


def test_configs_exist_and_parse():
    """Verify all required YAML configs exist and are well-formed."""
    config_dir = Path("configs")
    assert config_dir.exists()

    required_configs = [
        config_dir / "base.yaml",
        config_dir / "live.yaml",
        config_dir / "bin_mapping.yaml",
        config_dir / "augmentation.yaml",
        config_dir / "backbones" / "efficientnet_b0.yaml",
        config_dir / "backbones" / "mobilenetv3.yaml",
        config_dir / "backbones" / "resnet50.yaml",
    ]

    for cfg_path in required_configs:
        assert cfg_path.exists(), f"Missing config: {cfg_path}"
        cfg = OmegaConf.load(cfg_path)
        assert cfg is not None


def test_base_config_split_ratios():
    """Verify base config split adds up to 1.0."""
    base_cfg = OmegaConf.load("configs/base.yaml")
    split = base_cfg.data.split
    total = split.train + split.val + split.test
    assert round(total, 4) == 1.0
    assert split.train == 0.70
    assert split.val == 0.15
    assert split.test == 0.15


def test_bin_mapping_accessibility():
    """Verify bin mapping contains text labels alongside colors for accessibility."""
    bin_cfg = OmegaConf.load("configs/bin_mapping.yaml")
    categories = bin_cfg.categories

    expected_classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    for cls_name in expected_classes:
        assert cls_name in categories, f"Missing category {cls_name} in bin_mapping.yaml"
        cat = categories[cls_name]
        assert "bin_name" in cat and len(cat.bin_name) > 0
        assert "text_label" in cat and len(cat.text_label) > 0, (
            f"Accessibility violation: {cls_name} lacks text_label"
        )
        assert "color_hex" in cat and cat.color_hex.startswith("#")
        assert "instructions" in cat and len(cat.instructions) > 0
