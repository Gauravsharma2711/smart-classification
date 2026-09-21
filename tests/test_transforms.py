"""Tests for preprocessing and camera-realistic augmentation (FR-3)."""

import albumentations as A
import numpy as np
import pytest
import torch
from PIL import Image

from waste_classifier.data.transforms import (
    assert_eval_transform_is_deterministic,
    build_eval_transforms,
    build_inference_transforms,
    build_train_transforms,
    is_transform_deterministic,
    preprocess_image,
)


@pytest.fixture
def sample_raw_image() -> np.ndarray:
    """Return a fixed 512x384 synthetic RGB image matching TrashNet dimensions."""
    np.random.seed(42)
    return np.random.randint(0, 255, (384, 512, 3), dtype=np.uint8)


def test_train_transform_output_shape_and_type(sample_raw_image: np.ndarray):
    """Verify train transform outputs 3x224x224 float tensor."""
    train_tf = build_train_transforms()
    out = train_tf(image=sample_raw_image)["image"]

    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32


def test_eval_transform_output_shape_and_type(sample_raw_image: np.ndarray):
    """Verify eval transform outputs 3x224x224 float tensor."""
    eval_tf = build_eval_transforms()
    out = eval_tf(image=sample_raw_image)["image"]

    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 224, 224)
    assert out.dtype == torch.float32


def test_critical_invariant_eval_is_strictly_deterministic(sample_raw_image: np.ndarray):
    """Acceptance criteria: Validation and test pipelines contain zero random transforms."""
    eval_tf = build_eval_transforms()
    # Must be 100% deterministic across multiple trials
    assert is_transform_deterministic(eval_tf, test_image=sample_raw_image, num_trials=15)
    # Assertion helper should pass cleanly
    assert_eval_transform_is_deterministic(eval_tf)


def test_train_transform_contains_stochasticity(sample_raw_image: np.ndarray):
    """Verify training pipeline exhibits stochastic behavior under augmentation."""
    train_tf = build_train_transforms()
    # Training transform should produce different outputs across repeated applications
    is_det = is_transform_deterministic(train_tf, test_image=sample_raw_image, num_trials=10)
    assert not is_det, "Training pipeline should not be deterministic"


def test_eval_fails_if_stochasticity_is_injected(sample_raw_image: np.ndarray):
    """Verify assert_eval_transform_is_deterministic catches accidentally injected randomness."""
    leaky_eval_tf = A.Compose(
        [
            A.Resize(height=224, width=224),
            A.HorizontalFlip(p=0.5),  # Stochastic!
            A.Normalize(),
        ]
    )

    assert not is_transform_deterministic(leaky_eval_tf, test_image=sample_raw_image)
    with pytest.raises(AssertionError, match="Critical invariant violation"):
        assert_eval_transform_is_deterministic(leaky_eval_tf)


def test_inference_eval_parity(sample_raw_image: np.ndarray):
    """Verify parity: inference preprocessing equals evaluation preprocessing (Rule 8)."""
    eval_tf = build_eval_transforms()
    inf_tf = build_inference_transforms()

    eval_out = eval_tf(image=sample_raw_image)["image"]
    inf_out = inf_tf(image=sample_raw_image)["image"]

    assert torch.equal(eval_out, inf_out), "Inference and evaluation transforms must be identical!"


def test_normalization_values():
    """Verify standard ImageNet normalization arithmetic on black and white inputs."""
    eval_tf = build_eval_transforms()

    black = np.zeros((384, 512, 3), dtype=np.uint8)
    white = np.full((384, 512, 3), 255, dtype=np.uint8)

    black_tensor = eval_tf(image=black)["image"]
    white_tensor = eval_tf(image=white)["image"]

    # (0.0 - mean) / std for ImageNet mean ~0.45, std ~0.22 -> ~ -2.1
    assert -2.3 < black_tensor.min().item() < -1.7
    # (1.0 - mean) / std for ImageNet -> ~ +2.2 to +2.6
    assert 2.0 < white_tensor.max().item() < 2.7


def test_config_overrides(sample_raw_image: np.ndarray):
    """Verify custom configuration overrides (e.g. changing input size)."""
    custom_cfg = {
        "train": {
            "random_resized_crop": {"size": [128, 128], "scale": [0.8, 1.0], "p": 1.0},
        },
        "eval": {
            "resize": [128, 128],
            "normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        },
    }

    train_tf = build_train_transforms(config=custom_cfg)
    eval_tf = build_eval_transforms(config=custom_cfg)

    t_out = train_tf(image=sample_raw_image)["image"]
    e_out = eval_tf(image=sample_raw_image)["image"]

    assert t_out.shape == (3, 128, 128)
    assert e_out.shape == (3, 128, 128)


def test_preprocess_image_pil_and_numpy(sample_raw_image: np.ndarray):
    """Verify preprocess_image helper handles both PIL images and NumPy arrays."""
    eval_tf = build_eval_transforms()

    # From numpy array
    tensor_from_np = preprocess_image(sample_raw_image, eval_tf)
    assert isinstance(tensor_from_np, torch.Tensor)

    # From PIL Image
    pil_img = Image.fromarray(sample_raw_image)
    tensor_from_pil = preprocess_image(pil_img, eval_tf)
    assert isinstance(tensor_from_pil, torch.Tensor)

    assert torch.equal(tensor_from_np, tensor_from_pil)
