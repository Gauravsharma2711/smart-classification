"""Preprocessing and camera-realistic augmentation pipelines using Albumentations.

Implements FR-3:
- 224x224 input sizing.
- Backbone-appropriate normalization.
- Camera-realistic augmentation on TRAIN ONLY (PRD 8.3).
- Strict deterministic transforms on validation, test, and inference.
- Configuration-driven parameters without hardcoding.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig, OmegaConf
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/augmentation.yaml")

# Standard ImageNet normalization parameters (default across timm backbones)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_augmentation_config(
    config_path: Path | str | None = None,
) -> dict[str, Any]:
    """Load augmentation configuration from YAML file or return defaults."""
    cfg_p = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if cfg_p.exists():
        raw_cfg = OmegaConf.load(cfg_p)
        return OmegaConf.to_container(raw_cfg, resolve=True)  # type: ignore

    logger.warning(f"Augmentation config not found at {cfg_p}, using built-in defaults.")
    return {
        "train": {
            "random_resized_crop": {"size": [224, 224], "scale": [0.6, 1.0], "p": 1.0},
            "rotation": {"limit": 25, "p": 0.5},
            "perspective": {"scale": [0.05, 0.1], "p": 0.3},
            "color_jitter": {
                "brightness": 0.2,
                "contrast": 0.2,
                "saturation": 0.2,
                "hue": 0.1,
                "p": 0.5,
            },
            "blur": {"motion_blur_limit": 7, "gaussian_blur_limit": 7, "p": 0.3},
            "noise": {"std_range": [0.02, 0.1], "p": 0.3},
            "jpeg_compression": {"quality_range": [60, 100], "p": 0.3},
            "coarse_dropout": {
                "num_holes_range": [1, 8],
                "hole_height_range": [8, 32],
                "hole_width_range": [8, 32],
                "p": 0.3,
            },
        },
        "eval": {
            "resize": [224, 224],
            "normalize": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        },
    }


def build_train_transforms(
    config: dict[str, Any] | DictConfig | None = None,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    image_size: int | None = None,
) -> A.Compose:
    """Build camera-realistic augmentation pipeline for the TRAIN split only.

    Includes:
    - Random resized crop
    - Rotation
    - Perspective transforms
    - Color/brightness/contrast jitter
    - Motion & Gaussian blur
    - Gaussian noise
    - JPEG compression artifacts
    - Coarse dropout / random erasing
    - Normalization & Tensor conversion
    """
    if config is None:
        cfg = load_augmentation_config()
    elif isinstance(config, DictConfig):
        cfg = OmegaConf.to_container(config, resolve=True)  # type: ignore
    else:
        cfg = config

    t_cfg = cfg.get("train", {})
    crop_cfg = t_cfg.get("random_resized_crop", {})
    rot_cfg = t_cfg.get("rotation", {})
    persp_cfg = t_cfg.get("perspective", {})
    color_cfg = t_cfg.get("color_jitter", {})
    blur_cfg = t_cfg.get("blur", {})
    noise_cfg = t_cfg.get("noise", {})
    jpeg_cfg = t_cfg.get("jpeg_compression", {})
    drop_cfg = t_cfg.get("coarse_dropout", {})

    target_size = (
        (image_size, image_size)
        if image_size is not None
        else tuple(crop_cfg.get("size", [224, 224]))
    )
    scale_range = tuple(crop_cfg.get("scale", [0.6, 1.0]))

    transforms: list[A.BasicTransform] = [
        # 1. Random Crop & Scale
        A.RandomResizedCrop(
            size=(target_size[0], target_size[1]),
            scale=scale_range,
            p=float(crop_cfg.get("p", 1.0)),
        ),
        # 2. Rotation
        A.Rotate(limit=int(rot_cfg.get("limit", 25)), p=float(rot_cfg.get("p", 0.5))),
        # 3. Perspective
        A.Perspective(
            scale=tuple(persp_cfg.get("scale", [0.05, 0.1])),
            p=float(persp_cfg.get("p", 0.3)),
        ),
        # 4. Color / Lighting / Contrast Jitter
        A.ColorJitter(
            brightness=float(color_cfg.get("brightness", 0.2)),
            contrast=float(color_cfg.get("contrast", 0.2)),
            saturation=float(color_cfg.get("saturation", 0.2)),
            hue=float(color_cfg.get("hue", 0.1)),
            p=float(color_cfg.get("p", 0.5)),
        ),
        # 5. Motion and Gaussian Blur
        A.OneOf(
            [
                A.MotionBlur(blur_limit=int(blur_cfg.get("motion_blur_limit", 7))),
                A.GaussianBlur(blur_limit=int(blur_cfg.get("gaussian_blur_limit", 7))),
            ],
            p=float(blur_cfg.get("p", 0.3)),
        ),
        # 6. Sensor Noise
        A.GaussNoise(
            std_range=tuple(noise_cfg.get("std_range", [0.02, 0.1])),
            p=float(noise_cfg.get("p", 0.3)),
        ),
        # 7. JPEG Compression Artifacts
        A.ImageCompression(
            quality_range=tuple(jpeg_cfg.get("quality_range", [60, 100])),
            p=float(jpeg_cfg.get("p", 0.3)),
        ),
        # 8. Partial Occlusion / Coarse Dropout
        A.CoarseDropout(
            num_holes_range=tuple(drop_cfg.get("num_holes_range", [1, 8])),
            hole_height_range=tuple(drop_cfg.get("hole_height_range", [8, 32])),
            hole_width_range=tuple(drop_cfg.get("hole_width_range", [8, 32])),
            p=float(drop_cfg.get("p", 0.3)),
        ),
        # 9. Normalization & PyTorch Tensor Conversion
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]

    return A.Compose(transforms)


def build_eval_transforms(
    config: dict[str, Any] | DictConfig | None = None,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    image_size: int = 224,
) -> A.Compose:
    """Build strictly DETERMINISTIC preprocessing pipeline for validation and testing.

    Acceptance criterion (FR-3):
    Contains ZERO random transformations. Unit-tested for strict determinism.
    """
    if config is not None:
        if isinstance(config, DictConfig):
            cfg = OmegaConf.to_container(config, resolve=True)  # type: ignore
        else:
            cfg = config
        eval_cfg = cfg.get("eval", {})
        resize_size = eval_cfg.get("resize", [image_size, image_size])
        norm_cfg = eval_cfg.get("normalize", {})
        norm_mean = tuple(norm_cfg.get("mean", list(mean)))
        norm_std = tuple(norm_cfg.get("std", list(std)))
    else:
        resize_size = [image_size, image_size]
        norm_mean = mean
        norm_std = std

    transforms: list[A.BasicTransform] = [
        A.Resize(height=resize_size[0], width=resize_size[1]),
        A.Normalize(mean=norm_mean, std=norm_std),
        ToTensorV2(),
    ]

    eval_pipeline = A.Compose(transforms)
    assert_eval_transform_is_deterministic(eval_pipeline)
    return eval_pipeline


def build_inference_transforms(
    config: dict[str, Any] | DictConfig | None = None,
    mean: tuple[float, float, float] = IMAGENET_MEAN,
    std: tuple[float, float, float] = IMAGENET_STD,
    image_size: int = 224,
) -> A.Compose:
    """Build inference transformation pipeline.

    Parity guarantee (AGENTS.md Rule 8):
    Inference preprocessing exactly matches evaluation preprocessing.
    """
    return build_eval_transforms(config=config, mean=mean, std=std, image_size=image_size)


def preprocess_image(image: Image.Image | np.ndarray, transform: A.Compose) -> torch.Tensor:
    """Apply an Albumentations transform to a PIL image or NumPy array."""
    if isinstance(image, Image.Image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        img_np = np.array(image, dtype=np.uint8)
    else:
        img_np = image

    transformed = transform(image=img_np)
    return transformed["image"]


def is_transform_deterministic(
    transform: A.Compose,
    test_image: np.ndarray | None = None,
    num_trials: int = 10,
) -> bool:
    """Verify whether a transform pipeline produces 100% identical outputs on repeat runs."""
    if test_image is None:
        np.random.seed(0)
        test_image = np.random.randint(0, 255, (384, 512, 3), dtype=np.uint8)

    base_out = transform(image=test_image)["image"]
    base_tensor = base_out if isinstance(base_out, torch.Tensor) else torch.tensor(base_out)

    for _ in range(num_trials):
        trial_out = transform(image=test_image)["image"]
        trial_tensor = trial_out if isinstance(trial_out, torch.Tensor) else torch.tensor(trial_out)
        if not torch.allclose(base_tensor, trial_tensor, atol=1e-6):
            return False
    return True


def assert_eval_transform_is_deterministic(transform: A.Compose) -> None:
    """Enforce the critical invariant that eval/test transforms contain zero stochastic transforms."""
    if not is_transform_deterministic(transform):
        raise AssertionError(
            "Critical invariant violation: Validation or test transformation pipeline contains random/stochastic transforms!"
        )
