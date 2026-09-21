"""PyTorch Dataset implementation for waste classification images.

Loads samples based on the split manifest DataFrame and applies
camera-realistic train augmentations or deterministic evaluation transforms.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import albumentations as A
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class WasteDataset(Dataset):
    """PyTorch Dataset reading waste classification images from manifest records."""

    def __init__(
        self,
        df: pd.DataFrame,
        data_root: Path | str,
        class_to_idx: dict[str, int],
        transform: A.Compose | None = None,
    ) -> None:
        """Initialize WasteDataset.

        Args:
            df: DataFrame containing at least 'rel_path' and 'class_name' columns.
            data_root: Root directory where relative image paths are located.
            class_to_idx: Mapping from class name string to integer label index.
            transform: Albumentations Compose transform pipeline.
        """
        required_cols = {"rel_path", "class_name"}
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"Manifest DataFrame missing required columns: {missing}")

        self.df = df.reset_index(drop=True)
        self.data_root = Path(data_root)
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        """Load and transform sample at index.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (transformed image tensor, integer class label).
        """
        row = self.df.iloc[idx]
        rel_path = row["rel_path"]
        class_name = row["class_name"]

        if class_name not in self.class_to_idx:
            raise KeyError(
                f"Class '{class_name}' at index {idx} not found in class_to_idx mapping: "
                f"{list(self.class_to_idx.keys())}"
            )

        target = self.class_to_idx[class_name]
        image_path = self.data_root / rel_path

        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            with Image.open(image_path) as pil_img:
                rgb_img = pil_img.convert("RGB")
                img_np = np.array(rgb_img)
        except Exception as err:
            raise RuntimeError(f"Failed to read image at {image_path}: {err}") from err

        if self.transform is not None:
            res: dict[str, Any] = self.transform(image=img_np)
            tensor_img = res["image"]
        else:
            # Fallback to standard conversion if no transform provided
            tensor_img = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        return tensor_img, target
