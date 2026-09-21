"""PyTorch Lightning DataModule for waste classification.

Manages data loading, class-to-index mapping, balanced class weights computation,
and DataLoader creation across train, validation, and test splits.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import albumentations as A
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from waste_classifier.data.dataset import WasteDataset
from waste_classifier.data.transforms import build_eval_transforms, build_train_transforms

logger = logging.getLogger(__name__)


class WasteDataModule(pl.LightningDataModule):
    """LightningDataModule for the waste classification dataset."""

    def __init__(
        self,
        manifest_path: str | Path = "data/splits.csv",
        data_root: str | Path = "data/raw",
        batch_size: int = 32,
        num_workers: int = 2,
        image_size: int = 224,
        pin_memory: bool = False,
        train_transform: A.Compose | None = None,
        eval_transform: A.Compose | None = None,
    ) -> None:
        """Initialize WasteDataModule.

        Args:
            manifest_path: Path to the splits manifest CSV file.
            data_root: Root directory containing raw images.
            batch_size: Batch size for training and evaluation.
            num_workers: Number of DataLoader worker subprocesses.
            image_size: Square dimension to resize images to.
            pin_memory: Whether to pin memory in DataLoader.
            train_transform: Optional custom training transform pipeline.
            eval_transform: Optional custom evaluation transform pipeline.
        """
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.pin_memory = pin_memory

        self.train_transform = train_transform or build_train_transforms(image_size=image_size)
        self.eval_transform = eval_transform or build_eval_transforms(image_size=image_size)

        self.classes: list[str] = []
        self.class_to_idx: dict[str, int] = {}
        self.idx_to_class: dict[int, str] = {}
        self.class_weights: torch.Tensor | None = None

        self.train_dataset: WasteDataset | None = None
        self.val_dataset: WasteDataset | None = None
        self.test_dataset: WasteDataset | None = None

    def prepare_data(self) -> None:
        """Verify presence of manifest file and dataset directory."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest CSV not found at: {self.manifest_path}")
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root directory not found at: {self.data_root}")

    def setup(self, stage: str | None = None) -> None:
        """Load manifest, set up splits, mappings, class weights, and datasets."""
        df = pd.read_csv(self.manifest_path)

        # Derive class labels dynamically without hardcoding
        self.classes = sorted(df["class_name"].unique().tolist())
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
        self.idx_to_class = {idx: name for idx, name in enumerate(self.classes)}

        train_df = df[df["split"] == "train"].reset_index(drop=True)
        val_df = df[df["split"] == "val"].reset_index(drop=True)
        test_df = df[df["split"] == "test"].reset_index(drop=True)

        # Compute balanced class weights on the training split only
        if not train_df.empty:
            counts = train_df["class_name"].value_counts()
            total_samples = len(train_df)
            num_classes = len(self.classes)
            weights = [
                total_samples / (num_classes * counts.get(cls_name, 1)) for cls_name in self.classes
            ]
            self.class_weights = torch.tensor(weights, dtype=torch.float32)
            logger.info(f"Computed training class weights: {self.class_weights.tolist()}")

        if stage in (None, "fit"):
            self.train_dataset = WasteDataset(
                df=train_df,
                data_root=self.data_root,
                class_to_idx=self.class_to_idx,
                transform=self.train_transform,
            )
            self.val_dataset = WasteDataset(
                df=val_df,
                data_root=self.data_root,
                class_to_idx=self.class_to_idx,
                transform=self.eval_transform,
            )

        if stage in (None, "test"):
            self.test_dataset = WasteDataset(
                df=test_df,
                data_root=self.data_root,
                class_to_idx=self.class_to_idx,
                transform=self.eval_transform,
            )

    def train_dataloader(self) -> DataLoader:
        """Create DataLoader for training set."""
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is not initialized. Run setup() first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        """Create DataLoader for validation set."""
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is not initialized. Run setup() first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        """Create DataLoader for held-out test set."""
        if self.test_dataset is None:
            raise RuntimeError("test_dataset is not initialized. Run setup() first.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> WasteDataModule:
        """Create WasteDataModule from configuration dictionary."""
        data_cfg = cfg.get("data", {})
        return cls(
            manifest_path=data_cfg.get("manifest", "data/splits.csv"),
            data_root=data_cfg.get("root", "data/raw"),
            batch_size=data_cfg.get("batch_size", 32),
            num_workers=data_cfg.get("num_workers", 2),
            image_size=data_cfg.get("image_size", 224),
        )
