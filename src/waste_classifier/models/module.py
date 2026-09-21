"""PyTorch Lightning Module for waste classification.

Implements FR-5 (Phase 1 Transfer Learning):
- Pretrained backbone frozen with only classifier head trainable.
- AdamW optimizer with configurable learning rate.
- Cross-entropy loss with configurable label smoothing and balanced class weights.
- Per-epoch loss, accuracy, and macro-F1 tracking using TorchMetrics.
"""

from __future__ import annotations

import logging
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from waste_classifier.models.factory import WasteClassifier

logger = logging.getLogger(__name__)


class WasteLightningModule(pl.LightningModule):
    """PyTorch Lightning Module encapsulating waste classification training & evaluation."""

    def __init__(
        self,
        model: WasteClassifier,
        num_classes: int,
        class_weights: torch.Tensor | None = None,
        label_smoothing: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        phase: int = 1,
    ) -> None:
        """Initialize WasteLightningModule.

        Args:
            model: WasteClassifier architecture instance.
            num_classes: Number of target categories.
            class_weights: Optional 1D tensor of per-class loss weights.
            label_smoothing: Label smoothing factor for CrossEntropyLoss.
            lr: Learning rate for optimizer.
            weight_decay: Weight decay factor for AdamW.
            phase: Fine-tuning phase (1 for frozen backbone, 2 for fine-tuning).
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model", "class_weights"])

        self.model = model
        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.phase = phase
        self.label_smoothing = label_smoothing

        # Register class weights as buffer so it moves across devices
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )

        # TorchMetrics for multiclass evaluation
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.train_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")

        # Configure phase 1 frozen backbone invariant
        if self.phase == 1:
            self.model.freeze_backbone()
            self._verify_phase1_invariants()

    def _verify_phase1_invariants(self) -> None:
        """Verify that backbone parameters are frozen and head parameters are trainable."""
        backbone_trainable = [p for p in self.model.backbone.parameters() if p.requires_grad]
        head_trainable = [p for p in self.model.head.parameters() if p.requires_grad]

        if backbone_trainable:
            raise RuntimeError(
                f"Phase 1 invariant violated: {len(backbone_trainable)} backbone "
                "parameters have requires_grad=True!"
            )
        if not head_trainable:
            raise RuntimeError(
                "Phase 1 invariant violated: classifier head has no trainable parameters!"
            )
        logger.info("Phase 1 invariants verified: backbone is fully frozen, head is trainable.")

    def on_fit_start(self) -> None:
        """Ensure invariants hold at fit start."""
        if self.phase == 1:
            self._verify_phase1_invariants()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through underlying classifier."""
        return self.model(x)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Single training step."""
        images, targets = batch
        logits = self(images)
        loss = self.criterion(logits, targets)

        preds = torch.argmax(logits, dim=1)
        self.train_acc.update(preds, targets)
        self.train_f1.update(preds, targets)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        """Compute and log epoch-level training metrics."""
        self.log("train_acc", self.train_acc.compute(), prog_bar=True)
        self.log("train_f1", self.train_f1.compute(), prog_bar=True)
        self.train_acc.reset()
        self.train_f1.reset()

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Single validation step."""
        images, targets = batch
        logits = self(images)
        loss = self.criterion(logits, targets)

        preds = torch.argmax(logits, dim=1)
        self.val_acc.update(preds, targets)
        self.val_f1.update(preds, targets)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self) -> None:
        """Compute and log epoch-level validation metrics."""
        self.log("val_acc", self.val_acc.compute(), prog_bar=True)
        self.log("val_f1", self.val_f1.compute(), prog_bar=True)
        self.val_acc.reset()
        self.val_f1.reset()

    def configure_optimizers(self) -> Any:
        """Configure AdamW optimizer for trainable parameters only."""
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        logger.info(
            f"Configuring AdamW optimizer with {len(trainable_params)} trainable "
            f"parameter tensors, lr={self.lr}, weight_decay={self.weight_decay}"
        )
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer
