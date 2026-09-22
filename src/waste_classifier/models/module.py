"""PyTorch Lightning Module for waste classification.

Implements:
- FR-5 (Phase 1 Transfer Learning): Pretrained backbone frozen, head trainable only.
- FR-6 (Phase 2 Fine-Tuning): Partial backbone unfreezing (last 20-30%), differential
  learning rates (lr_backbone << lr_head), cosine annealing scheduler, early stopping,
  and BatchNorm kept in eval mode.
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
        lr_backbone: float = 1e-5,
        lr_head: float = 1e-4,
        max_epochs: int = 15,
        weight_decay: float = 1e-2,
        phase: int = 1,
        freeze_bn: bool = True,
    ) -> None:
        """Initialize WasteLightningModule.

        Args:
            model: WasteClassifier architecture instance.
            num_classes: Number of target categories.
            class_weights: Optional 1D tensor of per-class loss weights.
            label_smoothing: Label smoothing factor for CrossEntropyLoss.
            lr: Learning rate for Phase 1 optimizer.
            lr_backbone: Fine-tuning learning rate for backbone parameters in Phase 2.
            lr_head: Fine-tuning learning rate for classifier head in Phase 2.
            max_epochs: Total epochs for CosineAnnealingLR horizon.
            weight_decay: Weight decay factor for AdamW.
            phase: Training phase (1 for frozen backbone, 2 for partial fine-tuning).
            freeze_bn: Keep BatchNorm layers in evaluation mode during Phase 2 training.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["model", "class_weights"])

        self.model = model
        self.num_classes = num_classes
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.lr_head = lr_head
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.phase = phase
        self.freeze_bn = freeze_bn
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

        # Configure phase invariants
        if self.phase == 1:
            self.model.freeze_backbone()
            self._verify_phase1_invariants()
        elif self.phase == 2:
            self._verify_phase2_invariants()

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

    def _verify_phase2_invariants(self) -> None:
        """Verify that backbone is partially unfrozen and head is trainable."""
        backbone_trainable = [p for p in self.model.backbone.parameters() if p.requires_grad]
        backbone_frozen = [p for p in self.model.backbone.parameters() if not p.requires_grad]
        head_trainable = [p for p in self.model.head.parameters() if p.requires_grad]

        if not backbone_trainable:
            raise RuntimeError(
                "Phase 2 invariant violated: backbone has no trainable parameters! "
                "Trailing layers must be unfrozen."
            )
        if not backbone_frozen:
            raise RuntimeError(
                "Phase 2 invariant violated: entire backbone is unfrozen! "
                "Earlier layers must remain frozen."
            )
        if not head_trainable:
            raise RuntimeError(
                "Phase 2 invariant violated: classifier head has no trainable parameters!"
            )
        logger.info(
            f"Phase 2 invariants verified: backbone has {len(backbone_trainable)} trainable "
            f"and {len(backbone_frozen)} frozen tensors; head is trainable."
        )

    def freeze_batchnorm(self) -> None:
        """Freeze all BatchNorm layers in eval mode to prevent statistics corruption."""
        count = 0
        for m in self.model.modules():
            if isinstance(
                m,
                (
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.BatchNorm3d,
                    nn.modules.batchnorm._BatchNorm,
                ),
            ):
                m.eval()
                count += 1
        if count > 0:
            logger.debug(f"Locked {count} BatchNorm layers in eval mode.")

    def on_train_epoch_start(self) -> None:
        """Enforce BatchNorm eval mode at the beginning of each training epoch."""
        if self.phase == 2 and self.freeze_bn:
            self.freeze_batchnorm()

    def on_fit_start(self) -> None:
        """Ensure phase invariants hold at fit start."""
        if self.phase == 1:
            self._verify_phase1_invariants()
        elif self.phase == 2:
            self._verify_phase2_invariants()
            if self.freeze_bn:
                self.freeze_batchnorm()

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
        """Configure optimizer and learning rate scheduler according to phase."""
        if self.phase == 1:
            trainable_params = [p for p in self.parameters() if p.requires_grad]
            logger.info(
                f"Phase 1: AdamW with {len(trainable_params)} trainable "
                f"parameter tensors, lr={self.lr}, weight_decay={self.weight_decay}"
            )
            return torch.optim.AdamW(
                trainable_params,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

        # Phase 2: Separate parameter groups for backbone and head with Cosine Annealing
        backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]
        head_params = [p for p in self.model.head.parameters() if p.requires_grad]

        logger.info(
            f"Phase 2: AdamW with {len(backbone_params)} backbone tensors (lr={self.lr_backbone}), "
            f"{len(head_params)} head tensors (lr={self.lr_head}), weight_decay={self.weight_decay}"
        )

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.lr_backbone},
                {"params": head_params, "lr": self.lr_head},
            ],
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,
            eta_min=1e-7,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
