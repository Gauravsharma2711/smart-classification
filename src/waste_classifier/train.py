"""Training orchestration routine for Phase 1 transfer learning.

Implements FR-5:
- Frozen backbone baseline training with AdamW and label smoothing.
- Best checkpoint saving based on validation macro-F1 score.
- Metrics tracking and append-only experiment logging in reports/runs.csv.
"""

from __future__ import annotations

import csv
import datetime
import logging
import subprocess
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from waste_classifier.data.datamodule import WasteDataModule
from waste_classifier.models.factory import create_model
from waste_classifier.models.module import WasteLightningModule

logger = logging.getLogger(__name__)


def get_git_commit_hash() -> str:
    """Retrieve the current Git commit hash if available."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def log_run_to_csv(
    run_record: dict[str, Any],
    csv_path: Path | str = "reports/runs.csv",
) -> None:
    """Append a training run record to reports/runs.csv."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "git_commit",
        "config_name",
        "phase",
        "backbone",
        "seed",
        "epochs",
        "val_loss",
        "val_acc",
        "val_f1",
        "checkpoint_path",
    ]
    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(run_record)
    logger.info(f"Recorded run summary in {path}")


def train_phase1(
    config_path: Path | str = "configs/base.yaml",
    epochs: int | None = None,
    lr: float | None = None,
    backbone: str | None = None,
    batch_size: int | None = None,
    smoke_test: bool = False,
    checkpoint_dir: Path | str = "checkpoints/phase1",
    seed: int | None = None,
) -> tuple[WasteLightningModule, dict[str, Any]]:
    """Execute Phase 1 training: freeze backbone and train classifier head.

    Args:
        config_path: Path to configuration YAML file.
        epochs: Optional epoch count override.
        lr: Optional learning rate override.
        backbone: Optional backbone architecture override.
        batch_size: Optional batch size override.
        smoke_test: If True, runs 1 epoch on minimal batches for fast verification.
        checkpoint_dir: Directory to save checkpoints.
        seed: Optional random seed override.

    Returns:
        Tuple of (trained WasteLightningModule, results dictionary).
    """
    cfg_p = Path(config_path)
    if not cfg_p.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_p}")

    with open(cfg_p, encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    # Resolve settings with CLI / argument overrides
    run_seed = seed if seed is not None else cfg.get("seed", 42)
    pl.seed_everything(run_seed, workers=True)

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    loss_cfg = cfg.get("loss", {})
    phase1_cfg = cfg.get("phase1", {})

    active_backbone = backbone or model_cfg.get("backbone", "efficientnet_b0")
    active_batch_size = batch_size or data_cfg.get("batch_size", 32)
    active_epochs = 1 if smoke_test else (epochs or phase1_cfg.get("epochs", 10))
    active_lr = lr or float(phase1_cfg.get("lr", 1e-3))
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.1))
    use_class_weights = loss_cfg.get("use_class_weights", True)
    pretrained = model_cfg.get("pretrained", True)
    dropout = float(model_cfg.get("dropout", 0.3))

    num_workers = 0 if smoke_test else data_cfg.get("num_workers", 2)

    logger.info(
        f"Starting Phase 1 training: backbone={active_backbone}, epochs={active_epochs}, "
        f"lr={active_lr}, batch_size={active_batch_size}, smoke_test={smoke_test}"
    )

    # Setup DataModule
    datamodule = WasteDataModule(
        manifest_path=data_cfg.get("manifest", "data/splits.csv"),
        data_root=data_cfg.get("root", "data/raw"),
        batch_size=active_batch_size,
        num_workers=num_workers,
        image_size=data_cfg.get("image_size", 224),
    )
    datamodule.prepare_data()
    datamodule.setup("fit")

    num_classes = len(datamodule.classes)
    class_weights = datamodule.class_weights if use_class_weights else None

    # Instantiate model with custom head
    classifier = create_model(
        backbone_name=active_backbone,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
    )

    # Instantiate LightningModule
    module = WasteLightningModule(
        model=classifier,
        num_classes=num_classes,
        class_weights=class_weights,
        label_smoothing=label_smoothing,
        lr=active_lr,
        phase=1,
    )

    # Callbacks & Logging
    ckpt_path = Path(checkpoint_dir)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_path),
        filename=f"phase1-{active_backbone}-{{epoch:02d}}-{{val_f1:.3f}}",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    tb_logger = TensorBoardLogger(
        save_dir="reports/tensorboard",
        name=f"phase1_{active_backbone}",
    )

    trainer_kwargs: dict[str, Any] = {
        "max_epochs": active_epochs,
        "callbacks": [checkpoint_callback, lr_monitor],
        "logger": tb_logger,
        "log_every_n_steps": 5 if smoke_test else 10,
        "enable_progress_bar": True,
    }

    if smoke_test:
        trainer_kwargs["limit_train_batches"] = 2
        trainer_kwargs["limit_val_batches"] = 2

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(module, datamodule=datamodule)

    # Extract final validation metrics
    val_metrics = trainer.callback_metrics
    val_loss = float(val_metrics.get("val_loss", 0.0))
    val_acc = float(val_metrics.get("val_acc", 0.0))
    val_f1 = float(val_metrics.get("val_f1", 0.0))
    best_model_path = checkpoint_callback.best_model_path

    logger.info(
        f"Phase 1 Complete. Best model saved to: {best_model_path}. "
        f"Final Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}"
    )

    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "git_commit": get_git_commit_hash(),
        "config_name": cfg_p.name,
        "phase": 1,
        "backbone": active_backbone,
        "seed": run_seed,
        "epochs": active_epochs,
        "val_loss": round(val_loss, 4),
        "val_acc": round(val_acc, 4),
        "val_f1": round(val_f1, 4),
        "checkpoint_path": best_model_path or str(ckpt_path / "last.ckpt"),
    }

    if not smoke_test:
        log_run_to_csv(results)

    return module, results
