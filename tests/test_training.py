"""Unit and integration tests for FR-5 Phase 1 training pipeline.

Validates:
- WasteDataset sample loading, error handling, and tensor output.
- WasteDataModule split filtering, class weighting, and DataLoader construction.
- WasteLightningModule Phase 1 invariants (frozen backbone, trainable head).
- Loss computation with label smoothing and class weights.
- Lightning Trainer smoke run producing metrics and checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from waste_classifier.data.datamodule import WasteDataModule
from waste_classifier.data.dataset import WasteDataset
from waste_classifier.models.factory import create_model
from waste_classifier.models.module import WasteLightningModule


def test_dataset_sample_retrieval(tmp_path: Path) -> None:
    """Verify WasteDataset correctly loads images and converts to tensors."""
    from PIL import Image

    # Create dummy images
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img_path = img_dir / "sample.jpg"
    Image.new("RGB", (64, 64), color=(255, 0, 0)).save(img_path)

    df = pd.DataFrame([{"rel_path": "sample.jpg", "class_name": "cardboard"}])
    class_to_idx = {"cardboard": 0, "glass": 1}

    dataset = WasteDataset(df=df, data_root=img_dir, class_to_idx=class_to_idx)
    assert len(dataset) == 1

    img_tensor, label = dataset[0]
    assert isinstance(img_tensor, torch.Tensor)
    assert label == 0
    assert img_tensor.shape == (3, 64, 64)


def test_dataset_missing_column_raises_value_error(tmp_path: Path) -> None:
    """Verify missing required columns raise ValueError."""
    df = pd.DataFrame([{"wrong_col": "sample.jpg"}])
    with pytest.raises(ValueError, match="missing required columns"):
        WasteDataset(df=df, data_root=tmp_path, class_to_idx={"c": 0})


def test_dataset_missing_image_raises_file_not_found(tmp_path: Path) -> None:
    """Verify non-existent image paths raise FileNotFoundError."""
    df = pd.DataFrame([{"rel_path": "nonexistent.jpg", "class_name": "glass"}])
    dataset = WasteDataset(df=df, data_root=tmp_path, class_to_idx={"glass": 0})
    with pytest.raises(FileNotFoundError):
        _ = dataset[0]


def test_datamodule_setup_and_weights() -> None:
    """Verify WasteDataModule properly sets up splits and calculates balanced class weights."""
    manifest_path = Path("data/splits.csv")
    raw_dir = Path("data/raw")
    if not manifest_path.exists() or not raw_dir.exists():
        pytest.skip("Dataset or manifest not available locally.")

    dm = WasteDataModule(
        manifest_path=manifest_path,
        data_root=raw_dir,
        batch_size=8,
        num_workers=0,
    )
    dm.setup("fit")

    assert len(dm.classes) == 6
    assert dm.class_weights is not None
    assert len(dm.class_weights) == 6
    assert all(w > 0 for w in dm.class_weights)

    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    images, targets = batch
    assert images.shape == (8, 3, 224, 224)
    assert targets.shape == (8,)


def test_lightning_module_phase1_invariants() -> None:
    """Verify Phase 1 freezes backbone and leaves classifier head trainable."""
    model = create_model(backbone_name="efficientnet_b0", num_classes=6, pretrained=False)
    module = WasteLightningModule(model=model, num_classes=6, phase=1)

    # Invariant: all backbone parameters frozen
    assert all(not p.requires_grad for p in module.model.backbone.parameters())
    # Invariant: head parameters trainable
    assert all(p.requires_grad for p in module.model.head.parameters())


def test_lightning_module_phase1_violation_raises_runtime_error() -> None:
    """Verify an error is raised if backbone parameters remain trainable in Phase 1."""
    model = create_model(backbone_name="efficientnet_b0", num_classes=6, pretrained=False)
    module = WasteLightningModule(model=model, num_classes=6, phase=1)

    # Intentionally unfreeze backbone to simulate Phase 1 invariant violation
    module.model.unfreeze_backbone(1.0)
    with pytest.raises(RuntimeError, match="Phase 1 invariant violated"):
        module._verify_phase1_invariants()


def test_lightning_module_step_and_loss() -> None:
    """Verify forward and training step computation."""
    model = create_model(backbone_name="efficientnet_b0", num_classes=6, pretrained=False)
    weights = torch.ones(6)
    module = WasteLightningModule(
        model=model,
        num_classes=6,
        class_weights=weights,
        label_smoothing=0.1,
        lr=1e-3,
        phase=1,
    )

    dummy_images = torch.randn(4, 3, 224, 224)
    dummy_targets = torch.tensor([0, 1, 2, 3])

    loss = module.training_step((dummy_images, dummy_targets), 0)
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    assert loss.item() > 0


def test_trainer_smoke_run(tmp_path: Path) -> None:
    """Verify end-to-end training smoke run using Lightning Trainer and ModelCheckpoint."""
    manifest_path = Path("data/splits.csv")
    raw_dir = Path("data/raw")
    if not manifest_path.exists() or not raw_dir.exists():
        pytest.skip("Dataset or manifest not available locally.")

    dm = WasteDataModule(
        manifest_path=manifest_path,
        data_root=raw_dir,
        batch_size=4,
        num_workers=0,
    )
    dm.setup("fit")

    model = create_model(backbone_name="mobilenetv3_large_100", num_classes=6, pretrained=False)
    module = WasteLightningModule(
        model=model,
        num_classes=6,
        class_weights=dm.class_weights,
        label_smoothing=0.1,
        lr=1e-3,
        phase=1,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(tmp_path),
        filename="smoke-{epoch:02d}",
        save_top_k=1,
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=2,
        enable_progress_bar=False,
        callbacks=[ckpt_cb],
        default_root_dir=str(tmp_path),
    )
    trainer.fit(module, datamodule=dm)

    # Verify last checkpoint was saved
    last_ckpt = tmp_path / "last.ckpt"
    assert last_ckpt.exists()
    assert last_ckpt.stat().st_size > 0


def test_lightning_module_phase2_invariants() -> None:
    """Verify Phase 2 requires partially unfrozen backbone and rejects fully frozen/unfrozen ones."""
    model = create_model(backbone_name="efficientnet_b0", num_classes=6, pretrained=False)
    model.freeze_backbone()
    model.unfreeze_backbone(0.25)

    # Valid Phase 2 setup
    module = WasteLightningModule(model=model, num_classes=6, phase=2)
    assert module.phase == 2

    # Invariant failure case 1: Fully frozen backbone in Phase 2
    model.freeze_backbone()
    with pytest.raises(RuntimeError, match="backbone has no trainable parameters"):
        WasteLightningModule(model=model, num_classes=6, phase=2)

    # Invariant failure case 2: Fully unfrozen backbone in Phase 2
    model.unfreeze_backbone(1.0)
    with pytest.raises(RuntimeError, match="entire backbone is unfrozen"):
        WasteLightningModule(model=model, num_classes=6, phase=2)


def test_phase2_optimizer_and_scheduler() -> None:
    """Verify differential learning rates and CosineAnnealingLR scheduler in Phase 2."""
    model = create_model(backbone_name="mobilenetv3_large_100", num_classes=6, pretrained=False)
    model.freeze_backbone()
    model.unfreeze_backbone(0.25)

    module = WasteLightningModule(
        model=model,
        num_classes=6,
        lr_backbone=1e-5,
        lr_head=1e-4,
        max_epochs=15,
        phase=2,
    )

    opt_config = module.configure_optimizers()
    assert isinstance(opt_config, dict)
    assert "optimizer" in opt_config
    assert "lr_scheduler" in opt_config

    optimizer = opt_config["optimizer"]
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == 1e-5
    assert optimizer.param_groups[1]["lr"] == 1e-4

    scheduler = opt_config["lr_scheduler"]["scheduler"]
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert scheduler.T_max == 15


def test_phase2_batchnorm_eval_mode() -> None:
    """Verify BatchNorm layers remain locked in eval mode during Phase 2 training."""
    model = create_model(backbone_name="resnet50", num_classes=6, pretrained=False)
    model.freeze_backbone()
    model.unfreeze_backbone(0.25)

    module = WasteLightningModule(model=model, num_classes=6, phase=2, freeze_bn=True)
    # Set entire module to train mode
    module.train()

    # Trigger epoch start hook which locks BatchNorm
    module.on_train_epoch_start()

    # Verify that all BatchNorm layers are in eval mode (training == False)
    bn_layers = [
        m
        for m in module.model.modules()
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d))
    ]
    assert len(bn_layers) > 0
    assert all(not m.training for m in bn_layers)


def test_phase2_smoke_training(tmp_path: Path) -> None:
    """Verify end-to-end Phase 2 smoke execution loading from Phase 1 checkpoint."""
    from waste_classifier.train import train_phase1, train_phase2

    manifest_path = Path("data/splits.csv")
    raw_dir = Path("data/raw")
    if not manifest_path.exists() or not raw_dir.exists():
        pytest.skip("Dataset or manifest not available locally.")

    p1_dir = tmp_path / "p1"
    p2_dir = tmp_path / "p2"

    # Quick 1-batch smoke Phase 1 to produce a valid checkpoint
    _, p1_results = train_phase1(
        epochs=1,
        smoke_test=True,
        checkpoint_dir=p1_dir,
    )
    p1_ckpt = p1_results["checkpoint_path"]

    # Execute Phase 2 smoke test starting from the Phase 1 checkpoint
    _, p2_results, comparison = train_phase2(
        phase1_checkpoint=p1_ckpt,
        epochs=1,
        smoke_test=True,
        checkpoint_dir=p2_dir,
    )

    assert p2_results["phase"] == 2
    assert "delta_val_f1" in comparison
    assert Path(p2_results["checkpoint_path"]).exists()
