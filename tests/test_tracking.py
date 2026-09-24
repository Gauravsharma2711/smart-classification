"""Unit tests for Experiment Tracking and Provenance Recording (FR-22)."""

from pathlib import Path

import pytest
import yaml

from waste_classifier.tracking import (
    ExperimentTracker,
    reproduce_experiment_config,
    verify_experiment_integrity,
)
from waste_classifier.train import train_phase1


@pytest.fixture
def dummy_config(tmp_path: Path) -> tuple[dict, Path]:
    """Create a sample configuration dictionary and YAML file for testing."""
    cfg = {
        "seed": 42,
        "data": {
            "root": "data/raw",
            "manifest": "data/splits.csv",
            "image_size": 224,
            "batch_size": 16,
        },
        "model": {
            "backbone": "efficientnet_b0",
            "dropout": 0.3,
        },
        "phase1": {
            "epochs": 1,
            "lr": 0.001,
        },
    }
    cfg_file = tmp_path / "test_base.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return cfg, cfg_file


def test_experiment_tracker_records_all_eight_mandatory_items(tmp_path: Path, dummy_config):
    """Verify that all 8 mandatory FR-22 items are recorded."""
    cfg, cfg_file = dummy_config
    tb_dir = tmp_path / "tensorboard" / "phase1_test"
    tb_dir.mkdir(parents=True, exist_ok=True)
    runs_csv = tmp_path / "runs.csv"

    tracker = ExperimentTracker(
        config=cfg,
        config_path=cfg_file,
        phase=1,
        backbone="efficientnet_b0",
        seed=42,
        dataset_split="data/splits.csv",
        epochs=3,
        runs_csv_path=runs_csv,
    )
    # Simulate run dir
    tracker.metadata.tensorboard_version_dir = str(tb_dir)

    tracker.log_metrics({"val_loss": 0.42, "val_acc": 0.85, "val_f1": 0.84})
    meta = tracker.end_run(status="COMPLETED", checkpoint_path=str(tmp_path / "model.ckpt"))

    meta_dict = meta.to_dict()

    # 1. Configuration recoverable
    assert "configuration" in meta_dict
    assert meta_dict["configuration"]["model"]["backbone"] == "efficientnet_b0"

    # 2. Seed
    assert meta_dict["seed"] == 42

    # 3. Metrics
    assert meta_dict["metrics"]["val_f1"] == 0.84
    assert meta_dict["metrics"]["val_acc"] == 0.85

    # 4. Model/backbone
    assert meta_dict["backbone"] == "efficientnet_b0"

    # 5. Git commit
    assert meta_dict["git_commit"] != ""

    # 6. Phase
    assert meta_dict["phase"] == 1

    # 7. Timestamp
    assert meta_dict["timestamp"] != ""

    # 8. Dataset/split identifier
    assert meta_dict["dataset_split"] == "data/splits.csv"
    assert meta_dict["dataset_split_checksum"] != ""

    # Verify integrity helper
    valid, issues = verify_experiment_integrity(meta_dict)
    assert valid is True
    assert len(issues) == 0


def test_configuration_recovery_and_reproducibility(tmp_path: Path, dummy_config):
    """Verify exact configuration recovery from experiment artifacts."""
    cfg, cfg_file = dummy_config
    run_dir = tmp_path / "run_0"
    run_dir.mkdir(parents=True)

    runs_csv = tmp_path / "runs.csv"
    tracker = ExperimentTracker(
        config=cfg,
        config_path=cfg_file,
        phase=1,
        backbone="efficientnet_b0",
        seed=123,
        runs_csv_path=runs_csv,
    )
    tracker.metadata.tensorboard_version_dir = str(run_dir)

    # Save config copy as done in tracker
    config_copy = run_dir / "config.yaml"
    with open(config_copy, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    meta_copy = run_dir / "metadata.json"
    tracker.end_run(status="COMPLETED")
    with open(meta_copy, "w", encoding="utf-8") as f:
        yaml.safe_dump(tracker.metadata.to_dict(), f)

    # Recover configuration to a new file
    recovered_file = tmp_path / "recovered.yaml"
    recovered_cfg = reproduce_experiment_config(run_dir, output_path=recovered_file)

    assert recovered_file.exists()
    assert recovered_cfg["seed"] == cfg["seed"]
    assert recovered_cfg["data"]["batch_size"] == cfg["data"]["batch_size"]
    assert recovered_cfg["model"]["backbone"] == cfg["model"]["backbone"]


def test_restarting_run_creates_distinct_records(tmp_path: Path, dummy_config):
    """Verify restarting a run creates separate, comparable version records."""
    cfg, cfg_file = dummy_config
    runs_csv = tmp_path / "runs.csv"

    tracker1 = ExperimentTracker(
        config=cfg,
        config_path=cfg_file,
        phase=1,
        backbone="efficientnet_b0",
        seed=42,
        runs_csv_path=runs_csv,
    )
    tracker1.end_run(status="COMPLETED", metrics={"val_f1": 0.70})

    # Restart second run
    tracker2 = ExperimentTracker(
        config=cfg,
        config_path=cfg_file,
        phase=1,
        backbone="efficientnet_b0",
        seed=42,
        runs_csv_path=runs_csv,
    )
    tracker2.end_run(status="COMPLETED", metrics={"val_f1": 0.72})

    assert tracker1.experiment_id != tracker2.experiment_id

    # Check runs.csv contains both distinct entries
    with open(runs_csv, encoding="utf-8") as f:
        lines = f.readlines()
    # 1 header + 2 run rows
    assert len(lines) == 3


def test_failed_run_tracking(tmp_path: Path, dummy_config):
    """Verify that incomplete or failed runs record status and error diagnostics."""
    cfg, cfg_file = dummy_config
    runs_csv = tmp_path / "runs.csv"
    run_dir = tmp_path / "failed_run"
    run_dir.mkdir()

    tracker = ExperimentTracker(
        config=cfg,
        config_path=cfg_file,
        phase=2,
        backbone="mobilenetv3_large_100",
        seed=456,
        runs_csv_path=runs_csv,
    )
    tracker.metadata.tensorboard_version_dir = str(run_dir)

    tracker.end_run(
        status="FAILED",
        error_message="RuntimeError: CUDA out of memory / simulated fault.",
    )

    assert tracker.metadata.status == "FAILED"
    assert "simulated fault" in tracker.metadata.error_message


def test_verify_experiment_integrity_detects_missing_fields():
    """Verify that integrity validation flags missing provenance items."""
    incomplete_meta = {
        "seed": 42,
        "phase": 1,
        "backbone": "resnet50",
        # Missing configuration, metrics, git_commit, timestamp, dataset_split
    }

    valid, issues = verify_experiment_integrity(incomplete_meta)
    assert valid is False
    assert len(issues) >= 4


def test_train_phase1_smoke_with_tracking(tmp_path: Path, dummy_config):
    """Verify integration of ExperimentTracker with live Phase 1 training."""
    _, cfg_file = dummy_config
    test_runs_csv = tmp_path / "smoke_runs.csv"
    test_ckpt_dir = tmp_path / "checkpoints"

    module, results = train_phase1(
        config_path=cfg_file,
        epochs=1,
        smoke_test=True,
        runs_csv_path=test_runs_csv,
        checkpoint_dir=test_ckpt_dir,
    )

    assert results["status"] == "COMPLETED"
    assert "dataset_split" in results
    assert "val_f1" in results
    assert "git_commit" in results
    assert test_runs_csv.exists()
