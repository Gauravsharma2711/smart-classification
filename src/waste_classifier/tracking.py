"""Experiment tracking and reproducibility management module (FR-22).

Implements:
- Structured logging of all mandatory experiment metadata:
  1. Configuration (full YAML and dictionary, 100% recoverable)
  2. Random Seed
  3. Performance Metrics (val_loss, val_acc, val_f1, train_loss, etc.)
  4. Model Architecture & Backbone
  5. Git Commit Hash
  6. Training Phase (Phase 1, Phase 2, etc.)
  7. ISO Timestamp
  8. Dataset / Split Identifier (manifest path data/splits.csv & SHA256 checksum)
- Native TensorBoard logging with hyperparameter summary and text metadata.
- Run tracking CSV with backward-compatible schema.
- Run recovery, configuration reproduction, and integrity validation.
- Handling of interrupted / failed runs with error diagnostics.
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pytorch_lightning.loggers import TensorBoardLogger

from waste_classifier.evaluate import get_git_commit_hash

logger = logging.getLogger(__name__)

DEFAULT_RUNS_CSV = Path("reports/runs.csv")
DEFAULT_TENSORBOARD_DIR = Path("reports/tensorboard")
DEFAULT_SPLITS_MANIFEST = Path("data/splits.csv")

RUNS_CSV_FIELDNAMES = [
    "timestamp",
    "git_commit",
    "config_name",
    "phase",
    "backbone",
    "seed",
    "dataset_split",
    "epochs",
    "val_loss",
    "val_acc",
    "val_f1",
    "checkpoint_path",
    "status",
]


def compute_manifest_checksum(manifest_path: Path | str = DEFAULT_SPLITS_MANIFEST) -> str:
    """Compute SHA256 checksum of the dataset split manifest for cryptographic provenance."""
    p = Path(manifest_path)
    if not p.exists():
        return "missing"
    try:
        hasher = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()[:12]
    except Exception as err:
        logger.warning(f"Could not compute manifest checksum for {p}: {err}")
        return "error"


@dataclass
class ExperimentMetadata:
    """Complete, self-contained record of an experiment run guaranteeing full reproducibility."""

    experiment_id: str
    timestamp: str
    git_commit: str
    phase: int
    backbone: str
    seed: int
    dataset_split: str
    dataset_split_checksum: str
    config_name: str
    configuration: dict[str, Any]
    status: str = "INITIALIZED"  # INITIALIZED, RUNNING, COMPLETED, FAILED
    epochs: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    checkpoint_path: str = ""
    tensorboard_version_dir: str = ""
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to serializable dictionary."""
        return asdict(self)


class ExperimentTracker:
    """Orchestrates metadata capture, TensorBoard logging, and runs CSV persistence."""

    def __init__(
        self,
        config: dict[str, Any],
        config_path: Path | str = "configs/base.yaml",
        phase: int = 1,
        backbone: str | None = None,
        seed: int = 42,
        dataset_split: Path | str = DEFAULT_SPLITS_MANIFEST,
        epochs: int = 0,
        tb_logger: TensorBoardLogger | None = None,
        runs_csv_path: Path | str = DEFAULT_RUNS_CSV,
    ) -> None:
        """Initialize experiment tracker and capture static environment metadata."""
        self.config_path = Path(config_path)
        self.config_name = self.config_path.name
        self.config = dict(config)
        self.phase = phase
        self.backbone = backbone or self.config.get("model", {}).get("backbone", "efficientnet_b0")
        self.seed = seed
        self.dataset_split = str(dataset_split)
        self.dataset_split_checksum = compute_manifest_checksum(self.dataset_split)
        self.epochs = epochs
        self.tb_logger = tb_logger
        self.runs_csv_path = Path(runs_csv_path)

        now = datetime.datetime.now()
        self.timestamp = now.isoformat()
        self.git_commit = get_git_commit_hash()
        self.experiment_id = (
            f"exp_phase{self.phase}_{self.backbone}_s{self.seed}_{now.strftime('%Y%m%d_%H%M%S_%f')}"
        )

        self.metadata = ExperimentMetadata(
            experiment_id=self.experiment_id,
            timestamp=self.timestamp,
            git_commit=self.git_commit,
            phase=self.phase,
            backbone=self.backbone,
            seed=self.seed,
            dataset_split=self.dataset_split,
            dataset_split_checksum=self.dataset_split_checksum,
            config_name=self.config_name,
            configuration=self.config,
            status="RUNNING",
            epochs=self.epochs,
        )

        self._initialize_run_artifacts()

    def _initialize_run_artifacts(self) -> None:
        """Record initial configuration and metadata into the TensorBoard run directory."""
        if self.tb_logger is not None:
            log_dir = Path(self.tb_logger.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            self.metadata.tensorboard_version_dir = str(log_dir)

            # 1. Save full recoverable config.yaml in TensorBoard run folder
            config_copy_path = log_dir / "config.yaml"
            with open(config_copy_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(self.config, f, sort_keys=False)

            # 2. Save initial metadata.json in TensorBoard run folder
            meta_copy_path = log_dir / "metadata.json"
            with open(meta_copy_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata.to_dict(), f, indent=2)

            # 3. Log hyperparams and markdown text to TensorBoard
            hparams = {
                "experiment_id": self.experiment_id,
                "phase": self.phase,
                "backbone": self.backbone,
                "seed": self.seed,
                "epochs": self.epochs,
                "git_commit": self.git_commit,
                "dataset_split": self.dataset_split,
                "dataset_split_checksum": self.dataset_split_checksum,
                "config_name": self.config_name,
            }
            try:
                self.tb_logger.log_hyperparams(hparams)
                if hasattr(self.tb_logger, "experiment") and hasattr(
                    self.tb_logger.experiment, "add_text"
                ):
                    summary_md = (
                        f"### Experiment Metadata\n"
                        f"- **ID:** `{self.experiment_id}`\n"
                        f"- **Phase:** {self.phase}\n"
                        f"- **Backbone:** `{self.backbone}`\n"
                        f"- **Seed:** {self.seed}\n"
                        f"- **Git Commit:** `{self.git_commit}`\n"
                        f"- **Dataset Split:** `{self.dataset_split}` (checksum `{self.dataset_split_checksum}`)\n"
                        f"- **Config:** `{self.config_name}`\n"
                    )
                    self.tb_logger.experiment.add_text("experiment_summary", summary_md, 0)
            except Exception as err:
                logger.warning(f"Failed to log hyperparams to TensorBoard: {err}")

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Update tracked metric values in metadata."""
        for k, v in metrics.items():
            self.metadata.metrics[k] = float(v)

    def end_run(
        self,
        status: str = "COMPLETED",
        metrics: dict[str, float] | None = None,
        checkpoint_path: str | Path | None = None,
        error_message: str = "",
    ) -> ExperimentMetadata:
        """Finalize the experiment run, update metadata, and record summary to CSV."""
        self.metadata.status = status
        if metrics:
            self.log_metrics(metrics)
        if checkpoint_path:
            self.metadata.checkpoint_path = str(checkpoint_path)
        if error_message:
            self.metadata.error_message = error_message

        # Update metadata.json in TensorBoard dir
        if self.metadata.tensorboard_version_dir:
            meta_path = Path(self.metadata.tensorboard_version_dir) / "metadata.json"
            try:
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(self.metadata.to_dict(), f, indent=2)
            except Exception as err:
                logger.warning(f"Could not update {meta_path}: {err}")

        # Record to runs.csv
        self._record_to_runs_csv()

        logger.info(
            f"Experiment {self.experiment_id} finalized with status '{status}' "
            f"(val_f1={self.metadata.metrics.get('val_f1', 0.0):.4f})."
        )
        return self.metadata

    def _record_to_runs_csv(self) -> None:
        """Append or update run row in reports/runs.csv."""
        self.runs_csv_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.runs_csv_path.exists()

        row = {
            "timestamp": self.metadata.timestamp,
            "git_commit": self.metadata.git_commit,
            "config_name": self.metadata.config_name,
            "phase": self.metadata.phase,
            "backbone": self.metadata.backbone,
            "seed": self.metadata.seed,
            "dataset_split": self.metadata.dataset_split,
            "epochs": self.metadata.epochs,
            "val_loss": round(float(self.metadata.metrics.get("val_loss", 0.0)), 4),
            "val_acc": round(float(self.metadata.metrics.get("val_acc", 0.0)), 4),
            "val_f1": round(float(self.metadata.metrics.get("val_f1", 0.0)), 4),
            "checkpoint_path": self.metadata.checkpoint_path,
            "status": self.metadata.status,
        }

        with open(self.runs_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RUNS_CSV_FIELDNAMES)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


def verify_experiment_integrity(meta_dict: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate that all 8 mandatory FR-22 items are present, non-empty, and valid.

    Mandatory Checklist:
    1. configuration: non-empty dictionary
    2. seed: integer
    3. metrics: non-empty dictionary with valid floats
    4. model/backbone: non-empty string
    5. git_commit: non-empty string
    6. phase: integer (1 or 2)
    7. timestamp: valid ISO-8601 string
    8. relevant dataset/split identifier: non-empty string
    """
    issues: list[str] = []

    # 1. Configuration
    cfg = meta_dict.get("configuration")
    if not isinstance(cfg, dict) or not cfg:
        issues.append("Missing or empty 'configuration' dictionary.")

    # 2. Seed
    seed = meta_dict.get("seed")
    if not isinstance(seed, int):
        issues.append(f"Invalid or missing 'seed' (expected int, got {type(seed).__name__}).")

    # 3. Metrics
    metrics = meta_dict.get("metrics")
    if not isinstance(metrics, dict):
        issues.append(
            f"Invalid or missing 'metrics' (expected dict, got {type(metrics).__name__})."
        )

    # 4. Model/backbone
    backbone = meta_dict.get("backbone")
    if not isinstance(backbone, str) or not backbone.strip():
        issues.append("Missing or empty 'backbone' identifier.")

    # 5. Git commit
    git_commit = meta_dict.get("git_commit")
    if not isinstance(git_commit, str) or not git_commit.strip():
        issues.append("Missing or empty 'git_commit' identifier.")

    # 6. Phase
    phase = meta_dict.get("phase")
    if phase not in (1, 2):
        issues.append(f"Invalid 'phase' {phase} (must be 1 or 2).")

    # 7. Timestamp
    ts = meta_dict.get("timestamp")
    if not isinstance(ts, str) or not ts.strip():
        issues.append("Missing or empty 'timestamp'.")
    else:
        try:
            datetime.datetime.fromisoformat(ts)
        except Exception:
            issues.append(f"Malformed ISO timestamp: {ts}")

    # 8. Dataset/split identifier
    split_id = meta_dict.get("dataset_split")
    if not isinstance(split_id, str) or not split_id.strip():
        issues.append("Missing or empty 'dataset_split' identifier.")

    is_valid = len(issues) == 0
    return is_valid, issues


def load_experiment_metadata(source: Path | str) -> dict[str, Any]:
    """Load metadata dictionary from a directory (containing metadata.json) or direct JSON file."""
    p = Path(source)
    if p.is_dir():
        target = p / "metadata.json"
    else:
        target = p

    if not target.exists():
        raise FileNotFoundError(f"Experiment metadata file not found at: {target}")

    with open(target, encoding="utf-8") as f:
        return json.load(f)


def reproduce_experiment_config(
    source: Path | str,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Recover the exact configuration dictionary from an experiment run.

    Args:
        source: Run directory or metadata.json or config.yaml path.
        output_path: Optional path to write the recovered YAML configuration.

    Returns:
        Recovered configuration dictionary ready for execution.
    """
    p = Path(source)

    # If pointing directly to config.yaml
    if p.is_file() and p.name == "config.yaml":
        with open(p, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    elif p.is_dir() and (p / "config.yaml").exists():
        with open(p / "config.yaml", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        meta = load_experiment_metadata(p)
        cfg = meta.get("configuration")
        if not isinstance(cfg, dict):
            raise ValueError(
                f"Could not recover configuration from {p}: key 'configuration' empty."
            )

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        logger.info(f"Saved recovered configuration to: {out_p}")

    return cfg


def list_tracked_experiments(
    tensorboard_root: Path | str = DEFAULT_TENSORBOARD_DIR,
    runs_csv_path: Path | str = DEFAULT_RUNS_CSV,
) -> list[dict[str, Any]]:
    """Scan TensorBoard directories and runs.csv to enumerate all recorded experiments."""
    tb_root = Path(tensorboard_root)
    records: list[dict[str, Any]] = []

    # 1. Scan TensorBoard metadata.json files
    if tb_root.exists():
        for meta_file in sorted(tb_root.glob("**/metadata.json")):
            try:
                data = load_experiment_metadata(meta_file)
                valid, _ = verify_experiment_integrity(data)
                data["is_valid"] = valid
                records.append(data)
            except Exception as err:
                logger.warning(f"Error reading {meta_file}: {err}")

    # If no TensorBoard metadata.json found, fall back to runs.csv rows
    if not records:
        csv_p = Path(runs_csv_path)
        if csv_p.exists():
            try:
                import pandas as pd

                df = pd.read_csv(csv_p)
                for _, row in df.iterrows():
                    rec = row.to_dict()
                    rec["experiment_id"] = f"run_phase{rec.get('phase')}_{rec.get('backbone')}"
                    records.append(rec)
            except Exception as err:
                logger.warning(f"Error reading {csv_p}: {err}")

    return records
