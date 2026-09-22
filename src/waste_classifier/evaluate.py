"""Evaluation module for waste classification models.

Implements FR-7:
- Held-out test set evaluation: accuracy, macro-F1, per-class precision, recall, F1.
- Confusion matrix computation and visualization (PNG).
- Formatted classification report export.
- Machine-readable metrics JSON with metadata (checkpoint, backbone, config, seed, commit).
- Parity preservation: evaluation preprocessing strictly equals inference preprocessing.
"""

from __future__ import annotations

import datetime
import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from waste_classifier.data.datamodule import WasteDataModule
from waste_classifier.models.factory import create_model

# Use headless backend for matplotlib
matplotlib.use("Agg")

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


@dataclass
class EvaluationResult:
    """Structured container for full evaluation results."""

    overall: dict[str, float | int]
    per_class: dict[str, dict[str, float | int]]
    confusion_matrix: list[list[int]]
    class_names: list[str]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert evaluation result to a serializable dictionary."""
        return asdict(self)


def find_best_evaluation_checkpoint(
    checkpoint_path: Path | str | None = None,
    runs_csv_path: Path | str = "reports/runs.csv",
    checkpoints_root: Path | str = "checkpoints",
) -> Path:
    """Locate the best trained checkpoint (preferring Phase 2, falling back to Phase 1).

    Args:
        checkpoint_path: Optional explicit checkpoint path.
        runs_csv_path: Path to CSV tracking training runs.
        checkpoints_root: Root directory where phase checkpoints reside.

    Returns:
        Path to resolved checkpoint file.
    """
    if checkpoint_path is not None:
        p = Path(checkpoint_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Explicit checkpoint not found at: {p}")

    # 1. Inspect runs.csv for highest val_f1 Phase 2 run, then Phase 1
    csv_p = Path(runs_csv_path)
    if csv_p.exists():
        try:
            df = pd.read_csv(csv_p)
            for phase in [2, 1]:
                phase_runs = df[df["phase"] == phase]
                if not phase_runs.empty:
                    best_run = phase_runs.sort_values(by="val_f1", ascending=False).iloc[0]
                    candidate = Path(best_run["checkpoint_path"])
                    if candidate.exists():
                        logger.info(
                            f"Selected best Phase {phase} checkpoint from runs.csv: {candidate} "
                            f"(val_f1={best_run['val_f1']})"
                        )
                        return candidate
        except Exception as err:
            logger.warning(f"Could not read runs.csv: {err}")

    # 2. Check filesystem under checkpoints/phase2 and checkpoints/phase1
    root_p = Path(checkpoints_root)
    for phase_dir in ["phase2", "phase1"]:
        dir_p = root_p / phase_dir
        if dir_p.exists():
            ckpts = sorted(dir_p.glob("phase*.ckpt"), reverse=True)
            if ckpts:
                return ckpts[0]
            last = dir_p / "last.ckpt"
            if last.exists():
                return last

    raise FileNotFoundError("Could not find any model checkpoint to evaluate. Run training first.")


def compute_evaluation_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    class_names: list[str],
) -> tuple[dict[str, float | int], dict[str, dict[str, float | int]], np.ndarray]:
    """Compute overall and per-class classification metrics.

    Args:
        y_true: Array of ground-truth integer labels.
        y_pred: Array of predicted integer labels.
        class_names: Deterministically ordered list of class name strings.

    Returns:
        Tuple of (overall metrics dict, per-class metrics dict, confusion matrix ndarray).
    """
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)

    acc = float(accuracy_score(y_true_np, y_pred_np))
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_np, y_pred_np, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true_np, y_pred_np, average="weighted", zero_division=0
    )

    overall = {
        "accuracy": round(acc, 4),
        "macro_precision": round(float(precision_macro), 4),
        "macro_recall": round(float(recall_macro), 4),
        "macro_f1": round(float(f1_macro), 4),
        "weighted_f1": round(float(f1_weighted), 4),
        "total_samples": int(len(y_true_np)),
    }

    # Per-class metrics
    prec_per, rec_per, f1_per, sup_per = precision_recall_fscore_support(
        y_true_np, y_pred_np, labels=list(range(len(class_names))), zero_division=0
    )

    per_class: dict[str, dict[str, float | int]] = {}
    for idx, name in enumerate(class_names):
        per_class[name] = {
            "precision": round(float(prec_per[idx]), 4),
            "recall": round(float(rec_per[idx]), 4),
            "f1_score": round(float(f1_per[idx]), 4),
            "support": int(sup_per[idx]),
        }

    cm = confusion_matrix(y_true_np, y_pred_np, labels=list(range(len(class_names))))

    return overall, per_class, cm


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    output_path: Path | str = "reports/confusion_matrix.png",
    title: str = "Test Set Confusion Matrix",
) -> Path:
    """Generate and save an annotated confusion matrix visualization."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Normalized Ratio"},
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
    )

    # Annotate with raw counts inside parentheses
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            norm_val = cm_norm[i, j]
            color = "white" if norm_val > 0.55 else "black"
            ax.text(
                j + 0.5,
                i + 0.72,
                f"({val})",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
                alpha=0.85,
            )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()

    plt.savefig(out_file, dpi=150)
    plt.close(fig)
    logger.info(f"Saved visual confusion matrix to {out_file}")
    return out_file


def evaluate_checkpoint(
    checkpoint_path: Path | str | None = None,
    config_path: Path | str = "configs/base.yaml",
    split: str = "test",
    output_dir: Path | str = "reports",
    device: str = "cpu",
) -> EvaluationResult:
    """Execute evaluation of a trained model checkpoint on the specified dataset split.

    Args:
        checkpoint_path: Path to checkpoint file (auto-detected if None).
        config_path: Path to configuration YAML file.
        split: Dataset split to evaluate on ('test' or 'val').
        output_dir: Directory where evaluation artifacts will be written.
        device: Device to run evaluation on ('cpu' or 'cuda').

    Returns:
        Populated EvaluationResult object.
    """
    cfg_p = Path(config_path)
    if not cfg_p.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_p}")

    with open(cfg_p, encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

    ckpt_p = find_best_evaluation_checkpoint(checkpoint_path)
    logger.info(f"Evaluating checkpoint: {ckpt_p} on split '{split}'")

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    backbone_name = model_cfg.get("backbone", "efficientnet_b0")
    image_size = data_cfg.get("image_size", 224)
    batch_size = data_cfg.get("batch_size", 32)
    seed = cfg.get("seed", 42)

    # 1. Setup DataModule and DataLoader
    datamodule = WasteDataModule(
        manifest_path=data_cfg.get("manifest", "data/splits.csv"),
        data_root=data_cfg.get("root", "data/raw"),
        batch_size=batch_size,
        num_workers=0,  # Single-process for deterministic evaluation
        image_size=image_size,
    )
    datamodule.prepare_data()
    datamodule.setup(split)

    if split == "test":
        dataloader = datamodule.test_dataloader()
    elif split == "val":
        dataloader = datamodule.val_dataloader()
    else:
        raise ValueError(f"Unsupported evaluation split: '{split}'. Must be 'test' or 'val'.")

    class_names = datamodule.classes
    num_classes = len(class_names)

    # 2. Instantiate Model and Load Checkpoint Weights
    classifier = create_model(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=False,
        dropout=float(model_cfg.get("dropout", 0.3)),
    )

    ckpt_data = torch.load(ckpt_p, map_location="cpu")
    model_state = {}
    for k, v in ckpt_data["state_dict"].items():
        if k.startswith("model."):
            model_state[k[len("model.") :]] = v
        elif (
            not k.startswith("criterion.")
            and not k.startswith("train_")
            and not k.startswith("val_")
            and k != "class_weights"
        ):
            model_state[k] = v

    classifier.load_state_dict(model_state)
    classifier.to(device)
    classifier.eval()

    # 3. Perform Inference
    all_targets: list[int] = []
    all_preds: list[int] = []
    all_probs: list[list[float]] = []

    logger.info(f"Running evaluation inference across {len(dataloader.dataset)} samples...")  # type: ignore
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            logits = classifier(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_targets.extend(targets.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    # 4. Compute Metrics
    overall, per_class, cm = compute_evaluation_metrics(
        y_true=all_targets,
        y_pred=all_preds,
        class_names=class_names,
    )

    metadata = {
        "checkpoint_path": str(ckpt_p),
        "config_name": cfg_p.name,
        "backbone": backbone_name,
        "seed": seed,
        "split": split,
        "timestamp": datetime.datetime.now().isoformat(),
        "git_commit": get_git_commit_hash(),
    }

    result = EvaluationResult(
        overall=overall,
        per_class=per_class,
        confusion_matrix=cm.tolist(),
        class_names=class_names,
        metadata=metadata,
    )

    # 5. Export Reports and Visualizations
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # A. JSON Metrics
    metrics_json_file = out_dir / f"{split}_metrics.json"
    with open(metrics_json_file, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info(f"Saved machine-readable metrics to {metrics_json_file}")

    # B. Visual Confusion Matrix Plot
    plot_file = out_dir / "confusion_matrix.png"
    plot_confusion_matrix(
        cm=cm,
        class_names=class_names,
        output_path=plot_file,
        title=f"{split.capitalize()} Set Confusion Matrix ({backbone_name})",
    )

    # C. Text Classification Report
    report_text = classification_report(
        all_targets,
        all_preds,
        target_names=class_names,
        digits=4,
    )
    report_file = out_dir / "classification_report.txt"
    report_file.write_text(report_text, encoding="utf-8")
    logger.info(f"Saved classification report to {report_file}")

    return result
