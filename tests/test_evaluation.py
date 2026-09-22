"""Unit and integration tests for FR-7: Evaluation Pipeline.

Validates:
- Correctness of metric calculations (Accuracy, Macro-F1, Precision, Recall).
- Deterministic lexicographical class ordering.
- Confusion matrix dimensions (C x C) and sample conservation.
- Output schema completeness of EvaluationResult.
- Preprocessing parity between evaluation and inference transforms.
- Strict disjointness between train, validation, and test splits (zero leakage).
- Confusion matrix plotting functionality.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from waste_classifier.data.transforms import (
    build_eval_transforms,
    build_inference_transforms,
    is_transform_deterministic,
)
from waste_classifier.evaluate import (
    EvaluationResult,
    compute_evaluation_metrics,
    plot_confusion_matrix,
)


def test_metric_calculation_accuracy_and_f1() -> None:
    """Verify precision, recall, F1, and accuracy on synthetic ground-truth and predictions."""
    class_names = ["cardboard", "glass", "metal"]
    # 0: cardboard, 1: glass, 2: metal
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]

    overall, per_class, cm = compute_evaluation_metrics(
        y_true=y_true, y_pred=y_pred, class_names=class_names
    )

    # Expected: 4 out of 6 correct = 0.6667 accuracy
    assert overall["accuracy"] == 0.6667
    assert overall["total_samples"] == 6

    # Class 0: true=[0,0], pred=[0,2] -> TP=1, FP=1, FN=1 -> Prec=0.5, Rec=0.5, F1=0.5
    assert per_class["cardboard"]["precision"] == 0.5
    assert per_class["cardboard"]["recall"] == 0.5
    assert per_class["cardboard"]["f1_score"] == 0.5
    assert per_class["cardboard"]["support"] == 2

    # Class 1: true=[1,1], pred=[0,1,1] -> TP=2, FP=1, FN=0 -> Prec=2/3=0.6667, Rec=1.0, F1=0.8
    assert per_class["glass"]["recall"] == 1.0
    assert per_class["glass"]["precision"] == 0.6667
    assert per_class["glass"]["f1_score"] == 0.8
    assert per_class["glass"]["support"] == 2

    # Class 2: true=[2,2], pred=[2] -> TP=1, FP=0, FN=1 -> Prec=1.0, Rec=0.5, F1=0.6667
    assert per_class["metal"]["precision"] == 1.0
    assert per_class["metal"]["recall"] == 0.5
    assert per_class["metal"]["f1_score"] == 0.6667
    assert per_class["metal"]["support"] == 2

    # Macro-F1: (0.5 + 0.8 + 0.6667) / 3 = 0.6556
    assert overall["macro_f1"] == 0.6556


def test_deterministic_class_ordering() -> None:
    """Verify class ordering is strictly deterministic across evaluation outputs."""
    unordered_classes = ["trash", "cardboard", "plastic", "metal", "glass", "paper"]
    sorted_classes = sorted(unordered_classes)

    y_true = [0, 1, 2, 3, 4, 5]
    y_pred = [0, 1, 2, 3, 4, 5]

    _, per_class, cm = compute_evaluation_metrics(
        y_true=y_true, y_pred=y_pred, class_names=sorted_classes
    )

    assert list(per_class.keys()) == sorted_classes
    assert cm.shape == (len(sorted_classes), len(sorted_classes))


def test_confusion_matrix_dimensions_and_conservation() -> None:
    """Verify confusion matrix has dimensions (C, C) and total counts equal total samples."""
    class_names = ["c0", "c1", "c2", "c3", "c4", "c5"]
    np.random.seed(42)
    y_true = np.random.randint(0, 6, size=100)
    y_pred = np.random.randint(0, 6, size=100)

    overall, per_class, cm = compute_evaluation_metrics(
        y_true=y_true, y_pred=y_pred, class_names=class_names
    )

    assert cm.shape == (6, 6)
    assert int(cm.sum()) == 100
    assert overall["total_samples"] == 100
    assert sum(pc["support"] for pc in per_class.values()) == 100


def test_evaluation_output_schema() -> None:
    """Verify EvaluationResult produces all required schema fields."""
    result = EvaluationResult(
        overall={"accuracy": 0.85, "macro_f1": 0.83, "total_samples": 100},
        per_class={"cardboard": {"precision": 0.8, "recall": 0.8, "f1_score": 0.8, "support": 50}},
        confusion_matrix=[[40, 10], [5, 45]],
        class_names=["cardboard", "glass"],
        metadata={"checkpoint_path": "best.ckpt", "seed": 42},
    )

    d = result.to_dict()
    assert "overall" in d
    assert "per_class" in d
    assert "confusion_matrix" in d
    assert "class_names" in d
    assert "metadata" in d

    # Verify JSON serializability
    serialized = json.dumps(d)
    deserialized = json.loads(serialized)
    assert deserialized["overall"]["accuracy"] == 0.85


def test_evaluation_preprocessing_parity() -> None:
    """Verify evaluation transforms exactly equal inference transforms and are deterministic."""
    eval_transform = build_eval_transforms()
    inf_transform = build_inference_transforms()

    # Preprocessing parity guarantee
    assert is_transform_deterministic(eval_transform)
    assert is_transform_deterministic(inf_transform)

    # Identical output on synthetic image
    dummy_img = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
    out_eval = eval_transform(image=dummy_img)["image"]
    out_inf = inf_transform(image=dummy_img)["image"]
    assert np.allclose(out_eval.numpy(), out_inf.numpy(), atol=1e-6)


def test_accidental_split_overlap_guard() -> None:
    """Verify zero file-path leakage between train, val, and test splits."""
    manifest_p = Path("data/splits.csv")
    if not manifest_p.exists():
        pytest.skip("Manifest splits.csv not present.")

    df = pd.read_csv(manifest_p)
    train_paths = set(df[df["split"] == "train"]["rel_path"])
    val_paths = set(df[df["split"] == "val"]["rel_path"])
    test_paths = set(df[df["split"] == "test"]["rel_path"])

    assert len(train_paths & val_paths) == 0, "Leakage detected between train and val!"
    assert len(train_paths & test_paths) == 0, "Leakage detected between train and test!"
    assert len(val_paths & test_paths) == 0, "Leakage detected between val and test!"


def test_plot_confusion_matrix(tmp_path: Path) -> None:
    """Verify visual confusion matrix is generated and valid image is written to disk."""
    cm = np.array([[10, 2], [1, 15]])
    class_names = ["metal", "paper"]
    out_file = tmp_path / "test_cm.png"

    saved_path = plot_confusion_matrix(
        cm=cm,
        class_names=class_names,
        output_path=out_file,
        title="Test CM",
    )

    assert saved_path.exists()
    assert saved_path.stat().st_size > 0

    # Verify readable image
    with Image.open(saved_path) as img:
        assert img.size[0] > 0
        assert img.size[1] > 0
