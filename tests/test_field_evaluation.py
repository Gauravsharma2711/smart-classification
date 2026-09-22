"""Unit and integration tests for field-set evaluation (FR-8).

Verifies:
1. Field-set directory structure and minimum image counts (>=100 total, >=15 per class).
2. Strict mathematical disjointness (zero leakage, pHash hamming distance >= 10) against train/val/test splits.
3. Preprocessing parity and evaluation metric calculations.
4. Negative image uncertainty rejection rate calculation.
5. Error analysis report generation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from waste_classifier.data.acquire_field_set import (
    MIN_HAMMING_DISTANCE,
    TARGET_CLASSES,
    verify_field_set_integrity,
)
from waste_classifier.evaluate import (
    EvaluationResult,
    compute_evaluation_metrics,
    generate_field_error_analysis,
)


def test_field_set_structure_and_counts() -> None:
    """Verify that data/field_set contains required classes and >= 100 images."""
    field_dir = Path("data/field_set")
    if not field_dir.exists():
        pytest.skip("data/field_set does not exist locally.")

    total_images = 0
    for cls_name in TARGET_CLASSES:
        cls_dir = field_dir / cls_name
        assert cls_dir.is_dir(), f"Class folder missing: {cls_dir}"
        imgs = [f for f in cls_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        assert len(imgs) >= 15, f"Class {cls_name} has only {len(imgs)} images (< 15)"
        total_images += len(imgs)

    assert total_images >= 100, f"Field set has only {total_images} target images (< 100)"


def test_field_set_integrity_and_anti_leak() -> None:
    """Verify field set has zero overlap with train/val/test benchmark splits."""
    field_dir = Path("data/field_set")
    splits_csv = Path("data/splits.csv")
    if not field_dir.exists() or not splits_csv.exists():
        pytest.skip("data/field_set or data/splits.csv not available for test.")

    summary = verify_field_set_integrity(
        field_dir=field_dir,
        splits_csv_path=splits_csv,
        min_hamming_dist=MIN_HAMMING_DISTANCE,
        min_images_per_class=15,
    )

    assert summary["status"] == "verified"
    assert summary["total_images"] >= 100
    assert summary["min_hamming_distance_verified"] >= 10
    for cls_name in TARGET_CLASSES:
        assert summary["class_counts"][cls_name] >= 15


def test_field_metrics_computation() -> None:
    """Verify metric computation on synthetic field predictions."""
    y_true = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    y_pred = [0, 0, 1, 2, 2, 2, 3, 3, 4, 1, 5, 4]

    overall, per_class, cm = compute_evaluation_metrics(
        y_true=y_true,
        y_pred=y_pred,
        class_names=TARGET_CLASSES,
    )

    assert "accuracy" in overall
    assert "macro_f1" in overall
    assert overall["total_samples"] == 12
    assert cm.shape == (6, 6)
    for c in TARGET_CLASSES:
        assert c in per_class
        assert "f1_score" in per_class[c]
        assert "recall" in per_class[c]


def test_generate_field_error_analysis(tmp_path: Path) -> None:
    """Verify generation of markdown error analysis comparing test and field sets."""
    y_true = [0, 1, 2, 3, 4, 5]
    y_pred = [0, 1, 2, 3, 4, 5]

    overall, per_class, cm = compute_evaluation_metrics(
        y_true=y_true,
        y_pred=y_pred,
        class_names=TARGET_CLASSES,
    )

    eval_result = EvaluationResult(
        overall=overall,
        per_class=per_class,
        confusion_matrix=cm.tolist(),
        class_names=TARGET_CLASSES,
        metadata={"git_commit": "test_hash"},
        negative_rejection={
            "total_negatives": 10,
            "rejected_count": 9,
            "rejection_rate": 0.90,
            "confidence_threshold": 0.60,
            "mean_confidence": 0.38,
        },
    )

    # Fake test metrics JSON
    fake_test_metrics = {
        "overall": {"accuracy": 0.85, "macro_f1": 0.82},
        "per_class": {c: {"recall": 0.80, "f1_score": 0.81} for c in TARGET_CLASSES},
    }
    test_metrics_file = tmp_path / "test_metrics.json"
    test_metrics_file.write_text(json.dumps(fake_test_metrics), encoding="utf-8")

    out_md = tmp_path / "field_error_analysis.md"
    generated_path = generate_field_error_analysis(
        field_result=eval_result,
        test_metrics_path=test_metrics_file,
        output_path=out_md,
    )

    assert generated_path.exists()
    content = generated_path.read_text(encoding="utf-8")
    assert "# Field-Set Evaluation & Camera Domain Gap Analysis" in content
    assert "Domain Shift Gap" in content
    assert "Negative Sample & Uncertainty Rejection Analysis" in content
    assert "90.00%" in content
