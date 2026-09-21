"""Tests for dataset ingestion, discovery, and EDA (FR-1, PRD 7.1, 7.2)."""

from pathlib import Path

import pytest
from PIL import Image

from waste_classifier.data.download import EXPECTED_CLASSES, validate_and_clean_dataset
from waste_classifier.data.ingestion import (
    discover_classes,
    inspect_dataset,
    run_eda,
)


def test_validate_and_clean_dataset_structure(tmp_path: Path):
    """Test validation logic on a synthetic folder-per-class dataset."""
    dataset_dir = tmp_path / "test_raw"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Create expected class folders with sample dummy images
    for idx, class_name in enumerate(EXPECTED_CLASSES):
        class_folder = dataset_dir / class_name
        class_folder.mkdir(parents=True, exist_ok=True)
        # Create 2 valid images per class
        for img_idx in range(2):
            img = Image.new("RGB", (32, 32), color=(idx * 20, img_idx * 50, 100))
            img.save(class_folder / f"sample_{img_idx}.jpg")

    # Inject 1 corrupt file
    corrupt_file = dataset_dir / "plastic" / "corrupt.jpg"
    with open(corrupt_file, "wb") as f:
        f.write(b"NOT_AN_IMAGE_CONTENT")

    # Inject 1 non-image file
    dummy_txt = dataset_dir / "trash" / "notes.txt"
    dummy_txt.write_text("dummy text")

    # Run validation with cleanup enabled
    stats = validate_and_clean_dataset(dataset_dir, remove_corrupt=True)

    assert stats["dataset_name"] == "TrashNet"
    assert stats["total_valid_images"] == len(EXPECTED_CLASSES) * 2
    assert stats["corrupted_images_count"] == 1
    assert stats["unsupported_files_count"] == 1
    assert not corrupt_file.exists(), "Corrupt file should have been removed"
    assert not dummy_txt.exists(), "Unsupported text file should have been removed"

    for class_name in EXPECTED_CLASSES:
        assert stats["class_counts"][class_name] == 2


def test_discover_classes(tmp_path: Path):
    """Verify dynamic discovery of class directories."""
    dataset_dir = tmp_path / "classes_test"
    dataset_dir.mkdir()
    for name in ["paper", "glass", "plastic"]:
        (dataset_dir / name).mkdir()
    # Hidden folder should be ignored
    (dataset_dir / ".hidden").mkdir()

    discovered = discover_classes(dataset_dir)
    assert discovered == ["glass", "paper", "plastic"]


def test_discover_classes_errors(tmp_path: Path):
    """Verify error handling on non-existent or empty directories."""
    with pytest.raises(FileNotFoundError):
        discover_classes(tmp_path / "non_existent_folder")

    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="No class subdirectories"):
        discover_classes(empty_dir)


def test_inspect_dataset_and_weights(tmp_path: Path):
    """Verify inspection metrics and balanced class weight calculation."""
    dataset_dir = tmp_path / "inspect_test"
    dataset_dir.mkdir()

    # Create class A with 2 images of size 100x200
    dir_a = dataset_dir / "class_a"
    dir_a.mkdir()
    for i in range(2):
        img = Image.new("RGB", (100, 200), color=(10, 10, 10))
        img.save(dir_a / f"img_{i}.jpg")

    # Create class B with 4 images of size 50x50
    dir_b = dataset_dir / "class_b"
    dir_b.mkdir()
    for i in range(4):
        img = Image.new("RGB", (50, 50), color=(20, 20, 20))
        img.save(dir_b / f"img_{i}.png")

    records, stats = inspect_dataset(dataset_dir)
    assert len(records) == 6
    assert stats.valid_images_count == 6
    assert stats.classes == ["class_a", "class_b"]

    # Balanced weight: Total / (num_classes * count)
    # class_a: 6 / (2 * 2) = 1.5
    # class_b: 6 / (2 * 4) = 0.75
    assert stats.class_summaries["class_a"].class_weight == 1.5
    assert stats.class_summaries["class_b"].class_weight == 0.75
    assert stats.imbalance_ratio == 2.0


def test_run_eda_generates_reports(tmp_path: Path):
    """Verify full EDA pipeline generates plots, JSON, and Markdown reports."""
    dataset_dir = tmp_path / "eda_test"
    dataset_dir.mkdir()
    class_dir = dataset_dir / "recyclables"
    class_dir.mkdir()
    img = Image.new("RGB", (64, 64), color="blue")
    img.save(class_dir / "item.jpg")

    reports_dir = tmp_path / "reports_out"
    stats = run_eda(data_dir=dataset_dir, output_dir=reports_dir)

    assert stats.valid_images_count == 1
    assert (reports_dir / "eda_summary.json").exists()
    assert (reports_dir / "eda_report.md").exists()
    assert (reports_dir / "class_distribution.png").exists()
    assert (reports_dir / "image_resolution_distribution.png").exists()


def test_real_dataset_structure_if_present():
    """Verify data/raw directory structure and counts if dataset is downloaded."""
    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        return

    # If class subdirectories exist, verify them dynamically
    classes = discover_classes(raw_dir)
    assert len(classes) == 6
    total_files = sum(len(list((raw_dir / c).glob("*"))) for c in classes)
    assert total_files >= 2000
