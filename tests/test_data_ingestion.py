"""Tests for dataset ingestion and validation (FR-1)."""

from pathlib import Path

from PIL import Image

from waste_classifier.data.download import EXPECTED_CLASSES, validate_and_clean_dataset


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


def test_real_dataset_structure_if_present():
    """Verify data/raw directory structure and counts if dataset is downloaded."""
    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        return

    # If class subdirectories exist, verify them
    present_classes = [c for c in EXPECTED_CLASSES if (raw_dir / c).is_dir()]
    if present_classes:
        assert set(present_classes) == set(EXPECTED_CLASSES)
        total_files = sum(len(list((raw_dir / c).glob("*"))) for c in EXPECTED_CLASSES)
        assert total_files >= 2000
