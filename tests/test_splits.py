"""Tests for stratified leak-free split generation and hash deduplication (FR-2)."""

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from waste_classifier.data.splits import (
    compute_image_hash,
    create_stratified_splits,
    find_cross_split_duplicates,
    load_split_manifest,
    save_split_manifest,
    validate_split_disjointness,
)


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    """Create a synthetic dataset with 3 classes and 20 images each (60 total)."""
    root = tmp_path / "raw"
    root.mkdir()
    for c_idx, class_name in enumerate(["cardboard", "glass", "metal"]):
        class_dir = root / class_name
        class_dir.mkdir()
        for i in range(20):
            # Create distinct image patterns to yield unique pHashes
            img = Image.new("RGB", (32, 32), color=((c_idx + 1) * 40, (i + 1) * 8, 100))
            for px in range(i + 1):
                img.putpixel((px % 32, (px * 3) % 32), (255, (px * 20) % 255, 0))
            img.save(class_dir / f"img_{i:02d}.jpg")
    return root


def test_stratified_split_proportions(synthetic_dataset: Path):
    """Verify stratified split produces approximate 70/15/15 ratios per class."""
    df = create_stratified_splits(
        data_dir=synthetic_dataset,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
        compute_hashes=True,
    )

    assert len(df) == 60
    assert set(df["split"]) == {"train", "val", "test"}

    # Check stratification per class (20 images per class -> 14 train, 3 val, 3 test)
    for class_name in ["cardboard", "glass", "metal"]:
        class_df = df[df["class_name"] == class_name]
        assert len(class_df) == 20
        train_count = len(class_df[class_df["split"] == "train"])
        val_count = len(class_df[class_df["split"] == "val"])
        test_count = len(class_df[class_df["split"] == "test"])

        assert train_count == 14
        assert val_count == 3
        assert test_count == 3


def test_validate_split_disjointness_success(synthetic_dataset: Path):
    """Verify valid split passes disjointness assertion."""
    df = create_stratified_splits(synthetic_dataset, seed=42, compute_hashes=False)
    # Should not raise any error
    validate_split_disjointness(df)


def test_validate_split_disjointness_catches_train_val_overlap():
    """Verify intentional leakage between train and val triggers AssertionError."""
    leaky_df = pd.DataFrame(
        [
            {"rel_path": "glass/01.jpg", "class_name": "glass", "split": "train"},
            {"rel_path": "glass/02.jpg", "class_name": "glass", "split": "train"},
            {"rel_path": "glass/01.jpg", "class_name": "glass", "split": "val"},  # Overlap!
            {"rel_path": "glass/03.jpg", "class_name": "glass", "split": "test"},
        ]
    )
    with pytest.raises(AssertionError, match="Data leakage detected.*train and val"):
        validate_split_disjointness(leaky_df)


def test_validate_split_disjointness_catches_train_test_overlap():
    """Verify intentional leakage between train and test triggers AssertionError."""
    leaky_df = pd.DataFrame(
        [
            {"rel_path": "metal/01.jpg", "class_name": "metal", "split": "train"},
            {"rel_path": "metal/02.jpg", "class_name": "metal", "split": "val"},
            {"rel_path": "metal/01.jpg", "class_name": "metal", "split": "test"},  # Overlap!
        ]
    )
    with pytest.raises(AssertionError, match="Data leakage detected.*train and test"):
        validate_split_disjointness(leaky_df)


def test_perceptual_hash_cross_split_duplicate_detection(tmp_path: Path):
    """Verify perceptual hash detects cross-split duplicate images."""
    img_dir = tmp_path / "hash_test"
    img_dir.mkdir()

    img1 = Image.new("RGB", (64, 64), color="red")
    img1_path = img_dir / "img1.jpg"
    img1.save(img1_path)

    # Identical image with different filename
    img2_path = img_dir / "img2.jpg"
    img1.save(img2_path)

    h1 = compute_image_hash(img1_path)
    h2 = compute_image_hash(img2_path)
    assert h1 == h2

    df = pd.DataFrame(
        [
            {"rel_path": "img1.jpg", "class_name": "trash", "split": "train", "phash": h1},
            {"rel_path": "img2.jpg", "class_name": "trash", "split": "test", "phash": h2},
        ]
    )

    dups = find_cross_split_duplicates(df, max_hamming_distance=0)
    assert len(dups) == 1
    assert dups[0]["path_1"] == "img1.jpg"
    assert dups[0]["path_2"] == "img2.jpg"
    assert dups[0]["hamming_distance"] == 0


def test_deterministic_reproduction(synthetic_dataset: Path):
    """Verify fixed seed produces identical split manifest byte-for-byte."""
    df1 = create_stratified_splits(synthetic_dataset, seed=123, compute_hashes=False)
    df2 = create_stratified_splits(synthetic_dataset, seed=123, compute_hashes=False)
    pd.testing.assert_frame_equal(df1, df2)

    # Different seed produces different split allocation
    df3 = create_stratified_splits(synthetic_dataset, seed=456, compute_hashes=False)
    assert not df1["split"].equals(df3["split"])


def test_save_and_load_manifest(tmp_path: Path, synthetic_dataset: Path):
    """Verify saving and reloading split manifest preserves data integrity."""
    df = create_stratified_splits(synthetic_dataset, seed=42, compute_hashes=False)
    manifest_file = tmp_path / "splits.csv"

    save_split_manifest(df, manifest_file)
    assert manifest_file.exists()

    loaded_df = load_split_manifest(manifest_file)
    pd.testing.assert_frame_equal(df, loaded_df)


def test_real_splits_csv_disjointness_and_counts():
    """Verify data/splits.csv manifest has zero overlap and zero cross-split duplicates."""
    manifest_path = Path("data/splits.csv")
    if not manifest_path.exists():
        pytest.skip("data/splits.csv does not exist yet")

    df = load_split_manifest(manifest_path)
    assert len(df) == 2527

    # Acceptance criteria: zero overlap between splits
    validate_split_disjointness(df)

    train_paths = set(df[df["split"] == "train"]["rel_path"])
    val_paths = set(df[df["split"] == "val"]["rel_path"])
    test_paths = set(df[df["split"] == "test"]["rel_path"])

    assert len(train_paths & val_paths) == 0
    assert len(train_paths & test_paths) == 0
    assert len(val_paths & test_paths) == 0
    assert len(train_paths) + len(val_paths) + len(test_paths) == 2527

    # Check that cross-split duplicates are 0
    dups = find_cross_split_duplicates(df, max_hamming_distance=0)
    assert len(dups) == 0, f"Found {len(dups)} cross-split duplicate collisions: {dups}"

    # Verify per-class stratification
    for cls in ["cardboard", "glass", "metal", "paper", "plastic", "trash"]:
        cls_df = df[df["class_name"] == cls]
        train_ratio = len(cls_df[cls_df["split"] == "train"]) / len(cls_df)
        val_ratio = len(cls_df[cls_df["split"] == "val"]) / len(cls_df)
        test_ratio = len(cls_df[cls_df["split"] == "test"]) / len(cls_df)

        assert 0.68 <= train_ratio <= 0.72
        assert 0.13 <= val_ratio <= 0.17
        assert 0.13 <= test_ratio <= 0.17
