"""Data acquisition and validation for TrashNet dataset.

Implements FR-1:
- Download and unpack the dataset into folder-per-class structure.
- Remove unreadable or corrupted images.
- Print and record verified class counts.
"""

from __future__ import annotations

import json
import logging
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TRASHNET_URL = (
    "https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip"
)
EXPECTED_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def download_dataset(url: str, output_zip: Path | str) -> Path:
    """Download dataset archive with progress logging."""
    output_path = Path(output_zip)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading dataset from {url} to {output_path}...")

    def _progress(count: int, block_size: int, total_size: int) -> None:
        if total_size > 0 and count % 500 == 0:
            percent = int(count * block_size * 100 / total_size)
            logger.info(f"Download progress: {min(percent, 100)}%")

    urllib.request.urlretrieve(url, output_path, reporthook=_progress)
    logger.info(f"Downloaded {output_path.stat().st_size} bytes.")
    return output_path


def _find_class_root(root: Path) -> Path | None:
    """Recursively search for directory containing all expected classes."""
    for dirpath in [root] + [p for p in root.rglob("*") if p.is_dir()]:
        if all((dirpath / cls_name).is_dir() for cls_name in EXPECTED_CLASSES):
            return dirpath
    return None


def extract_and_organize(zip_path: Path | str, target_dir: Path | str) -> Path:
    """Extract zip archive and ensure clean folder-per-class layout directly under target_dir."""
    zip_p = Path(zip_path)
    target_p = Path(target_dir)
    target_p.mkdir(parents=True, exist_ok=True)

    extract_tmp = target_p / "_extract_tmp"
    if extract_tmp.exists():
        shutil.rmtree(extract_tmp)
    extract_tmp.mkdir(parents=True, exist_ok=True)

    logger.info(f"Extracting {zip_p} to temporary directory {extract_tmp}...")
    with zipfile.ZipFile(zip_p, "r") as zip_ref:
        zip_ref.extractall(extract_tmp)

    source_root = _find_class_root(extract_tmp)
    if source_root is None:
        raise RuntimeError(
            f"Could not locate directory with classes {EXPECTED_CLASSES} inside {zip_p}"
        )

    logger.info(f"Organizing classes from {source_root} directly into {target_p}...")
    for class_name in EXPECTED_CLASSES:
        class_src = source_root / class_name
        class_dst = target_p / class_name
        class_dst.mkdir(parents=True, exist_ok=True)
        for item in class_src.iterdir():
            if item.is_file() and not item.name.startswith("."):
                shutil.copy2(item, class_dst / item.name)

    # Clean up extraction temp directory
    shutil.rmtree(extract_tmp, ignore_errors=True)
    logger.info(f"Extraction and organization complete directly in {target_p}")
    return target_p


def validate_and_clean_dataset(
    dataset_dir: Path | str, remove_corrupt: bool = True
) -> dict[str, Any]:
    """Validate all images in dataset_dir, remove corrupted ones, and return statistics."""
    dataset_p = Path(dataset_dir)
    if not dataset_p.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_p}")

    class_counts: dict[str, int] = {}
    corrupted_files: list[str] = []
    unsupported_files: list[str] = []
    total_images = 0

    for class_name in EXPECTED_CLASSES:
        class_dir = dataset_p / class_name
        if not class_dir.exists() or not class_dir.is_dir():
            class_counts[class_name] = 0
            continue

        valid_count = 0
        for file_path in sorted(class_dir.iterdir()):
            if file_path.name.startswith("."):
                file_path.unlink()
                continue

            if file_path.suffix.lower() not in VALID_EXTENSIONS:
                unsupported_files.append(str(file_path))
                if remove_corrupt:
                    file_path.unlink()
                continue

            # Verify image readability and decoding
            is_valid = False
            try:
                with Image.open(file_path) as img:
                    img.verify()
                # Second pass to ensure actual decoding succeeds
                with Image.open(file_path) as img:
                    img.load()
                is_valid = True
            except Exception as e:
                logger.warning(f"Corrupt image detected: {file_path} (Error: {e})")
                corrupted_files.append(str(file_path))
                if remove_corrupt:
                    file_path.unlink()

            if is_valid:
                valid_count += 1

        class_counts[class_name] = valid_count
        total_images += valid_count

    stats: dict[str, Any] = {
        "dataset_name": "TrashNet",
        "source": "https://huggingface.co/datasets/garythung/trashnet",
        "license": "MIT",
        "dataset_dir": str(dataset_p.as_posix()),
        "classes": EXPECTED_CLASSES,
        "class_counts": class_counts,
        "total_valid_images": total_images,
        "corrupted_images_count": len(corrupted_files),
        "corrupted_files": corrupted_files,
        "unsupported_files_count": len(unsupported_files),
    }

    return stats


def acquire_trashnet(
    target_dir: Path | str = "data/raw",
    url: str = DEFAULT_TRASHNET_URL,
    keep_archive: bool = False,
) -> dict[str, Any]:
    """Complete acquisition pipeline: download, extract, clean, and validate."""
    target_p = Path(target_dir)
    target_p.mkdir(parents=True, exist_ok=True)

    # Check if classes are nested inside target_dir and flatten them
    found_root = _find_class_root(target_p)
    if found_root is not None and found_root != target_p:
        logger.info(f"Flattening classes from {found_root} into {target_p}...")
        for class_name in EXPECTED_CLASSES:
            src_c = found_root / class_name
            dst_c = target_p / class_name
            if src_c != dst_c and src_c.exists():
                dst_c.mkdir(parents=True, exist_ok=True)
                for item in src_c.iterdir():
                    if item.is_file() and not item.name.startswith("."):
                        shutil.move(str(item), str(dst_c / item.name))
        shutil.rmtree(found_root, ignore_errors=True)

    # Check if dataset is already unpacked and valid
    if all((target_p / cls_name).is_dir() for cls_name in EXPECTED_CLASSES):
        existing_images = sum(len(list((target_p / c).glob("*"))) for c in EXPECTED_CLASSES)
        if existing_images >= 2000:
            logger.info(
                f"Dataset already present in {target_p} with {existing_images} files. Validating..."
            )
            stats = validate_and_clean_dataset(target_p, remove_corrupt=True)
            _save_metadata(stats, target_p)
            return stats

    zip_path = target_p.parent / "trashnet_raw.zip"
    try:
        download_dataset(url, zip_path)
        extract_and_organize(zip_path, target_p)
        stats = validate_and_clean_dataset(target_p, remove_corrupt=True)
        _save_metadata(stats, target_p)
        return stats
    finally:
        if not keep_archive and zip_path.exists():
            zip_path.unlink(missing_ok=True)


def _save_metadata(stats: dict[str, Any], target_dir: Path) -> None:
    """Save dataset metadata JSON to reports/ and data/."""
    meta_path = target_dir / "dataset_summary.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_summary = reports_dir / "dataset_summary.json"
    with open(report_summary, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Dataset metadata saved to {meta_path} and {report_summary}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and validate TrashNet dataset.")
    parser.add_argument(
        "--target-dir", default="data/raw", help="Target directory for raw dataset."
    )
    parser.add_argument(
        "--url", default=DEFAULT_TRASHNET_URL, help="Download URL for TrashNet zip."
    )
    args = parser.parse_args()

    results = acquire_trashnet(target_dir=args.target_dir, url=args.url)
    print("\n--- DATASET INGESTION SUMMARY ---")
    print(f"Dataset: {results['dataset_name']}")
    print(f"Source: {results['source']}")
    print(f"License: {results['license']}")
    print(f"Location: {results['dataset_dir']}")
    print(f"Total Valid Images: {results['total_valid_images']}")
    print(f"Corrupt Images Removed: {results['corrupted_images_count']}")
    print("Class Counts:")
    for cls, count in results["class_counts"].items():
        print(f"  {cls:<12}: {count}")
    print("---------------------------------\n")
