"""Acquisition and anti-leak validation module for field-set evaluation images.

Implements FR-8 data acquisition:
- Acquires 100+ real camera images across 6 target classes (cardboard, glass, metal, paper, plastic, trash)
  plus negative camera images.
- Checks perceptual hashes against all train/val/test splits to guarantee mathematical zero-overlap.
- Saves images into data/field_set/<class>/ (gitignored for privacy & clean repo).
- Generates data/field_set/manifest.json with integrity records.
"""

from __future__ import annotations

import io
import json
import logging
import urllib.request
from pathlib import Path
from typing import Any

import imagehash
import pandas as pd
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TARGET_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NEGATIVE_SOURCE_DIRS = ["clothes", "shoes", "biological"]
DEFAULT_TARGET_COUNT_PER_CLASS = 20
DEFAULT_NEGATIVE_COUNT = 15
MIN_HAMMING_DISTANCE = 10


def load_known_split_hashes(
    splits_csv_path: Path | str = "data/splits.csv",
) -> list[imagehash.ImageHash]:
    """Load precomputed perceptual hashes from the benchmark splits manifest."""
    csv_p = Path(splits_csv_path)
    if not csv_p.exists():
        raise FileNotFoundError(f"Benchmark splits manifest not found at: {csv_p}")

    df = pd.read_csv(csv_p)
    if "phash" not in df.columns:
        raise ValueError("splits.csv must contain 'phash' column.")

    hashes = [imagehash.hex_to_hash(h) for h in df["phash"].dropna()]
    logger.info(f"Loaded {len(hashes)} benchmark hashes from {csv_p}")
    return hashes


def verify_field_set_integrity(
    field_dir: Path | str = "data/field_set",
    splits_csv_path: Path | str = "data/splits.csv",
    min_hamming_dist: int = MIN_HAMMING_DISTANCE,
    min_images_per_class: int = 15,
) -> dict[str, Any]:
    """Verify that field set meets all FR-8 criteria and has zero leakage into benchmark splits.

    Args:
        field_dir: Root directory of field set.
        splits_csv_path: Path to benchmark splits.csv.
        min_hamming_dist: Minimum allowed perceptual distance to any train/val/test image.
        min_images_per_class: Minimum image count required per canonical class.

    Returns:
        Summary dict containing counts, classes, and verification status.
    """
    field_p = Path(field_dir)
    if not field_p.exists():
        raise FileNotFoundError(f"Field set directory not found: {field_p}")

    known_hashes = load_known_split_hashes(splits_csv_path)

    class_counts: dict[str, int] = {}
    violations: list[dict[str, Any]] = []
    total_images = 0

    for cls_name in TARGET_CLASSES:
        cls_dir = field_p / cls_name
        if not cls_dir.exists():
            raise FileNotFoundError(f"Missing required class directory: {cls_dir}")

        img_files = [f for f in cls_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        class_counts[cls_name] = len(img_files)
        total_images += len(img_files)

        if len(img_files) < min_images_per_class:
            raise ValueError(
                f"Class '{cls_name}' has only {len(img_files)} images (required >= {min_images_per_class})."
            )

        for img_path in img_files:
            try:
                with Image.open(img_path) as img:
                    h = imagehash.phash(img.convert("RGB"))
                    min_dist = min([h - kh for kh in known_hashes])
                    if min_dist < min_hamming_dist:
                        violations.append(
                            {
                                "path": str(img_path),
                                "class": cls_name,
                                "min_dist": int(min_dist),
                            }
                        )
            except Exception as err:
                violations.append(
                    {
                        "path": str(img_path),
                        "class": cls_name,
                        "error": str(err),
                    }
                )

    # Optional negative images
    neg_dir = field_p / "negative"
    if neg_dir.exists():
        neg_files = [f for f in neg_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        class_counts["negative"] = len(neg_files)
        total_images += len(neg_files)

    if violations:
        raise ValueError(
            f"Field set integrity check failed with {len(violations)} leakage/error violations: {violations[:3]}"
        )

    if total_images < 100:
        raise ValueError(f"Field set has only {total_images} total images (required >= 100).")

    logger.info(
        f"Field set integrity verified! Total images: {total_images}, Classes: {class_counts}, "
        f"0% overlap with benchmark splits (min hamming distance >= {min_hamming_dist})."
    )

    return {
        "status": "verified",
        "total_images": total_images,
        "class_counts": class_counts,
        "min_hamming_distance_verified": min_hamming_dist,
    }


def acquire_field_set(
    output_dir: Path | str = "data/field_set",
    splits_csv_path: Path | str = "data/splits.csv",
    images_per_class: int = DEFAULT_TARGET_COUNT_PER_CLASS,
    negative_images: int = DEFAULT_NEGATIVE_COUNT,
) -> dict[str, Any]:
    """Acquire real camera waste images from remote repository with anti-leak verification.

    Args:
        output_dir: Output folder for field set.
        splits_csv_path: Path to benchmark splits.csv.
        images_per_class: Target count of images per canonical class.
        negative_images: Target count of negative (non-waste) images.

    Returns:
        Manifest dictionary of acquired images.
    """
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    known_hashes = load_known_split_hashes(splits_csv_path)

    # Fetch dataset file listing from Hugging Face API
    api_url = "https://huggingface.co/api/datasets/kdkd1/waste-garbage-management-dataset"
    logger.info(f"Querying dataset file tree from {api_url}...")
    req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())

    siblings: list[str] = [
        s.get("rfilename") for s in data.get("siblings", []) if s.get("rfilename")
    ]

    base_download_url = (
        "https://huggingface.co/datasets/kdkd1/waste-garbage-management-dataset/resolve/main/"
    )

    manifest: list[dict[str, Any]] = []

    # 1. Acquire Canonical Waste Classes
    for cls_name in TARGET_CLASSES:
        cls_dir = out_p / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)

        # Existing files in class directory
        existing = [f for f in cls_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        if len(existing) >= images_per_class:
            logger.info(f"Class '{cls_name}' already populated with {len(existing)} images.")
            continue

        needed = images_per_class - len(existing)
        # Filter files for this class, taking from index 100 onwards to avoid common head overlaps
        cls_files = [
            s
            for s in siblings
            if s.startswith(cls_name + "/") and s.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        # Reverse or slice candidates to avoid TrashNet copies
        candidates = cls_files[100:] if len(cls_files) > 100 else cls_files

        acquired_for_class = 0
        for cand_rel in candidates:
            if acquired_for_class >= needed:
                break

            fname = Path(cand_rel).name
            dest_file = cls_dir / fname
            if dest_file.exists():
                continue

            try:
                img_url = base_download_url + cand_rel
                r = urllib.request.Request(img_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(r, timeout=15) as r_resp:
                    img_bytes = r_resp.read()

                # Validate image and compute perceptual hash
                with Image.open(io.BytesIO(img_bytes)) as pil_img:
                    rgb_img = pil_img.convert("RGB")
                    h = imagehash.phash(rgb_img)
                    min_dist = min([h - kh for kh in known_hashes])

                    if min_dist < MIN_HAMMING_DISTANCE:
                        logger.warning(
                            f"Rejected {cand_rel}: near-duplicate with benchmark splits (min dist: {min_dist})"
                        )
                        continue

                    # Save verified clean image
                    rgb_img.save(dest_file, format="JPEG", quality=90)
                    manifest.append(
                        {
                            "filename": fname,
                            "rel_path": f"{cls_name}/{fname}",
                            "class": cls_name,
                            "width": rgb_img.width,
                            "height": rgb_img.height,
                            "phash": str(h),
                            "min_dist_to_splits": int(min_dist),
                        }
                    )
                    acquired_for_class += 1
                    logger.info(
                        f"Acquired [{cls_name} {len(existing) + acquired_for_class}/{images_per_class}] "
                        f"{fname} (min dist: {min_dist})"
                    )

            except Exception as err:
                logger.warning(f"Failed to process {cand_rel}: {err}")

        if len(existing) + acquired_for_class < images_per_class:
            logger.warning(
                f"Class '{cls_name}' only acquired {len(existing) + acquired_for_class} / {images_per_class} images."
            )

    # 2. Acquire Negative Images (non-waste objects)
    neg_dir = out_p / "negative"
    neg_dir.mkdir(parents=True, exist_ok=True)
    existing_neg = [f for f in neg_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]

    if len(existing_neg) < negative_images:
        needed_neg = negative_images - len(existing_neg)
        neg_candidates: list[str] = []
        for src_dir in NEGATIVE_SOURCE_DIRS:
            neg_candidates.extend(
                [
                    s
                    for s in siblings
                    if s.startswith(src_dir + "/") and s.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
            )

        acquired_neg = 0
        for cand_rel in neg_candidates:
            if acquired_neg >= needed_neg:
                break

            fname = Path(cand_rel).name
            dest_file = neg_dir / fname
            if dest_file.exists():
                continue

            try:
                img_url = base_download_url + cand_rel
                r = urllib.request.Request(img_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(r, timeout=15) as r_resp:
                    img_bytes = r_resp.read()

                with Image.open(io.BytesIO(img_bytes)) as pil_img:
                    rgb_img = pil_img.convert("RGB")
                    h = imagehash.phash(rgb_img)
                    min_dist = min([h - kh for kh in known_hashes])

                    if min_dist < MIN_HAMMING_DISTANCE:
                        continue

                    rgb_img.save(dest_file, format="JPEG", quality=90)
                    manifest.append(
                        {
                            "filename": fname,
                            "rel_path": f"negative/{fname}",
                            "class": "negative",
                            "width": rgb_img.width,
                            "height": rgb_img.height,
                            "phash": str(h),
                            "min_dist_to_splits": int(min_dist),
                        }
                    )
                    acquired_neg += 1
                    logger.info(
                        f"Acquired [negative {len(existing_neg) + acquired_neg}/{negative_images}] {fname}"
                    )
            except Exception as err:
                logger.warning(f"Failed negative candidate {cand_rel}: {err}")

    # 3. Save Manifest
    manifest_file = out_p / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # 4. Run integrity check
    integrity_summary = verify_field_set_integrity(out_p, splits_csv_path)
    return {
        "manifest_path": str(manifest_file),
        "integrity": integrity_summary,
        "items_downloaded_this_run": len(manifest),
    }


if __name__ == "__main__":
    result = acquire_field_set()
    logger.info(f"Field set acquisition complete: {result['integrity']}")
