"""Stratified leak-free dataset splitting and perceptual hash deduplication.

Implements FR-2:
- Stratified 70/15/15 train/val/test split with fixed seed.
- Manifest CSV persistence (data/splits.csv).
- Zero file overlap assertion across splits.
- Cross-split perceptual hash collision and duplicate detection.
- Deterministic reproduction.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import imagehash
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from PIL import Image
from sklearn.model_selection import train_test_split

from waste_classifier.data.ingestion import SUPPORTED_EXTENSIONS, discover_classes

logger = logging.getLogger(__name__)


def compute_image_hash(image_path: Path | str, hash_size: int = 8) -> str:
    """Compute perceptual hash (pHash) for an image.

    Args:
        image_path: Path to the image file.
        hash_size: Size of the hash matrix (default 8 -> 64-bit hash).

    Returns:
        Hexadecimal string representation of the perceptual hash.
    """
    path = Path(image_path)
    with Image.open(path) as img:
        # Convert to RGB to ensure consistent hashing
        if img.mode != "RGB":
            img = img.convert("RGB")
        h = imagehash.phash(img, hash_size=hash_size)
    return str(h)


def create_stratified_splits(
    data_dir: Path | str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    compute_hashes: bool = True,
    group_by_hash: bool = True,
) -> pd.DataFrame:
    """Create a stratified train/val/test split across all classes.

    Args:
        data_dir: Root dataset folder containing class directories.
        train_ratio: Fraction for training split (default 0.70).
        val_ratio: Fraction for validation split (default 0.15).
        test_ratio: Fraction for test split (default 0.15).
        seed: Random seed for reproducibility.
        compute_hashes: Whether to compute perceptual hashes for each image.
        group_by_hash: Whether to group identical perceptual hashes together into the same split
                       to prevent cross-split near-duplicate data leakage (PRD 7.2).

    Returns:
        DataFrame with columns ['rel_path', 'class_name', 'split', 'phash'].
    """
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0, atol=1e-4):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {train_ratio} + {val_ratio} + {test_ratio} = {total_ratio}"
        )

    data_path = Path(data_dir)
    classes = discover_classes(data_path)

    items: list[dict[str, Any]] = []
    for cls_name in classes:
        cls_dir = data_path / cls_name
        for f in sorted(cls_dir.iterdir()):
            if (
                f.is_file()
                and not f.name.startswith(".")
                and f.suffix.lower() in SUPPORTED_EXTENSIONS
            ):
                rel_p = f.relative_to(data_path).as_posix()
                items.append({"rel_path": rel_p, "class_name": cls_name, "abs_path": str(f)})

    if not items:
        raise ValueError(f"No image files found in {data_path}")

    df = pd.DataFrame(items)

    if compute_hashes:
        logger.info("Computing perceptual hashes for split verification...")
        df["phash"] = [compute_image_hash(row["abs_path"]) for _, row in df.iterrows()]
    else:
        df["phash"] = ""

    temp_ratio = val_ratio + test_ratio
    val_share = val_ratio / temp_ratio

    if group_by_hash and compute_hashes and df["phash"].nunique() < len(df):
        logger.info(
            f"Grouping {len(df)} images into {df['phash'].nunique()} perceptual hash groups to prevent cross-split duplicate leakage..."
        )
        # Unique group dataframe (first instance per hash defines representative class)
        group_df = df.groupby("phash").first().reset_index()

        min_class_count = group_df["class_name"].value_counts().min()
        stratify_train = group_df["class_name"] if min_class_count >= 2 else None

        train_g, temp_g = train_test_split(
            group_df,
            test_size=temp_ratio,
            random_state=seed,
            stratify=stratify_train,
        )

        min_temp_count = temp_g["class_name"].value_counts().min()
        stratify_val = temp_g["class_name"] if min_temp_count >= 2 else None

        val_g, test_g = train_test_split(
            temp_g,
            test_size=(1.0 - val_share),
            random_state=seed,
            stratify=stratify_val,
        )

        split_map: dict[str, str] = {}
        for h in train_g["phash"]:
            split_map[h] = "train"
        for h in val_g["phash"]:
            split_map[h] = "val"
        for h in test_g["phash"]:
            split_map[h] = "test"

        df["split"] = df["phash"].map(split_map)
        manifest_df = df.sort_values(by=["class_name", "rel_path"]).reset_index(drop=True)
    else:
        # Standard sample-level stratified split
        train_df, temp_df = train_test_split(
            df,
            test_size=temp_ratio,
            random_state=seed,
            stratify=df["class_name"],
        )

        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1.0 - val_share),
            random_state=seed,
            stratify=temp_df["class_name"],
        )

        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()

        train_df["split"] = "train"
        val_df["split"] = "val"
        test_df["split"] = "test"

        manifest_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        manifest_df = manifest_df.sort_values(by=["class_name", "rel_path"]).reset_index(drop=True)

    # Drop abs_path so manifest stores portable relative paths
    manifest_df = manifest_df[["rel_path", "class_name", "split", "phash"]]

    # Validate disjointness before returning
    validate_split_disjointness(manifest_df)
    return manifest_df


def validate_split_disjointness(df: pd.DataFrame) -> None:
    """Assert that there is strictly zero file overlap between splits.

    Args:
        df: DataFrame containing at least 'rel_path' and 'split'.

    Raises:
        AssertionError: If any path appears in more than one split or is duplicated.
    """
    splits = df["split"].unique()
    expected_splits = {"train", "val", "test"}
    if not expected_splits.issubset(set(splits)):
        raise ValueError(f"Manifest missing expected splits. Found: {splits}")

    train_paths = set(df[df["split"] == "train"]["rel_path"])
    val_paths = set(df[df["split"] == "val"]["rel_path"])
    test_paths = set(df[df["split"] == "test"]["rel_path"])

    train_val_overlap = train_paths.intersection(val_paths)
    train_test_overlap = train_paths.intersection(test_paths)
    val_test_overlap = val_paths.intersection(test_paths)

    if train_val_overlap:
        raise AssertionError(
            f"Data leakage detected! {len(train_val_overlap)} files overlap between train and val: {list(train_val_overlap)[:5]}"
        )
    if train_test_overlap:
        raise AssertionError(
            f"Data leakage detected! {len(train_test_overlap)} files overlap between train and test: {list(train_test_overlap)[:5]}"
        )
    if val_test_overlap:
        raise AssertionError(
            f"Data leakage detected! {len(val_test_overlap)} files overlap between val and test: {list(val_test_overlap)[:5]}"
        )

    # Check for overall duplicate paths in the manifest
    if df["rel_path"].duplicated().any():
        dup_count = df["rel_path"].duplicated().sum()
        dups = df[df["rel_path"].duplicated()]["rel_path"].tolist()
        raise AssertionError(
            f"Duplicate entries found in split manifest: {dup_count} duplicates (e.g. {dups[:5]})"
        )


def find_cross_split_duplicates(
    df: pd.DataFrame, max_hamming_distance: int = 0
) -> list[dict[str, Any]]:
    """Scan for identical or near-duplicate perceptual hashes across different splits.

    Args:
        df: DataFrame with 'rel_path', 'split', and 'phash'.
        max_hamming_distance: Maximum Hamming distance to consider as duplicate (0 = exact match).

    Returns:
        List of cross-split duplicate records.
    """
    if "phash" not in df.columns or df["phash"].str.len().eq(0).all():
        logger.warning("No perceptual hashes found in manifest; skipping duplicate scan.")
        return []

    duplicates: list[dict[str, Any]] = []

    # Map each hash to its image representations
    parsed_hashes = [imagehash.hex_to_hash(h) for h in df["phash"]]
    records = df[["rel_path", "split", "class_name", "phash"]].to_dict(orient="records")

    num_records = len(records)
    logger.info(
        f"Scanning {num_records} images for cross-split hash collisions (max Hamming dist: {max_hamming_distance})..."
    )

    # For max_hamming_distance == 0, use dictionary grouping for O(N) fast lookup
    if max_hamming_distance == 0:
        hash_map: dict[str, list[dict[str, Any]]] = {}
        for rec in records:
            h = rec["phash"]
            hash_map.setdefault(h, []).append(rec)

        for h, group in hash_map.items():
            splits_in_group = {g["split"] for g in group}
            if len(splits_in_group) > 1:
                # Collision across splits
                for i in range(len(group)):
                    for j in range(i + 1, len(group)):
                        if group[i]["split"] != group[j]["split"]:
                            duplicates.append(
                                {
                                    "path_1": group[i]["rel_path"],
                                    "split_1": group[i]["split"],
                                    "path_2": group[j]["rel_path"],
                                    "split_2": group[j]["split"],
                                    "hamming_distance": 0,
                                    "phash": h,
                                }
                            )
        return duplicates

    # Otherwise compare pairwise across splits
    for i in range(num_records):
        for j in range(i + 1, num_records):
            if records[i]["split"] == records[j]["split"]:
                continue
            dist = parsed_hashes[i] - parsed_hashes[j]
            if dist <= max_hamming_distance:
                duplicates.append(
                    {
                        "path_1": records[i]["rel_path"],
                        "split_1": records[i]["split"],
                        "path_2": records[j]["rel_path"],
                        "split_2": records[j]["split"],
                        "hamming_distance": int(dist),
                        "phash_1": records[i]["phash"],
                        "phash_2": records[j]["phash"],
                    }
                )

    return duplicates


def save_split_manifest(df: pd.DataFrame, manifest_path: Path | str = "data/splits.csv") -> Path:
    """Save the validated split manifest CSV."""
    out_p = Path(manifest_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_p, index=False)
    logger.info(f"Saved split manifest with {len(df)} records to {out_p}")
    return out_p


def load_split_manifest(manifest_path: Path | str = "data/splits.csv") -> pd.DataFrame:
    """Load and validate an existing split manifest CSV."""
    p = Path(manifest_path)
    if not p.exists():
        raise FileNotFoundError(f"Manifest not found: {p}")
    df = pd.read_csv(
        p, dtype={"rel_path": str, "class_name": str, "split": str, "phash": str}
    ).fillna({"phash": ""})
    validate_split_disjointness(df)
    return df


def generate_and_save_splits(
    config_path: Path | str = "configs/base.yaml", force: bool = False
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Orchestrate split generation from config file, checking duplicates and saving reports.

    Args:
        config_path: Path to base YAML config.
        force: If True, regenerates even if manifest exists.

    Returns:
        Tuple of (split DataFrame, list of cross-split duplicate records).
    """
    cfg = OmegaConf.load(config_path)
    data_dir = cfg.data.root
    manifest_path = Path(cfg.data.manifest)
    seed = int(cfg.seed)
    train_ratio = float(cfg.data.split.train)
    val_ratio = float(cfg.data.split.val)
    test_ratio = float(cfg.data.split.test)

    if manifest_path.exists() and not force:
        logger.info(f"Manifest {manifest_path} already exists. Loading and validating...")
        df = load_split_manifest(manifest_path)
        duplicates = find_cross_split_duplicates(df, max_hamming_distance=0)
        return df, duplicates

    logger.info(
        f"Generating stratified split ({train_ratio}/{val_ratio}/{test_ratio}) with seed {seed}..."
    )
    df = create_stratified_splits(
        data_dir=data_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        compute_hashes=True,
    )

    save_split_manifest(df, manifest_path)

    # Check for perceptual hash collisions across splits
    duplicates = find_cross_split_duplicates(df, max_hamming_distance=0)
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    dup_report_path = reports_dir / "duplicate_analysis.json"
    with open(dup_report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_images": len(df),
                "identical_cross_split_duplicates_count": len(duplicates),
                "identical_cross_split_duplicates": duplicates,
            },
            f,
            indent=2,
        )

    # Generate split distribution summary report
    split_summary = df.groupby(["class_name", "split"]).size().unstack(fill_value=0).reset_index()
    split_summary["total"] = split_summary["train"] + split_summary["val"] + split_summary["test"]
    split_summary_path = reports_dir / "split_distribution.md"
    with open(split_summary_path, "w", encoding="utf-8") as f:
        f.write("# Dataset Split Distribution (FR-2)\n\n")
        f.write(f"- **Seed**: {seed}\n")
        f.write(f"- **Total Samples**: {len(df)}\n")
        f.write(
            f"- **Train**: {len(df[df['split'] == 'train'])} ({len(df[df['split'] == 'train']) / len(df):.1%})\n"
        )
        f.write(
            f"- **Validation**: {len(df[df['split'] == 'val'])} ({len(df[df['split'] == 'val']) / len(df):.1%})\n"
        )
        f.write(
            f"- **Test**: {len(df[df['split'] == 'test'])} ({len(df[df['split'] == 'test']) / len(df):.1%})\n"
        )
        f.write(f"- **Cross-split Duplicate Collisions**: {len(duplicates)}\n\n")
        f.write("### Per-Class Split Counts\n\n")
        f.write("| Class Name | Train | Validation | Test | Total |\n")
        f.write("|---|---|---|---|---|\n")
        for _, row in split_summary.iterrows():
            f.write(
                f"| `{row['class_name']}` | {row['train']} | {row['val']} | {row['test']} | {row['total']} |\n"
            )
        f.write("\n")

    logger.info(f"Split reports saved to {dup_report_path} and {split_summary_path}")
    return df, duplicates


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create and validate leak-free dataset split.")
    parser.add_argument(
        "--config", default="configs/base.yaml", help="Path to base configuration YAML."
    )
    parser.add_argument(
        "--force", action="store_true", help="Force regeneration of split manifest."
    )
    args = parser.parse_args()

    split_df, dups = generate_and_save_splits(config_path=args.config, force=args.force)
    print("\n=== SPLIT SUMMARY ===")
    print(f"Total samples: {len(split_df)}")
    print(f"Train samples: {len(split_df[split_df['split'] == 'train'])}")
    print(f"Val samples:   {len(split_df[split_df['split'] == 'val'])}")
    print(f"Test samples:  {len(split_df[split_df['split'] == 'test'])}")
    print(f"Cross-split duplicates: {len(dups)}")
    print("=====================\n")
