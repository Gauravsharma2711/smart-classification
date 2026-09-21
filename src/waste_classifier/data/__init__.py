"""Data acquisition, ingestion, validation, and split modules."""

from waste_classifier.data.download import acquire_trashnet, validate_and_clean_dataset
from waste_classifier.data.ingestion import (
    ClassSummary,
    DatasetStats,
    ImageRecord,
    discover_classes,
    inspect_dataset,
    run_eda,
)
from waste_classifier.data.splits import (
    compute_image_hash,
    create_stratified_splits,
    find_cross_split_duplicates,
    generate_and_save_splits,
    load_split_manifest,
    save_split_manifest,
    validate_split_disjointness,
)

__all__ = [
    "acquire_trashnet",
    "validate_and_clean_dataset",
    "discover_classes",
    "inspect_dataset",
    "run_eda",
    "ClassSummary",
    "DatasetStats",
    "ImageRecord",
    "compute_image_hash",
    "create_stratified_splits",
    "find_cross_split_duplicates",
    "generate_and_save_splits",
    "load_split_manifest",
    "save_split_manifest",
    "validate_split_disjointness",
]
