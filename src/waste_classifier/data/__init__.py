"""Data acquisition, ingestion, and validation modules."""

from waste_classifier.data.download import acquire_trashnet, validate_and_clean_dataset
from waste_classifier.data.ingestion import (
    ClassSummary,
    DatasetStats,
    ImageRecord,
    discover_classes,
    inspect_dataset,
    run_eda,
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
]
