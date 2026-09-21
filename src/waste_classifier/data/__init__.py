"""Data acquisition, ingestion, validation, splits, and transforms modules."""

from waste_classifier.data.datamodule import WasteDataModule
from waste_classifier.data.dataset import WasteDataset
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
from waste_classifier.data.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    assert_eval_transform_is_deterministic,
    build_eval_transforms,
    build_inference_transforms,
    build_train_transforms,
    is_transform_deterministic,
    load_augmentation_config,
    preprocess_image,
)

__all__ = [
    "WasteDataset",
    "WasteDataModule",
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
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "build_train_transforms",
    "build_eval_transforms",
    "build_inference_transforms",
    "preprocess_image",
    "is_transform_deterministic",
    "assert_eval_transform_is_deterministic",
    "load_augmentation_config",
]
