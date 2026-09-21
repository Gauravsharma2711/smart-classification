"""Data ingestion, discovery, and Exploratory Data Analysis (EDA) module.

Implements FR-1:
- Discovers dataset structure dynamically from config or filesystem.
- Detects unreadable or corrupted images.
- Produces per-class statistics and balanced class weights.
- Generates EDA visualization artifacts and summary tables in reports/.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Use non-interactive backend for headless environments
matplotlib.use("Agg")

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class ImageRecord:
    """Metadata for a single inspected image."""

    rel_path: str
    class_name: str
    width: int
    height: int
    channels: int
    mode: str
    file_size_bytes: int
    is_valid: bool
    error: str | None = None


@dataclass
class ClassSummary:
    """Per-class aggregated metrics."""

    class_name: str
    count: int
    percentage: float
    class_weight: float


@dataclass
class DatasetStats:
    """Aggregated dataset statistics and EDA summaries."""

    data_dir: str
    classes: list[str]
    total_images_scanned: int
    valid_images_count: int
    corrupt_images_count: int
    class_summaries: dict[str, ClassSummary]
    imbalance_ratio: float
    width_stats: dict[str, float]
    height_stats: dict[str, float]
    aspect_ratios: dict[str, float]
    file_size_kb_stats: dict[str, float]
    color_modes: dict[str, int]
    corrupted_files: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to a serializable dictionary."""
        d = asdict(self)
        return d


def discover_classes(data_dir: Path | str) -> list[str]:
    """Dynamically discover class directories inside data_dir without hardcoding names.

    Args:
        data_dir: Root directory containing class subfolders.

    Returns:
        Sorted list of class directory names.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_path}")

    classes = [
        d.name
        for d in data_path.iterdir()
        if d.is_dir() and not d.name.startswith(".") and not d.name.startswith("_")
    ]
    if not classes:
        raise ValueError(f"No class subdirectories found in: {data_path}")

    return sorted(classes)


def inspect_dataset(data_dir: Path | str) -> tuple[list[ImageRecord], DatasetStats]:
    """Scan and inspect all images in data_dir, recording dimensions and validity.

    Args:
        data_dir: Root dataset directory.

    Returns:
        Tuple of (list of image records, comprehensive dataset stats).
    """
    data_path = Path(data_dir)
    classes = discover_classes(data_path)

    records: list[ImageRecord] = []
    class_counts: dict[str, int] = {c: 0 for c in classes}
    corrupted_files: list[str] = []

    widths: list[int] = []
    heights: list[int] = []
    aspect_ratios: list[float] = []
    file_sizes_kb: list[float] = []
    color_modes: dict[str, int] = {}

    for class_name in classes:
        class_folder = data_path / class_name
        for file_path in sorted(class_folder.iterdir()):
            if file_path.name.startswith(".") or file_path.is_dir():
                continue

            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            rel_path = file_path.relative_to(data_path).as_posix()
            file_size = file_path.stat().st_size
            file_size_kb = file_size / 1024.0

            try:
                with Image.open(file_path) as img:
                    img.verify()
                # Reopen to load actual dimensions and verify decoding
                with Image.open(file_path) as img:
                    img.load()
                    w, h = img.size
                    mode = img.mode
                    channels = len(img.getbands())

                record = ImageRecord(
                    rel_path=rel_path,
                    class_name=class_name,
                    width=w,
                    height=h,
                    channels=channels,
                    mode=mode,
                    file_size_bytes=file_size,
                    is_valid=True,
                )
                records.append(record)
                class_counts[class_name] += 1
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h if h > 0 else 0.0)
                file_sizes_kb.append(file_size_kb)
                color_modes[mode] = color_modes.get(mode, 0) + 1

            except Exception as e:
                logger.warning(f"Unreadable image: {rel_path} ({e})")
                corrupted_files.append(rel_path)
                records.append(
                    ImageRecord(
                        rel_path=rel_path,
                        class_name=class_name,
                        width=0,
                        height=0,
                        channels=0,
                        mode="UNKNOWN",
                        file_size_bytes=file_size,
                        is_valid=False,
                        error=str(e),
                    )
                )

    total_valid = sum(class_counts.values())
    if total_valid == 0:
        raise ValueError(f"No valid images found in {data_path}")

    # Compute balanced class weights: N / (num_classes * class_count)
    num_classes = len(classes)
    class_summaries: dict[str, ClassSummary] = {}
    for c in classes:
        cnt = class_counts[c]
        pct = (cnt / total_valid) * 100.0 if total_valid > 0 else 0.0
        weight = (total_valid / (num_classes * cnt)) if cnt > 0 else 0.0
        class_summaries[c] = ClassSummary(
            class_name=c,
            count=cnt,
            percentage=round(pct, 2),
            class_weight=round(weight, 4),
        )

    counts = list(class_counts.values())
    imbalance_ratio = (max(counts) / min(counts)) if min(counts) > 0 else 0.0

    def _dist_stats(arr: list[float] | list[int]) -> dict[str, float]:
        if not arr:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0, "std": 0.0}
        np_arr = np.array(arr, dtype=float)
        return {
            "min": round(float(np.min(np_arr)), 2),
            "max": round(float(np.max(np_arr)), 2),
            "mean": round(float(np.mean(np_arr)), 2),
            "median": round(float(np.median(np_arr)), 2),
            "std": round(float(np.std(np_arr)), 2),
        }

    dataset_stats = DatasetStats(
        data_dir=str(data_path.as_posix()),
        classes=classes,
        total_images_scanned=len(records),
        valid_images_count=total_valid,
        corrupt_images_count=len(corrupted_files),
        class_summaries=class_summaries,
        imbalance_ratio=round(imbalance_ratio, 2),
        width_stats=_dist_stats(widths),
        height_stats=_dist_stats(heights),
        aspect_ratios=_dist_stats(aspect_ratios),
        file_size_kb_stats=_dist_stats(file_sizes_kb),
        color_modes=color_modes,
        corrupted_files=corrupted_files,
    )

    return records, dataset_stats


def generate_eda_plots(stats: DatasetStats, output_dir: Path | str = "reports") -> list[Path]:
    """Generate EDA visualization plots (class distribution and resolution scatter)."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    generated_plots: list[Path] = []

    # 1. Class Distribution Chart
    fig, ax1 = plt.subplots(figsize=(8, 5))
    classes = stats.classes
    counts = [stats.class_summaries[c].count for c in classes]
    weights = [stats.class_summaries[c].class_weight for c in classes]

    x = np.arange(len(classes))
    width = 0.45

    bars1 = ax1.bar(
        x, counts, width, label="Sample Count", color="#1976D2", edgecolor="#0D47A1", alpha=0.9
    )
    ax1.set_ylabel("Image Count", color="#0D47A1", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes, rotation=20, fontsize=10)
    ax1.set_title(
        f"TrashNet Class Distribution (Total: {stats.valid_images_count} images)",
        fontsize=12,
        fontweight="bold",
    )

    # Add count labels on top of bars
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 5,
            f"{int(yval)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Secondary axis for class weights
    ax2 = ax1.twinx()
    ax2.plot(x, weights, color="#D32F2F", marker="o", linewidth=2, label="Balanced Class Weight")
    ax2.set_ylabel("Balanced Loss Weight", color="#D32F2F", fontsize=11, fontweight="bold")
    ax2.grid(False)

    fig.tight_layout()
    dist_plot = out_path / "class_distribution.png"
    plt.savefig(dist_plot, dpi=150)
    plt.close(fig)
    generated_plots.append(dist_plot)

    # 2. Image Resolution & Aspect Ratio Distribution
    fig, (ax_res, ax_size) = plt.subplots(1, 2, figsize=(10, 4.5))

    ax_res.bar(
        ["Width", "Height"],
        [stats.width_stats["mean"], stats.height_stats["mean"]],
        yerr=[stats.width_stats["std"], stats.height_stats["std"]],
        capsize=5,
        color=["#388E3C", "#689F38"],
        alpha=0.85,
    )
    ax_res.set_ylabel("Pixels", fontsize=10)
    ax_res.set_title("Mean Dimensions (± Std)", fontsize=11, fontweight="bold")
    for idx, (_, val) in enumerate(
        [("Width", stats.width_stats["mean"]), ("Height", stats.height_stats["mean"])]
    ):
        ax_res.text(idx, val / 2, f"{int(val)} px", ha="center", color="white", fontweight="bold")

    # File size box representation
    ax_size.bar(
        ["Min", "Median", "Mean", "Max"],
        [
            stats.file_size_kb_stats["min"],
            stats.file_size_kb_stats["median"],
            stats.file_size_kb_stats["mean"],
            stats.file_size_kb_stats["max"],
        ],
        color="#7B1FA2",
        alpha=0.85,
    )
    ax_size.set_ylabel("File Size (KB)", fontsize=10)
    ax_size.set_title("File Size Metrics", fontsize=11, fontweight="bold")

    fig.tight_layout()
    res_plot = out_path / "image_resolution_distribution.png"
    plt.savefig(res_plot, dpi=150)
    plt.close(fig)
    generated_plots.append(res_plot)

    return generated_plots


def generate_eda_markdown(stats: DatasetStats, output_path: Path | str = "reports/eda_report.md"):
    """Generate comprehensive markdown summary report of the dataset."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# Exploratory Data Analysis (EDA) Report",
        "",
        "## 1. Dataset Overview",
        f"- **Directory**: `{stats.data_dir}`",
        f"- **Total Scanned Images**: {stats.total_images_scanned}",
        f"- **Valid Decodable Images**: {stats.valid_images_count}",
        f"- **Corrupt / Unreadable Images**: {stats.corrupt_images_count}",
        f"- **Imbalance Ratio (Max/Min Class)**: {stats.imbalance_ratio:.2f}",
        "",
        "## 2. Class Counts and Loss Weights",
        "| Class Name | Image Count | Percentage (%) | Balanced Class Weight |",
        "|---|---|---|---|",
    ]

    for c in stats.classes:
        sm = stats.class_summaries[c]
        lines.append(f"| `{c}` | {sm.count} | {sm.percentage:.2f}% | {sm.class_weight:.4f} |")

    lines.extend(
        [
            "",
            "## 3. Image Dimensions and Resolution",
            f"- **Width (px)**: Min={stats.width_stats['min']}, Mean={stats.width_stats['mean']}, Max={stats.width_stats['max']} (Median={stats.width_stats['median']})",
            f"- **Height (px)**: Min={stats.height_stats['min']}, Mean={stats.height_stats['mean']}, Max={stats.height_stats['max']} (Median={stats.height_stats['median']})",
            f"- **Aspect Ratio**: Mean={stats.aspect_ratios['mean']:.2f}, Median={stats.aspect_ratios['median']:.2f}",
            "",
            "## 4. File Attributes",
            f"- **Color Modes**: {json.dumps(stats.color_modes)}",
            f"- **File Size (KB)**: Min={stats.file_size_kb_stats['min']} KB, Mean={stats.file_size_kb_stats['mean']} KB, Max={stats.file_size_kb_stats['max']} KB",
            "",
            "## 5. Visualizations",
            "- Class Distribution: `reports/class_distribution.png`",
            "- Dimensions & Size: `reports/image_resolution_distribution.png`",
        ]
    )

    content = "\n".join(lines) + "\n"
    out_file.write_text(content, encoding="utf-8")
    return out_file


def run_eda(data_dir: Path | str = "data/raw", output_dir: Path | str = "reports") -> DatasetStats:
    """End-to-end EDA execution: scans data, generates plots, and writes JSON & Markdown reports."""
    logger.info(f"Running EDA on {data_dir}...")
    _, stats = inspect_dataset(data_dir)

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    # Save JSON summary
    summary_json = out_p / "eda_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(stats.to_dict(), f, indent=2)

    # Generate Plots
    plots = generate_eda_plots(stats, out_p)
    logger.info(f"Generated {len(plots)} plots in {out_p}")

    # Generate Markdown Report
    report_md = generate_eda_markdown(stats, out_p / "eda_report.md")
    logger.info(f"EDA report written to {report_md}")

    return stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run dataset discovery and EDA on raw dataset.")
    parser.add_argument("--data-dir", default="data/raw", help="Path to raw dataset directory.")
    parser.add_argument(
        "--output-dir", default="reports", help="Directory to save EDA reports and plots."
    )
    args = parser.parse_args()

    stats_result = run_eda(data_dir=args.data_dir, output_dir=args.output_dir)
    print("\n=== EDA SUMMARY ===")
    print(f"Location: {stats_result.data_dir}")
    print(f"Total Valid Images: {stats_result.valid_images_count}")
    print(f"Classes ({len(stats_result.classes)}): {', '.join(stats_result.classes)}")
    print("Class Counts & Weights:")
    for cls_name, cs in stats_result.class_summaries.items():
        print(
            f"  {cls_name:<12}: {cs.count:>4} images ({cs.percentage:>5.1f}%) | Weight: {cs.class_weight:.4f}"
        )
    print(f"Imbalance Ratio: {stats_result.imbalance_ratio}")
    print(
        f"Mean Resolution: {stats_result.width_stats['mean']} x {stats_result.height_stats['mean']}"
    )
    print("===================\n")
