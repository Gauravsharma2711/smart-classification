"""Unit tests for Empirical Backbone Comparison & Multi-Seed Benchmarking (FR-18, FR-19)."""

from pathlib import Path

import numpy as np

from waste_classifier.benchmark import (
    aggregate_benchmark_statistics,
    generate_comparison_markdown_report,
    measure_model_specs,
    run_full_multi_seed_comparison,
)


def test_measure_model_specs():
    """Verify hardware and latency profiling for all 3 backbones."""
    for backbone in ["mobilenetv3_large_100", "efficientnet_b0", "resnet50"]:
        specs = measure_model_specs(
            backbone_name=backbone,
            num_classes=6,
            image_size=224,
            warmup=2,
            iterations=5,
            device="cpu",
        )

        assert specs["backbone"] == backbone
        assert specs["total_params"] > 0
        assert specs["trainable_params"] > 0
        assert specs["model_size_mb"] > 0
        assert specs["latency_ms_mean"] > 0
        assert specs["latency_ms_std"] >= 0
        assert specs["throughput_fps"] > 0

    # Architectural relative sizing sanity checks
    mob_specs = measure_model_specs("mobilenetv3_large_100", warmup=1, iterations=2)
    res_specs = measure_model_specs("resnet50", warmup=1, iterations=2)

    assert mob_specs["total_params"] < res_specs["total_params"]
    assert mob_specs["model_size_mb"] < res_specs["model_size_mb"]


def test_aggregate_benchmark_statistics():
    """Verify statistical aggregation (mean ± sample std) across runs."""
    dummy_runs = [
        {
            "backbone": "efficientnet_b0",
            "seed": 42,
            "test_macro_f1": 0.85,
            "test_acc": 0.86,
            "field_acc": 0.60,
            "field_f1": 0.58,
            "val_f1": 0.87,
        },
        {
            "backbone": "efficientnet_b0",
            "seed": 123,
            "test_macro_f1": 0.87,
            "test_acc": 0.88,
            "field_acc": 0.62,
            "field_f1": 0.60,
            "val_f1": 0.89,
        },
        {
            "backbone": "efficientnet_b0",
            "seed": 456,
            "test_macro_f1": 0.86,
            "test_acc": 0.87,
            "field_acc": 0.61,
            "field_f1": 0.59,
            "val_f1": 0.88,
        },
    ]

    dummy_specs = {
        "efficientnet_b0": {
            "total_params": 5300000,
            "model_size_mb": 21.2,
            "latency_ms_mean": 24.5,
            "latency_ms_std": 1.2,
            "throughput_fps": 40.8,
        }
    }

    agg = aggregate_benchmark_statistics(dummy_runs, dummy_specs)

    assert "efficientnet_b0" in agg
    data = agg["efficientnet_b0"]
    assert data["num_seeds"] == 3
    assert data["seeds_evaluated"] == [42, 123, 456]

    metrics = data["metrics"]
    expected_mean_f1 = round(float(np.mean([0.85, 0.87, 0.86])), 4)
    expected_std_f1 = round(float(np.std([0.85, 0.87, 0.86], ddof=1)), 4)

    assert metrics["test_macro_f1_mean"] == expected_mean_f1
    assert metrics["test_macro_f1_std"] == expected_std_f1


def test_generate_comparison_markdown_report(tmp_path: Path):
    """Verify Markdown report creation with formatted table."""
    report_file = tmp_path / "test_report.md"

    dummy_agg = {
        "mobilenetv3_large_100": {
            "specs": {
                "total_params": 4200000,
                "model_size_mb": 16.8,
                "latency_ms_mean": 18.2,
                "latency_ms_std": 1.1,
            },
            "metrics": {
                "test_macro_f1_mean": 0.84,
                "test_macro_f1_std": 0.01,
                "field_accuracy_mean": 0.62,
                "field_accuracy_std": 0.02,
            },
        },
        "resnet50": {
            "specs": {
                "total_params": 23500000,
                "model_size_mb": 94.0,
                "latency_ms_mean": 52.0,
                "latency_ms_std": 2.5,
            },
            "metrics": {
                "test_macro_f1_mean": 0.86,
                "test_macro_f1_std": 0.015,
                "field_accuracy_mean": 0.64,
                "field_accuracy_std": 0.02,
            },
        },
    }

    report_content = generate_comparison_markdown_report(dummy_agg, output_path=report_file)

    assert report_file.exists()
    assert "Backbone Comparison & Multi-Seed Benchmark Report" in report_content
    assert "mobilenetv3_large_100" in report_content
    assert "resnet50" in report_content
    assert "CPU Latency" in report_content
    assert "Test Macro-F1" in report_content


def test_smoke_multi_seed_comparison():
    """Verify smoke test execution of benchmark runner."""
    agg = run_full_multi_seed_comparison(
        backbones=["mobilenetv3_large_100"],
        seeds=[42],
        epochs=1,
        smoke_test=True,
    )

    assert "mobilenetv3_large_100" in agg
    res = agg["mobilenetv3_large_100"]
    assert res["num_seeds"] == 1
    assert "test_macro_f1_mean" in res["metrics"]
    assert "field_accuracy_mean" in res["metrics"]
