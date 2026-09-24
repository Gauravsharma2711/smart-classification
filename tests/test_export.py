"""Unit tests for Model Export to ONNX and Numerical Parity Checks (FR-21)."""

from pathlib import Path

import pytest
from PIL import Image

from waste_classifier.export import (
    ONNXPredictor,
    benchmark_onnx_latency,
    export_and_verify,
    export_model_to_onnx,
    verify_numerical_parity,
)
from waste_classifier.models.factory import create_model


@pytest.fixture
def dummy_pytorch_model():
    """Instantiate a lightweight PyTorch model for testing."""
    model = create_model("mobilenetv3_large_100", num_classes=6, pretrained=False)
    model.eval()
    return model


@pytest.fixture
def exported_onnx_path(tmp_path: Path, dummy_pytorch_model):
    """Export dummy model to a temporary ONNX file."""
    onnx_file = tmp_path / "test_model.onnx"
    export_model_to_onnx(dummy_pytorch_model, output_path=onnx_file, image_size=224)
    return onnx_file


def test_export_model_to_onnx(exported_onnx_path: Path):
    """Verify that export creates a valid non-empty ONNX file."""
    assert exported_onnx_path.exists()
    assert exported_onnx_path.stat().st_size > 1_000_000  # > 1 MB


def test_verify_numerical_parity_success(dummy_pytorch_model, exported_onnx_path: Path):
    """Verify that numerical parity passes under standard tolerance (1e-4)."""
    res = verify_numerical_parity(
        pytorch_model=dummy_pytorch_model,
        onnx_path=exported_onnx_path,
        tolerance=1e-4,
        image_size=224,
    )

    assert res["parity_passed"] is True
    assert res["max_absolute_difference"] < 1e-4
    assert res["max_probability_difference"] < 1e-4
    assert res["num_test_cases"] >= 5
    for case in res["case_results"]:
        assert case["passed"] is True
        assert case["predictions_match"] is True
        assert case["class_ordering_match"] is True


def test_verify_numerical_parity_failure_detection(dummy_pytorch_model, exported_onnx_path: Path):
    """Verify that parity check fails loudly if tolerance is impossibly strict."""
    # Under an impossibly tiny tolerance (e.g. 1e-12), floating point discrepancies between engines fail
    res = verify_numerical_parity(
        pytorch_model=dummy_pytorch_model,
        onnx_path=exported_onnx_path,
        tolerance=1e-12,
        image_size=224,
    )

    assert res["parity_passed"] is False


def test_benchmark_onnx_latency(exported_onnx_path: Path):
    """Verify ONNX Runtime latency benchmarking returns complete statistics."""
    stats = benchmark_onnx_latency(
        onnx_path=exported_onnx_path,
        image_size=224,
        warmup=2,
        iterations=5,
        batch_size=1,
    )

    assert "latency_ms_mean" in stats
    assert "latency_ms_std" in stats
    assert "latency_ms_p50" in stats
    assert "latency_ms_p95" in stats
    assert "latency_ms_p99" in stats
    assert "throughput_fps" in stats

    assert stats["latency_ms_mean"] > 0
    assert stats["throughput_fps"] > 0


def test_onnx_predictor_inference(exported_onnx_path: Path):
    """Verify ONNXPredictor executes inference on PIL image."""
    predictor = ONNXPredictor(onnx_path=exported_onnx_path)

    # Test valid image
    img = Image.new("RGB", (300, 300), color=(128, 64, 32))
    pred = predictor.predict(img)

    assert "top_class" in pred
    assert "confidence" in pred
    assert "probabilities" in pred

    assert pred["top_class"] in predictor.class_names
    assert 0.0 <= pred["confidence"] <= 1.0
    assert len(pred["probabilities"]) == 6
    assert pytest.approx(sum(pred["probabilities"].values()), rel=1e-3) == 1.0

    # Test invalid input type
    with pytest.raises(TypeError):
        predictor.predict("not_an_image")  # type: ignore


def test_export_and_verify_e2e(tmp_path: Path):
    """Verify complete end-to-end export routine writing artifacts."""
    out_onnx = tmp_path / "e2e_model.onnx"
    out_meta = tmp_path / "e2e_meta.json"
    out_report = tmp_path / "e2e_report.md"

    onnx_file, meta = export_and_verify(
        output_path=out_onnx,
        tolerance=1e-4,
        metadata_path=out_meta,
        report_path=out_report,
    )

    assert onnx_file.exists()
    assert out_meta.exists()
    assert out_report.exists()
    assert meta.parity_passed is True
    assert meta.max_absolute_difference < 1e-4
    assert meta.latency_ms_mean > 0
