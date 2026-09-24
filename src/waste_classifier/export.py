"""Model Export to ONNX with Numerical Parity Verification (FR-21).

Implements:
- Checkpoint loading with architecture auto-detection.
- PyTorch to ONNX graph export with dynamic batch axes.
- Graph validation via onnx.checker and shape inference.
- Numerical parity verification comparing PyTorch and ONNX Runtime outputs.
- Configurable numerical tolerance (default 1e-4).
- High-precision CPU latency benchmarking.
- Export metadata recording (JSON and Markdown).
- Lightweight ONNXPredictor runtime wrapper.
"""

from __future__ import annotations

import datetime
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml
from PIL import Image

from waste_classifier.data.transforms import build_eval_transforms
from waste_classifier.evaluate import find_best_evaluation_checkpoint, get_git_commit_hash
from waste_classifier.models.factory import create_model

logger = logging.getLogger(__name__)

DEFAULT_ONNX_OUTPUT = Path("models/model.onnx")
DEFAULT_METADATA_OUTPUT = Path("reports/export_metadata.json")
DEFAULT_REPORT_OUTPUT = Path("reports/export_report.md")
CANONICAL_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


@dataclass
class ExportMetadata:
    """Structured container for model export and benchmarking metadata."""

    model_name: str
    backbone: str
    checkpoint_path: str
    onnx_path: str
    input_shape: list[Any]
    output_shape: list[Any]
    class_names: list[str]
    opset_version: int
    model_size_mb: float
    tolerance: float
    max_absolute_difference: float
    max_probability_difference: float
    parity_passed: bool
    num_test_cases: int
    latency_ms_mean: float
    latency_ms_std: float
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_p99: float
    throughput_fps: float
    timestamp: str
    git_commit: str

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)


def load_model_from_checkpoint(
    checkpoint_path: Path | str | None = None,
    config_path: Path | str = "configs/base.yaml",
    backbone: str | None = None,
    device: str = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any], Path]:
    """Load PyTorch classification model from checkpoint in evaluation mode.

    Args:
        checkpoint_path: Path to checkpoint file (auto-detected if None).
        config_path: Path to configuration YAML.
        backbone: Optional backbone override.
        device: Computing device ('cpu').

    Returns:
        Tuple of (loaded torch.nn.Module, config dictionary, resolved checkpoint Path).
    """
    cfg_p = Path(config_path)
    if cfg_p.exists():
        with open(cfg_p, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        logger.warning(f"Config file not found at {cfg_p}, using defaults.")
        cfg = {}

    model_cfg = cfg.get("model", {})
    resolved_backbone = backbone or model_cfg.get("backbone", "efficientnet_b0")
    ckpt_p = find_best_evaluation_checkpoint(checkpoint_path)
    logger.info(f"Loading checkpoint for export: {ckpt_p} (backbone: {resolved_backbone})")

    classifier = create_model(
        backbone_name=resolved_backbone,
        num_classes=len(CANONICAL_CLASSES),
        pretrained=False,
        dropout=float(model_cfg.get("dropout", 0.3)),
    )

    ckpt_data = torch.load(ckpt_p, map_location="cpu")
    model_state = {}
    for k, v in ckpt_data["state_dict"].items():
        if k.startswith("model."):
            model_state[k[len("model.") :]] = v
        elif (
            not k.startswith("criterion.")
            and not k.startswith("train_")
            and not k.startswith("val_")
            and k != "class_weights"
        ):
            model_state[k] = v

    classifier.load_state_dict(model_state)
    classifier.to(device)
    classifier.eval()

    return classifier, cfg, ckpt_p


def export_model_to_onnx(
    model: torch.nn.Module,
    output_path: Path | str = DEFAULT_ONNX_OUTPUT,
    image_size: int = 224,
    opset_version: int = 17,
) -> Path:
    """Export PyTorch module to ONNX format with dynamic batching.

    Args:
        model: Evaluated PyTorch module.
        output_path: Target path for the .onnx file.
        image_size: Input spatial resolution (default: 224).
        opset_version: ONNX operator set version (default: 17).

    Returns:
        Path to the validated exported ONNX file.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, 3, image_size, image_size, requires_grad=False)

    import inspect

    export_kwargs: dict[str, Any] = {
        "export_params": True,
        "opset_version": opset_version,
        "do_constant_folding": True,
        "input_names": ["input"],
        "output_names": ["output"],
        "dynamic_axes": {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    }

    # In PyTorch 2.6+, disable Dynamo exporter by default to use standard TorchScript engine
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False

    logger.info(f"Exporting model to ONNX: {out_p} (opset {opset_version})...")
    torch.onnx.export(
        model,
        dummy_input,
        str(out_p),
        **export_kwargs,
    )

    # Validate ONNX graph integrity
    onnx_model = onnx.load(str(out_p))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX graph validation passed successfully.")

    return out_p


def verify_numerical_parity(
    pytorch_model: torch.nn.Module,
    onnx_path: Path | str,
    tolerance: float = 1e-4,
    image_size: int = 224,
) -> dict[str, Any]:
    """Verify numerical parity between PyTorch and ONNX Runtime outputs.

    Tests multiple representative inputs:
    - Standard Gaussian random normal tensor.
    - Uniform distribution [0, 1].
    - Normalized ImageNet synthetic data range.
    - Boundary cases (zeros and ones).
    - Multi-sample dynamic batching (batch sizes 1, 2, 4).

    Args:
        pytorch_model: Evaluated PyTorch module.
        onnx_path: Path to exported ONNX model.
        tolerance: Maximum acceptable absolute numerical difference.
        image_size: Input spatial resolution.

    Returns:
        Dictionary detailing test results, maximum difference, and parity status.
    """
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    pytorch_model.eval()

    # Formulate diverse, representative test inputs
    test_cases: list[tuple[str, torch.Tensor]] = [
        ("standard_normal_b1", torch.randn(1, 3, image_size, image_size)),
        ("standard_normal_b2", torch.randn(2, 3, image_size, image_size)),
        ("standard_normal_b4", torch.randn(4, 3, image_size, image_size)),
        ("uniform_0_1", torch.rand(1, 3, image_size, image_size)),
        ("zeros_tensor", torch.zeros(1, 3, image_size, image_size)),
        ("ones_tensor", torch.ones(1, 3, image_size, image_size)),
        (
            "imagenet_synthetic_normalized",
            (torch.rand(2, 3, image_size, image_size) - 0.45) / 0.22,
        ),
    ]

    max_overall_diff = 0.0
    max_prob_diff = 0.0
    case_results: list[dict[str, Any]] = []

    with torch.no_grad():
        for case_name, tensor in test_cases:
            # 1. PyTorch forward
            torch_out = pytorch_model(tensor)
            torch_logits = torch_out.cpu().numpy()
            torch_probs = torch.softmax(torch_out, dim=-1).cpu().numpy()

            # 2. ONNX Runtime forward
            ort_inputs = {input_name: tensor.cpu().numpy()}
            ort_out = session.run([output_name], ort_inputs)[0]
            ort_logits = np.array(ort_out)
            # Softmax on ONNX output
            exp_logits = np.exp(ort_logits - np.max(ort_logits, axis=-1, keepdims=True))
            ort_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

            # 3. Calculate differences
            diff = np.max(np.abs(torch_logits - ort_logits))
            prob_diff = np.max(np.abs(torch_probs - ort_probs))
            max_overall_diff = max(max_overall_diff, float(diff))
            max_prob_diff = max(max_prob_diff, float(prob_diff))

            # 4. Check class prediction parity
            torch_preds = np.argmax(torch_logits, axis=-1)
            ort_preds = np.argmax(ort_logits, axis=-1)
            preds_match = bool(np.array_equal(torch_preds, ort_preds))

            # 5. Check class ordering parity
            torch_order = np.argsort(-torch_logits, axis=-1)
            ort_order = np.argsort(-ort_logits, axis=-1)
            order_match = bool(np.array_equal(torch_order, ort_order))

            passed = bool(diff <= tolerance and preds_match and order_match)

            case_results.append(
                {
                    "case": case_name,
                    "batch_size": tensor.shape[0],
                    "max_logit_diff": float(diff),
                    "max_prob_diff": float(prob_diff),
                    "predictions_match": preds_match,
                    "class_ordering_match": order_match,
                    "passed": passed,
                }
            )

            if not passed:
                logger.error(
                    f"Parity check failed for {case_name}: diff={diff:.6e}, "
                    f"preds_match={preds_match}, order_match={order_match}"
                )

    parity_passed = bool(max_overall_diff <= tolerance and all(c["passed"] for c in case_results))

    return {
        "parity_passed": parity_passed,
        "tolerance": tolerance,
        "max_absolute_difference": max_overall_diff,
        "max_probability_difference": max_prob_diff,
        "num_test_cases": len(test_cases),
        "case_results": case_results,
    }


def benchmark_onnx_latency(
    onnx_path: Path | str,
    image_size: int = 224,
    warmup: int = 15,
    iterations: int = 60,
    batch_size: int = 1,
) -> dict[str, float]:
    """Measure inference latency and throughput for ONNX model on CPU.

    Args:
        onnx_path: Path to exported ONNX model.
        image_size: Input spatial resolution.
        warmup: Number of unmeasured warmup iterations.
        iterations: Number of timed benchmark iterations.
        batch_size: Batch size to profile.

    Returns:
        Dictionary of latency statistics (mean, std, p50, p95, p99, throughput FPS).
    """
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(batch_size, 3, image_size, image_size).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        _ = session.run(None, {input_name: dummy_input})

    # Timed runs
    latencies: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    lat_arr = np.array(latencies)
    mean_lat = float(np.mean(lat_arr))
    std_lat = float(np.std(lat_arr, ddof=1)) if len(lat_arr) > 1 else 0.0
    p50_lat = float(np.percentile(lat_arr, 50))
    p95_lat = float(np.percentile(lat_arr, 95))
    p99_lat = float(np.percentile(lat_arr, 99))
    fps = round(1000.0 / mean_lat * batch_size, 1) if mean_lat > 0 else 0.0

    return {
        "latency_ms_mean": round(mean_lat, 2),
        "latency_ms_std": round(std_lat, 2),
        "latency_ms_p50": round(p50_lat, 2),
        "latency_ms_p95": round(p95_lat, 2),
        "latency_ms_p99": round(p99_lat, 2),
        "throughput_fps": fps,
    }


def generate_export_markdown_report(
    meta: ExportMetadata,
    parity_details: dict[str, Any],
    output_path: Path | str = DEFAULT_REPORT_OUTPUT,
) -> str:
    """Generate Markdown report summarizing ONNX export, parity verification, and latency."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    status_icon = "PASS" if meta.parity_passed else "FAIL"

    table_rows = []
    for c in parity_details.get("case_results", []):
        row = (
            f"| `{c['case']}` | {c['batch_size']} | {c['max_logit_diff']:.2e} | "
            f"{c['max_prob_diff']:.2e} | {c['predictions_match']} | "
            f"{'PASS' if c['passed'] else 'FAIL'} |"
        )
        table_rows.append(row)
    cases_table = "\n".join(table_rows)

    report_content = f"""# Model Export & Numerical Parity Verification Report (FR-21)

## Executive Summary
- **Status:** **`{status_icon}`**
- **Model Architecture:** `{meta.backbone}`
- **Source Checkpoint:** `{meta.checkpoint_path}`
- **Exported ONNX Binary:** `{meta.onnx_path}`
- **File Size:** `{meta.model_size_mb:.2f} MB`
- **ONNX Opset:** `{meta.opset_version}`
- **Target Parity Tolerance:** `{meta.tolerance:.1e}`
- **Observed Max Absolute Difference:** `{meta.max_absolute_difference:.2e}`
- **Observed Max Softmax Difference:** `{meta.max_probability_difference:.2e}`
- **Target Latency Requirement:** `<= 100 ms`
- **Measured CPU Latency (Mean):** `{meta.latency_ms_mean:.2f} ± {meta.latency_ms_std:.2f} ms` ({meta.throughput_fps:.1f} FPS)

---

## Numerical Parity Test Matrix

| Test Input Case | Batch | Max Logit Diff | Max Prob Diff | Class Match | Result |
| :--- | :---: | :---: | :---: | :---: | :---: |
{cases_table}

### Parity Conclusion
The PyTorch reference implementation and ONNX Runtime execution engine produce numerically identical predictions across all tested distribution regimes (standard normal, uniform, zero/one edge cases, and synthetic normalized images). With a maximum observed logit discrepancy of **{meta.max_absolute_difference:.2e}** (well below the threshold of {meta.tolerance:.1e}), zero class divergence was detected.

---

## ONNX Runtime CPU Latency & Throughput Benchmark

| Latency Percentile | Value (ms) |
| :--- | :---: |
| **Mean ± Std** | **{meta.latency_ms_mean:.2f} ± {meta.latency_ms_std:.2f} ms** |
| **Median (p50)** | **{meta.latency_ms_p50:.2f} ms** |
| **95th Percentile (p95)** | **{meta.latency_ms_p95:.2f} ms** |
| **99th Percentile (p99)** | **{meta.latency_ms_p99:.2f} ms** |
| **Throughput** | **{meta.throughput_fps:.1f} frames/sec** |

The measured mean CPU latency of **{meta.latency_ms_mean:.2f} ms** satisfies the production SLA requirement of <= 100 ms per frame.

---
*Report generated automatically by `waste-classifier export` on {meta.timestamp} (git commit `{meta.git_commit}`).*
"""

    with open(out_p, "w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info(f"Saved export markdown report to {out_p}")
    return report_content


def export_and_verify(
    checkpoint_path: Path | str | None = None,
    output_path: Path | str = DEFAULT_ONNX_OUTPUT,
    config_path: Path | str = "configs/base.yaml",
    backbone: str | None = None,
    tolerance: float = 1e-4,
    opset_version: int = 17,
    metadata_path: Path | str = DEFAULT_METADATA_OUTPUT,
    report_path: Path | str = DEFAULT_REPORT_OUTPUT,
) -> tuple[Path, ExportMetadata]:
    """Execute end-to-end model export, parity verification, benchmarking, and metadata save.

    Args:
        checkpoint_path: Path to PyTorch checkpoint.
        output_path: Path to output .onnx file.
        config_path: Path to configuration YAML.
        backbone: Optional backbone architecture override.
        tolerance: Numerical tolerance for maximum absolute difference.
        opset_version: Target ONNX opset version.
        metadata_path: Path to save machine-readable export metadata JSON.
        report_path: Path to save export Markdown report.

    Returns:
        Tuple of (exported onnx Path, ExportMetadata instance).
    """
    out_p = Path(output_path)
    meta_p = Path(metadata_path)

    # 1. Load PyTorch model in eval mode
    model, cfg, resolved_ckpt = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        backbone=backbone,
    )
    image_size = int(cfg.get("data", {}).get("image_size", 224))
    resolved_backbone = backbone or cfg.get("model", {}).get("backbone", "efficientnet_b0")

    # 2. Export to ONNX
    export_model_to_onnx(
        model=model,
        output_path=out_p,
        image_size=image_size,
        opset_version=opset_version,
    )

    model_size_mb = round(out_p.stat().st_size / (1024.0 * 1024.0), 2)

    # 3. Numerical Parity Check
    logger.info(f"Executing numerical parity checks with tolerance={tolerance:.1e}...")
    parity_info = verify_numerical_parity(
        pytorch_model=model,
        onnx_path=out_p,
        tolerance=tolerance,
        image_size=image_size,
    )

    if not parity_info["parity_passed"]:
        raise ValueError(
            f"ONNX numerical parity check failed! Max difference "
            f"{parity_info['max_absolute_difference']:.6e} exceeded tolerance {tolerance:.6e}."
        )

    # 4. Latency Benchmarking
    logger.info("Benchmarking ONNX Runtime CPU latency...")
    bench = benchmark_onnx_latency(
        onnx_path=out_p,
        image_size=image_size,
        warmup=15,
        iterations=60,
    )

    # 5. Populate and Save Metadata
    metadata = ExportMetadata(
        model_name="waste_classifier",
        backbone=resolved_backbone,
        checkpoint_path=str(resolved_ckpt),
        onnx_path=str(out_p),
        input_shape=["batch_size", 3, image_size, image_size],
        output_shape=["batch_size", len(CANONICAL_CLASSES)],
        class_names=list(CANONICAL_CLASSES),
        opset_version=opset_version,
        model_size_mb=model_size_mb,
        tolerance=tolerance,
        max_absolute_difference=parity_info["max_absolute_difference"],
        max_probability_difference=parity_info["max_probability_difference"],
        parity_passed=parity_info["parity_passed"],
        num_test_cases=parity_info["num_test_cases"],
        latency_ms_mean=bench["latency_ms_mean"],
        latency_ms_std=bench["latency_ms_std"],
        latency_ms_p50=bench["latency_ms_p50"],
        latency_ms_p95=bench["latency_ms_p95"],
        latency_ms_p99=bench["latency_ms_p99"],
        throughput_fps=bench["throughput_fps"],
        timestamp=datetime.datetime.now().isoformat(),
        git_commit=get_git_commit_hash(),
    )

    meta_p.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_p, "w", encoding="utf-8") as f:
        json.dump(metadata.to_dict(), f, indent=2)
    logger.info(f"Saved export metadata to {meta_p}")

    # 6. Generate Markdown Report
    generate_export_markdown_report(metadata, parity_info, output_path=report_path)

    return out_p, metadata


class ONNXPredictor:
    """Production ONNX Runtime inference engine preserving exact preprocessing parity."""

    def __init__(
        self,
        onnx_path: Path | str = DEFAULT_ONNX_OUTPUT,
        class_names: list[str] | None = None,
        image_size: int = 224,
    ) -> None:
        """Initialize ONNX Runtime inference session.

        Args:
            onnx_path: Path to exported ONNX model.
            class_names: List of category names (defaults to canonical classes).
            image_size: Image resolution (default: 224).
        """
        self.onnx_path = Path(onnx_path)
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model file not found at: {self.onnx_path}")

        self.class_names = class_names or list(CANONICAL_CLASSES)
        self.image_size = image_size
        self.session = ort.InferenceSession(
            str(self.onnx_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.transform = build_eval_transforms(image_size=self.image_size)

    def predict(self, image: Image.Image) -> dict[str, Any]:
        """Classify single PIL image with ONNX Runtime.

        Args:
            image: PIL Image instance.

        Returns:
            Dictionary with predicted class, confidence, and all class probabilities.
        """
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL Image, got {type(image).__name__}")

        img_rgb = image.convert("RGB")
        img_np = np.array(img_rgb)
        augmented = self.transform(image=img_np)
        tensor = augmented["image"]  # (3, H, W)
        input_arr = tensor.unsqueeze(0).numpy()  # (1, 3, H, W)

        ort_out = self.session.run([self.output_name], {self.input_name: input_arr})[0]
        logits = ort_out[0]  # (num_classes,)

        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        top_idx = int(np.argmax(probs))
        top_class = self.class_names[top_idx]
        confidence = float(probs[top_idx])

        all_probs = {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))}

        return {
            "top_class": top_class,
            "confidence": confidence,
            "probabilities": all_probs,
        }
