"""Command-line interface (CLI) for waste classification project.

Supports:
- Training (Phase 1 baseline, Phase 2 fine-tuning, smoke testing).
- Evaluation (Test-set and field-set metrics).
- Export (PyTorch to ONNX format).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from waste_classifier.train import train_phase1, train_phase2

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("waste_classifier.cli")

app = typer.Typer(
    name="waste-classifier",
    help="Smart Waste Classification CLI: training, evaluation, and model export.",
    add_completion=False,
)
console = Console()


@app.command("train")
def train_cmd(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to YAML configuration file.",
        ),
    ] = Path("configs/base.yaml"),
    phase: Annotated[
        int,
        typer.Option(
            "--phase",
            "-p",
            help="Training phase: 1 for frozen backbone transfer learning, 2 for fine-tuning.",
        ),
    ] = 1,
    epochs: Annotated[
        int | None,
        typer.Option(
            "--epochs",
            "-e",
            help="Override total training epochs.",
        ),
    ] = None,
    lr: Annotated[
        float | None,
        typer.Option(
            "--lr",
            help="Override learning rate (Phase 1).",
        ),
    ] = None,
    lr_backbone: Annotated[
        float | None,
        typer.Option(
            "--lr-backbone",
            help="Override fine-tuning learning rate for backbone (Phase 2).",
        ),
    ] = None,
    lr_head: Annotated[
        float | None,
        typer.Option(
            "--lr-head",
            help="Override fine-tuning learning rate for classifier head (Phase 2).",
        ),
    ] = None,
    unfreeze_fraction: Annotated[
        float | None,
        typer.Option(
            "--unfreeze-fraction",
            help="Override fraction of trailing backbone parameter tensors to unfreeze (Phase 2).",
        ),
    ] = None,
    phase1_checkpoint: Annotated[
        Path | None,
        typer.Option(
            "--phase1-checkpoint",
            help="Explicit path to Phase 1 checkpoint (defaults to best checkpoint).",
        ),
    ] = None,
    backbone: Annotated[
        str | None,
        typer.Option(
            "--backbone",
            "-b",
            help="Override model backbone architecture.",
        ),
    ] = None,
    batch_size: Annotated[
        int | None,
        typer.Option(
            "--batch-size",
            help="Override batch size.",
        ),
    ] = None,
    smoke_test: Annotated[
        bool,
        typer.Option(
            "--smoke-test",
            help="Run minimal 1-epoch smoke test on small batches for validation.",
        ),
    ] = False,
    seed: Annotated[
        int | None,
        typer.Option(
            "--seed",
            help="Override random seed for reproducibility.",
        ),
    ] = None,
) -> None:
    """Train waste classifier model according to configuration and phase."""
    console.print(
        f"[bold green]Initiating Waste Classifier Training (Phase {phase})...[/bold green]"
    )

    if phase == 1:
        _, results = train_phase1(
            config_path=config,
            epochs=epochs,
            lr=lr,
            backbone=backbone,
            batch_size=batch_size,
            smoke_test=smoke_test,
            seed=seed,
        )

        table = Table(
            title="Phase 1 Training Summary", show_header=True, header_style="bold magenta"
        )
        table.add_column("Metric", style="dim")
        table.add_column("Value", style="bold green")

        for k, v in results.items():
            table.add_row(k, str(v))

        console.print(table)

    elif phase == 2:
        _, results, comparison = train_phase2(
            config_path=config,
            phase1_checkpoint=phase1_checkpoint,
            epochs=epochs,
            lr_backbone=lr_backbone,
            lr_head=lr_head,
            unfreeze_fraction=unfreeze_fraction,
            batch_size=batch_size,
            smoke_test=smoke_test,
            seed=seed,
            backbone=backbone,
        )

        table = Table(
            title="Phase 2 Fine-Tuning Summary", show_header=True, header_style="bold magenta"
        )
        table.add_column("Metric", style="dim")
        table.add_column("Value", style="bold green")

        for k, v in results.items():
            table.add_row(k, str(v))

        console.print(table)

        # Print Objective Comparison Table between Phase 1 and Phase 2
        comp_table = Table(
            title="Phase 1 Baseline vs Phase 2 Fine-Tuning Comparison",
            show_header=True,
            header_style="bold cyan",
        )
        comp_table.add_column("Metric")
        comp_table.add_column("Phase 1 Baseline", style="yellow")
        comp_table.add_column("Phase 2 Fine-Tuning", style="green")
        comp_table.add_column("Delta (P2 - P1)", style="bold")

        comp_table.add_row(
            "Validation Loss",
            str(comparison["phase1_val_loss"]),
            str(comparison["phase2_val_loss"]),
            f"{comparison['delta_val_loss']:+.4f}",
        )
        comp_table.add_row(
            "Validation Accuracy",
            f"{comparison['phase1_val_acc'] * 100:.2f}%",
            f"{comparison['phase2_val_acc'] * 100:.2f}%",
            f"{comparison['delta_val_acc'] * 100:+.2f}%",
        )
        comp_table.add_row(
            "Validation Macro-F1",
            f"{comparison['phase1_val_f1'] * 100:.2f}%",
            f"{comparison['phase2_val_f1'] * 100:.2f}%",
            f"{comparison['delta_val_f1'] * 100:+.2f}%",
        )

        console.print(comp_table)

    else:
        raise typer.BadParameter(f"Phase {phase} training is not valid (must be 1 or 2).")


@app.command("evaluate")
def evaluate_cmd(
    checkpoint: Annotated[
        Path | None,
        typer.Option(
            "--checkpoint",
            help="Path to model checkpoint (defaults to best available).",
        ),
    ] = None,
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to YAML configuration file.",
        ),
    ] = Path("configs/base.yaml"),
    split: Annotated[
        str,
        typer.Option(
            "--split",
            "-s",
            help="Dataset split to evaluate ('test', 'val', or 'field').",
        ),
    ] = "test",
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir",
            help="Directory to save evaluation reports.",
        ),
    ] = Path("reports"),
) -> None:
    """Evaluate model checkpoint on held-out test set, validation set, or field set."""
    from waste_classifier.evaluate import evaluate_checkpoint

    console.print(f"[bold cyan]Running Model Evaluation on split '{split}'...[/bold cyan]")

    result = evaluate_checkpoint(
        checkpoint_path=checkpoint,
        config_path=config,
        split=split,
        output_dir=output_dir,
    )

    # 1. Overall Metrics Table
    ov_table = Table(
        title=f"Overall Evaluation Metrics ({split.capitalize()} Set)",
        show_header=True,
        header_style="bold magenta",
    )
    ov_table.add_column("Metric", style="dim")
    ov_table.add_column("Score", style="bold green")

    for k, v in result.overall.items():
        if isinstance(v, float):
            ov_table.add_row(
                k.replace("_", " ").title(),
                f"{v:.4f} ({v * 100:.2f}%)"
                if "f1" in k or "acc" in k or "rec" in k or "prec" in k
                else f"{v:.4f}",
            )
        else:
            ov_table.add_row(k.replace("_", " ").title(), str(v))

    console.print(ov_table)

    # 2. Per-Class Metrics Table
    pc_table = Table(
        title="Per-Class Performance Metrics",
        show_header=True,
        header_style="bold cyan",
    )
    pc_table.add_column("Class Name", style="bold")
    pc_table.add_column("Precision", justify="right")
    pc_table.add_column("Recall", justify="right")
    pc_table.add_column("F1-Score", justify="right")
    pc_table.add_column("Support", justify="right")

    for cls_name, metrics in result.per_class.items():
        pc_table.add_row(
            cls_name,
            f"{metrics['precision']:.4f}",
            f"{metrics['recall']:.4f}",
            f"{metrics['f1_score']:.4f}",
            str(metrics["support"]),
        )

    console.print(pc_table)

    # 3. Negative Sample Rejection Table (if available)
    if result.negative_rejection:
        neg_table = Table(
            title="Negative Image Uncertainty Rejection",
            show_header=True,
            header_style="bold yellow",
        )
        neg_table.add_column("Metric", style="dim")
        neg_table.add_column("Value", style="bold yellow")
        neg = result.negative_rejection
        neg_table.add_row("Total Negatives Tested", str(neg.get("total_negatives", 0)))
        neg_table.add_row("Confidence Threshold", f"{neg.get('confidence_threshold', 0.60):.2f}")
        neg_table.add_row("Samples Rejected / Flagged Uncertain", str(neg.get("rejected_count", 0)))
        neg_table.add_row("Rejection Rate", f"{float(neg.get('rejection_rate', 0.0)) * 100:.2f}%")
        neg_table.add_row(
            "Mean Negative Confidence", f"{float(neg.get('mean_confidence', 0.0)) * 100:.2f}%"
        )
        console.print(neg_table)

    console.print(f"[bold green]Artifacts written to: {output_dir}/[/bold green]")
    console.print(f"  - Metrics JSON: {output_dir / f'{split}_metrics.json'}")
    if split == "field":
        console.print(f"  - Confusion Matrix: {output_dir / 'field_confusion_matrix.png'}")
        console.print(
            f"  - Classification Report: {output_dir / 'field_classification_report.txt'}"
        )
        console.print(f"  - Error Analysis: {output_dir / 'field_error_analysis.md'}")
    else:
        console.print(f"  - Confusion Matrix: {output_dir / 'confusion_matrix.png'}")
        console.print(f"  - Classification Report: {output_dir / 'classification_report.txt'}")


@app.command("predict")
def predict_cmd(
    image_path: Annotated[
        Path,
        typer.Argument(
            help="Path to image file for classification.",
        ),
    ],
    checkpoint: Annotated[
        Path | None,
        typer.Option(
            "--checkpoint",
            help="Path to model checkpoint (defaults to best available).",
        ),
    ] = None,
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to YAML configuration file.",
        ),
    ] = Path("configs/base.yaml"),
    top_k: Annotated[
        int | None,
        typer.Option(
            "--top-k",
            "-k",
            help="Number of top candidate classes to display.",
        ),
    ] = None,
    threshold: Annotated[
        float | None,
        typer.Option(
            "--threshold",
            "-t",
            help="Confidence threshold for certainty gating.",
        ),
    ] = None,
) -> None:
    """Classify a single image and display predictions and bin recommendations."""
    from PIL import Image

    from waste_classifier.inference import predict

    if not image_path.exists():
        console.print(f"[bold red]Error: Image not found at {image_path}[/bold red]")
        raise typer.Exit(code=1)

    try:
        with Image.open(image_path) as pil_img:
            img = pil_img.copy()
    except Exception as err:
        console.print(f"[bold red]Error: Could not open image {image_path}: {err}[/bold red]")
        raise typer.Exit(code=1) from err

    console.print(f"[bold cyan]Running Inference on:[/bold cyan] {image_path}")

    res = predict(
        image=img,
        checkpoint_path=checkpoint,
        config_path=config,
        top_k=top_k,
        confidence_threshold=threshold,
    )

    status_color = "green" if res.is_confident else "yellow"
    console.print(
        f"\n[bold {status_color}]Result: {res.top_class.upper()} "
        f"({res.confidence * 100:.2f}% confidence)[/bold {status_color}]"
    )
    console.print(f"[dim]{res.guidance}[/dim]\n")

    # Table of top-k candidates
    table = Table(
        title=f"Top-{len(res.predictions)} Predictions",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Rank", style="dim", justify="right")
    table.add_column("Class", style="bold")
    table.add_column("Probability", justify="right")
    table.add_column("Recommended Bin")

    for rank, item in enumerate(res.predictions, start=1):
        prob_str = f"{item.probability * 100:.2f}%"
        table.add_row(
            str(rank),
            item.class_name,
            prob_str,
            f"[{item.bin_color}]{item.bin_label}[/{item.bin_color}]",
        )

    console.print(table)

    # Bin Advice Card
    bin_table = Table(
        title="Disposal Recommendation",
        show_header=False,
        border_style="cyan",
    )
    bin_table.add_column("Property", style="bold cyan")
    bin_table.add_column("Details")

    bin_table.add_row("Action Bin", f"[{res.bin_color}]{res.bin_name}[/{res.bin_color}]")
    bin_table.add_row("Bin Label", f"[{res.bin_color}]{res.bin_label}[/{res.bin_color}]")
    bin_table.add_row("Instructions", res.bin_instructions)

    console.print(bin_table)


def _get_launch_app():
    """Dynamically import launch_app ensuring sys.path contains project root and app dir."""
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent.parent
    cwd = Path.cwd()

    for p in [cwd, project_root, project_root / "app", cwd / "app"]:
        p_str = str(p.resolve())
        if p.exists() and p_str not in sys.path:
            sys.path.insert(0, p_str)

    try:
        from app.app import launch_app

        return launch_app
    except ModuleNotFoundError:
        try:
            from app import launch_app

            return launch_app
        except ModuleNotFoundError as err:
            raise ImportError(
                f"Could not import launch_app from app.app or app. sys.path={sys.path}"
            ) from err


@app.command("app")
def app_cmd(
    host: Annotated[
        str,
        typer.Option(
            "--host",
            help="Server host address to bind.",
        ),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option(
            "--port",
            "-p",
            help="Port to run the Gradio application.",
        ),
    ] = 7860,
    share: Annotated[
        bool,
        typer.Option(
            "--share",
            help="Create a public Gradio share link.",
        ),
    ] = False,
) -> None:
    """Launch the Gradio web application for waste classification and bin recommendations."""
    launch_app = _get_launch_app()

    console.print(
        f"[bold cyan]Launching Smart Waste Classifier on http://{host}:{port}...[/bold cyan]"
    )
    launch_app(server_name=host, server_port=port, share=share)


@app.command("serve")
def serve_cmd(
    host: Annotated[
        str,
        typer.Option(
            "--host",
            help="Server host address to bind.",
        ),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option(
            "--port",
            "-p",
            help="Port to run the Gradio application.",
        ),
    ] = 7860,
    share: Annotated[
        bool,
        typer.Option(
            "--share",
            help="Create a public Gradio share link.",
        ),
    ] = False,
) -> None:
    """Alias for 'app': Launch the Gradio web application for waste classification."""
    app_cmd(host=host, port=port, share=share)


@app.command("deploy")
def deploy_cmd(
    host: Annotated[
        str,
        typer.Option(
            "--host",
            help="Server host address to bind (0.0.0.0 for external reachability).",
        ),
    ] = "0.0.0.0",
    port: Annotated[
        int,
        typer.Option(
            "--port",
            "-p",
            help="Port to run the Gradio application.",
        ),
    ] = 7860,
    share: Annotated[
        bool,
        typer.Option(
            "--share/--no-share",
            help="Generate a secure public HTTPS link via Gradio tunnel (required for phone camera access).",
        ),
    ] = True,
) -> None:
    """Deploy the Gradio demo with HTTPS sharing for mobile and remote browser access (FR-20)."""
    from app.app import launch_app

    deploy_panel = Panel(
        f"[bold green]Starting Smart Waste Classifier HTTPS Deployment (FR-20)[/bold green]\n\n"
        f"• [bold]Local Binding:[/bold] http://{host}:{port}\n"
        f"• [bold]HTTPS Public Tunnel:[/bold] {'Enabled (gradio.live)' if share else 'Disabled'}\n"
        f"• [bold]Camera HTTPS Rule:[/bold] Mobile browsers strictly enforce HTTPS for camera access.\n"
        f"• [bold]Privacy Guarantee:[/bold] Strictly in-memory ephemeral processing (zero disk persistence).\n\n"
        f"[cyan]Testing on phone:[/cyan] Open the generated HTTPS URL on your phone browser (Safari/Chrome).\n"
        f"Allow camera access when prompted. Center waste items in frame.",
        title="♻️ Smart Waste Classifier Deployment",
        border_style="green",
    )
    console.print(deploy_panel)

    launch_app(server_name=host, server_port=port, share=share)


@app.command("explain")
def explain_cmd(
    image: Annotated[
        Path | None,
        typer.Option(
            "--image",
            "-i",
            help="Path to input image file to explain.",
        ),
    ] = None,
    target_class: Annotated[
        str | None,
        typer.Option(
            "--target-class",
            "-t",
            help="Specific class to explain (defaults to model top prediction).",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output",
            "-o",
            help="Path to save Grad-CAM overlay image.",
        ),
    ] = None,
    generate_report: Annotated[
        bool,
        typer.Option(
            "--generate-report",
            help="Generate full Grad-CAM figures and README report under reports/gradcam/.",
        ),
    ] = False,
) -> None:
    """Generate on-demand Grad-CAM visual explanation for a waste photo (FR-17)."""
    from PIL import Image

    from waste_classifier.explainability import GradCAMExplainer, generate_gradcam_report_samples

    if generate_report:
        console.print(
            "[bold cyan]Generating Grad-CAM report figures in reports/gradcam/...[/bold cyan]"
        )
        saved = generate_gradcam_report_samples()
        console.print(
            f"[bold green]Successfully generated {len(saved)} figures in reports/gradcam/[/bold green]"
        )
        return

    if image is None:
        console.print("[bold red]Error: Please specify --image or --generate-report.[/bold red]")
        raise typer.Exit(code=1)

    if not image.exists():
        console.print(f"[bold red]Error: Image file not found at {image}[/bold red]")
        raise typer.Exit(code=1)

    pil_img = Image.open(image)
    explainer = GradCAMExplainer()
    res = explainer.explain(pil_img, target_class=target_class)

    console.print(
        f"[bold green]Predicted Category:[/] {res.predicted_class} ({res.predicted_prob * 100:.1f}%)"
    )
    console.print(f"[bold cyan]Explained Target:[/] {res.target_class}")
    console.print(f"[dim]{res.disclaimer}[/dim]")

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        res.overlay_image.save(output)
        console.print(f"[bold green]Saved overlay to {output}[/bold green]")


@app.command("benchmark")
def benchmark_cmd(
    backbones: Annotated[
        list[str] | None,
        typer.Option(
            "--backbone",
            "-b",
            help="Backbones to compare (can specify multiple times). Defaults to efficientnet_b0, mobilenetv3_large_100, resnet50.",
        ),
    ] = None,
    seeds: Annotated[
        list[int] | None,
        typer.Option(
            "--seed",
            "-s",
            help="Random seeds to evaluate (can specify multiple times). Defaults to 42, 123, 456.",
        ),
    ] = None,
    epochs: Annotated[
        int,
        typer.Option(
            "--epochs",
            "-e",
            help="Number of epochs per training run.",
        ),
    ] = 3,
    batch_size: Annotated[
        int,
        typer.Option(
            "--batch-size",
            help="Batch size.",
        ),
    ] = 64,
    smoke_test: Annotated[
        bool,
        typer.Option(
            "--smoke-test",
            help="Fast smoke test on a small synthetic subset.",
        ),
    ] = False,
) -> None:
    """Run empirical backbone comparison across multiple random seeds (FR-18, FR-19)."""
    from waste_classifier.benchmark import run_full_multi_seed_comparison

    console.print(
        "[bold cyan]Initiating Multi-Backbone Multi-Seed Benchmark Routine (FR-18 & FR-19)...[/bold cyan]"
    )

    aggregated = run_full_multi_seed_comparison(
        backbones=backbones,
        seeds=seeds,
        epochs=epochs,
        batch_size=batch_size,
        smoke_test=smoke_test,
    )

    table = Table(
        title="Empirical Backbone Comparison Summary (3 Seeds)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Architecture", style="bold cyan")
    table.add_column("Parameters", justify="right")
    table.add_column("Model Size", justify="right")
    table.add_column("CPU Latency", justify="right")
    table.add_column("Test Macro-F1", justify="right", style="green")
    table.add_column("Field-Set Accuracy", justify="right", style="yellow")

    for b_name, data in aggregated.items():
        specs = data["specs"]
        metrics = data["metrics"]
        table.add_row(
            b_name,
            f"{specs['total_params'] / 1e6:.1f}M",
            f"{specs['model_size_mb']:.1f} MB",
            f"{specs['latency_ms_mean']:.1f} ± {specs['latency_ms_std']:.1f} ms",
            f"{metrics['test_macro_f1_mean'] * 100:.2f}% ± {metrics['test_macro_f1_std'] * 100:.2f}%",
            f"{metrics['field_accuracy_mean'] * 100:.2f}% ± {metrics['field_accuracy_std'] * 100:.2f}%",
        )

    console.print(table)
    console.print("[bold green]Benchmark complete! Artifacts written to reports/[/bold green]")


@app.command("export")
def export_cmd(
    checkpoint: Annotated[
        Path | None,
        typer.Option(
            "--checkpoint",
            help="Path to model checkpoint (defaults to best available).",
        ),
    ] = None,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path to output ONNX file.",
        ),
    ] = Path("models/model.onnx"),
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to configuration YAML file.",
        ),
    ] = Path("configs/base.yaml"),
    backbone: Annotated[
        str | None,
        typer.Option(
            "--backbone",
            "-b",
            help="Backbone architecture override.",
        ),
    ] = None,
    tolerance: Annotated[
        float,
        typer.Option(
            "--tolerance",
            "-t",
            help="Maximum allowable absolute numerical difference for parity.",
        ),
    ] = 1e-4,
    opset: Annotated[
        int,
        typer.Option(
            "--opset",
            help="ONNX operator set version.",
        ),
    ] = 17,
) -> None:
    """Export PyTorch checkpoint to ONNX format with numerical parity verification (FR-21)."""
    from waste_classifier.export import export_and_verify

    console.print(
        "[bold cyan]Initiating Model Export to ONNX with Numerical Parity Check (FR-21)...[/bold cyan]"
    )

    try:
        onnx_file, meta = export_and_verify(
            checkpoint_path=checkpoint,
            output_path=output,
            config_path=config,
            backbone=backbone,
            tolerance=tolerance,
            opset_version=opset,
        )
    except Exception as err:
        console.print(f"[bold red]Export Failed: {err}[/bold red]")
        raise typer.Exit(code=1) from err

    # 1. Parity Table
    p_table = Table(
        title="ONNX Export & Numerical Parity Verification Summary",
        show_header=True,
        header_style="bold magenta",
    )
    p_table.add_column("Property", style="dim")
    p_table.add_column("Value", style="bold")

    status_str = (
        "[bold green]PASSED[/bold green]" if meta.parity_passed else "[bold red]FAILED[/bold red]"
    )
    p_table.add_row("Parity Status", status_str)
    p_table.add_row("Backbone", meta.backbone)
    p_table.add_row("Source Checkpoint", meta.checkpoint_path)
    p_table.add_row("Exported ONNX Binary", str(onnx_file))
    p_table.add_row("Model Size", f"{meta.model_size_mb:.2f} MB")
    p_table.add_row("ONNX Opset", str(meta.opset_version))
    p_table.add_row("Configured Tolerance", f"{meta.tolerance:.1e}")
    p_table.add_row("Observed Max Difference", f"{meta.max_absolute_difference:.2e}")
    p_table.add_row("Test Cases Verified", str(meta.num_test_cases))

    console.print(p_table)

    # 2. Latency Table
    l_table = Table(
        title="ONNX Runtime CPU Inference Latency Benchmark",
        show_header=True,
        header_style="bold cyan",
    )
    l_table.add_column("Metric", style="dim")
    l_table.add_column("Measurement", style="bold green", justify="right")

    sla_met = meta.latency_ms_mean <= 100.0
    sla_str = (
        "[bold green]PASSED (<= 100 ms)[/bold green]"
        if sla_met
        else "[bold yellow]MARGINAL (> 100 ms)[/bold yellow]"
    )

    l_table.add_row(
        "Latency Mean ± Std", f"{meta.latency_ms_mean:.2f} ± {meta.latency_ms_std:.2f} ms"
    )
    l_table.add_row("Latency Median (p50)", f"{meta.latency_ms_p50:.2f} ms")
    l_table.add_row("Latency 95th Percentile (p95)", f"{meta.latency_ms_p95:.2f} ms")
    l_table.add_row("Latency 99th Percentile (p99)", f"{meta.latency_ms_p99:.2f} ms")
    l_table.add_row("Inference Throughput", f"{meta.throughput_fps:.1f} FPS")
    l_table.add_row("Production Latency SLA", sla_str)

    console.print(l_table)

    console.print(
        "[bold green]Export complete! Metadata and report written to reports/[/bold green]"
    )


@app.command("experiments")
def experiments_cmd(
    inspect_run: Annotated[
        Path | None,
        typer.Option(
            "--inspect",
            "-i",
            help="Path to an experiment directory or metadata.json to inspect.",
        ),
    ] = None,
    reproduce_run: Annotated[
        Path | None,
        typer.Option(
            "--reproduce",
            "-r",
            help="Path to an experiment run to recover configuration from.",
        ),
    ] = None,
    output_config: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path to output recovered YAML configuration.",
        ),
    ] = Path("configs/reproduced.yaml"),
) -> None:
    """Inspect and manage tracked experiments and verify reproducibility (FR-22)."""
    from waste_classifier.tracking import (
        list_tracked_experiments,
        load_experiment_metadata,
        reproduce_experiment_config,
        verify_experiment_integrity,
    )

    if inspect_run:
        console.print(f"[bold cyan]Inspecting Experiment Metadata from:[/] {inspect_run}")
        meta = load_experiment_metadata(inspect_run)
        valid, issues = verify_experiment_integrity(meta)

        table = Table(
            title="Experiment Provenance & Reproducibility Audit",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Mandatory Requirement", style="bold")
        table.add_column("Recorded Value", style="cyan")
        table.add_column("Audit Status", justify="center")

        table.add_row(
            "1. Configuration",
            f"{meta.get('config_name')} ({len(meta.get('configuration', {}))} keys)",
            "[green]PASS[/green]" if meta.get("configuration") else "[red]FAIL[/red]",
        )
        table.add_row(
            "2. Random Seed",
            str(meta.get("seed")),
            "[green]PASS[/green]" if isinstance(meta.get("seed"), int) else "[red]FAIL[/red]",
        )
        table.add_row(
            "3. Metrics",
            f"val_f1={meta.get('metrics', {}).get('val_f1', 0.0):.4f}",
            "[green]PASS[/green]" if meta.get("metrics") else "[red]FAIL[/red]",
        )
        table.add_row(
            "4. Model Backbone",
            str(meta.get("backbone")),
            "[green]PASS[/green]" if meta.get("backbone") else "[red]FAIL[/red]",
        )
        table.add_row(
            "5. Git Commit",
            str(meta.get("git_commit")),
            "[green]PASS[/green]" if meta.get("git_commit") else "[red]FAIL[/red]",
        )
        table.add_row(
            "6. Training Phase",
            f"Phase {meta.get('phase')}",
            "[green]PASS[/green]" if meta.get("phase") in (1, 2) else "[red]FAIL[/red]",
        )
        table.add_row(
            "7. Timestamp",
            str(meta.get("timestamp")),
            "[green]PASS[/green]" if meta.get("timestamp") else "[red]FAIL[/red]",
        )
        table.add_row(
            "8. Dataset Split",
            f"{meta.get('dataset_split')} (cksum: {meta.get('dataset_split_checksum', 'N/A')})",
            "[green]PASS[/green]" if meta.get("dataset_split") else "[red]FAIL[/red]",
        )

        console.print(table)
        if valid:
            console.print(
                "[bold green]All 8 provenance criteria satisfied! Experiment is 100% reproducible.[/bold green]"
            )
        else:
            console.print("[bold red]Audit issues detected:[/bold red]")
            for issue in issues:
                console.print(f"  - [red]{issue}[/red]")
        return

    if reproduce_run:
        console.print(f"[bold cyan]Recovering Configuration from:[/] {reproduce_run}")
        recovered = reproduce_experiment_config(reproduce_run, output_path=output_config)
        console.print(
            f"[bold green]Successfully recovered configuration into {output_config}![/bold green]"
        )
        console.print(f"[dim]Top-level config sections: {list(recovered.keys())}[/dim]")
        return

    # Default: List all tracked experiments
    runs = list_tracked_experiments()
    table = Table(
        title="Tracked Machine Learning Experiments (FR-22 Provenance Log)",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Timestamp", style="dim")
    table.add_column("Phase", justify="center")
    table.add_column("Backbone", style="bold cyan")
    table.add_column("Seed", justify="right")
    table.add_column("Dataset Split")
    table.add_column("Git Commit", style="yellow")
    table.add_column("Val F1", justify="right", style="green")
    table.add_column("Status", justify="center")

    for r in runs:
        val_f1 = (
            r.get("metrics", {}).get("val_f1")
            if isinstance(r.get("metrics"), dict)
            else r.get("val_f1")
        )
        f1_str = f"{float(val_f1) * 100:.2f}%" if val_f1 is not None else "N/A"
        status = r.get("status", "COMPLETED")
        status_color = (
            "green" if status == "COMPLETED" else ("red" if status == "FAILED" else "yellow")
        )

        table.add_row(
            str(r.get("timestamp", ""))[:19],
            f"P{r.get('phase', '')}",
            str(r.get("backbone", "")),
            str(r.get("seed", "")),
            str(r.get("dataset_split", "data/splits.csv")),
            str(r.get("git_commit", "")),
            f1_str,
            f"[{status_color}]{status}[/{status_color}]",
        )

    console.print(table)


if __name__ == "__main__":
    app()
