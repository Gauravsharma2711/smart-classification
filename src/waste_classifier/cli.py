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
        Path,
        typer.Option(
            "--checkpoint",
            help="Path to model checkpoint.",
        ),
    ] = Path("checkpoints/phase1/last.ckpt"),
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            help="Path to YAML configuration file.",
        ),
    ] = Path("configs/base.yaml"),
) -> None:
    """Evaluate model checkpoint on held-out test set or field set."""
    console.print("[yellow]Evaluation module (FR-7/FR-8) will be executed.[/yellow]")
    raise typer.Exit(code=0)


@app.command("export")
def export_cmd(
    checkpoint: Annotated[
        Path,
        typer.Option(
            "--checkpoint",
            help="Path to model checkpoint.",
        ),
    ] = Path("checkpoints/phase1/last.ckpt"),
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path to output ONNX file.",
        ),
    ] = Path("models/model.onnx"),
) -> None:
    """Export PyTorch checkpoint to ONNX format."""
    console.print("[yellow]Export module (FR-21) will be executed.[/yellow]")
    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
