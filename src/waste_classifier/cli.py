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

from waste_classifier.train import train_phase1

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
            help="Override learning rate.",
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
    else:
        raise typer.BadParameter(f"Phase {phase} training is not yet supported in this version.")


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
