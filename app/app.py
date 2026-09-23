"""Gradio Web Application for Smart Waste Classification.

Implements:
- FR-10: Image Upload UI with instant classification.
- FR-11: Camera Snapshot UI with webcam capture and graceful fallback.
- FR-12: Smart Bin Recommendation with accessible text labels and instructions.
- FR-13: Throttled Live Camera Mode (~4 FPS) with continuous real-time updates.
- FR-14: Frame smoothing and stability confirmation (K-frame stability).
- FR-15: Quality gating (detect blur "Hold steady" and low light "Too dark").
- FR-16: Uncertainty and no-item handling ("Point the camera at an item").

Architecture rules:
- Strictly decoupled: zero ML model / PyTorch forward-pass logic in the UI layer.
- Strictly in-memory: images and video frames are never saved to disk.
- Accessibility compliance: every color indicator is accompanied by an explicit text label.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
from PIL import Image

from waste_classifier.explainability import GradCAMExplainer
from waste_classifier.inference import CANONICAL_CLASSES, PredictionResult, get_predictor
from waste_classifier.live import LiveCameraPipeline, LiveFrameResult

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = Path("reports")
RAW_DATA_DIR = Path("data/raw")

_DEFAULT_LIVE_PIPELINE: LiveCameraPipeline | None = None


def get_live_pipeline() -> LiveCameraPipeline:
    """Retrieve or initialize the cached LiveCameraPipeline instance."""
    global _DEFAULT_LIVE_PIPELINE
    if _DEFAULT_LIVE_PIPELINE is None:
        _DEFAULT_LIVE_PIPELINE = LiveCameraPipeline()
    return _DEFAULT_LIVE_PIPELINE


def load_report_summary() -> dict[str, Any]:
    """Load test and field evaluation summary metrics if available."""
    summary: dict[str, Any] = {
        "test_accuracy": "83.16%",
        "test_macro_f1": "80.90%",
        "field_accuracy": "60.00%",
        "field_macro_f1": "58.40%",
        "negative_rejection": "86.67%",
    }

    test_file = REPORTS_DIR / "test_metrics.json"
    if test_file.exists():
        try:
            with open(test_file, encoding="utf-8") as f:
                data = json.load(f)
                ov = data.get("overall", {})
                if "accuracy" in ov:
                    summary["test_accuracy"] = f"{float(ov['accuracy']) * 100:.2f}%"
                if "macro_f1" in ov:
                    summary["test_macro_f1"] = f"{float(ov['macro_f1']) * 100:.2f}%"
        except Exception as err:
            logger.warning(f"Could not read test metrics: {err}")

    field_file = REPORTS_DIR / "field_metrics.json"
    if field_file.exists():
        try:
            with open(field_file, encoding="utf-8") as f:
                fdata = json.load(f)
                fov = fdata.get("overall", {})
                if "accuracy" in fov:
                    summary["field_accuracy"] = f"{float(fov['accuracy']) * 100:.2f}%"
                if "macro_f1" in fov:
                    summary["field_macro_f1"] = f"{float(fov['macro_f1']) * 100:.2f}%"
                nr = fdata.get("negative_rejection", {})
                if "rejection_rate" in nr:
                    summary["negative_rejection"] = f"{float(nr['rejection_rate']) * 100:.2f}%"
        except Exception as err:
            logger.warning(f"Could not read field metrics: {err}")

    return summary


def get_available_examples() -> list[list[str]]:
    """Gather real benchmark image examples for the upload tab."""
    examples: list[list[str]] = []
    classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
    for c in classes:
        sample_path = RAW_DATA_DIR / c / f"{c}1.jpg"
        if sample_path.exists():
            examples.append([str(sample_path)])
    return examples[:6]


def format_bin_card(res: PredictionResult | LiveFrameResult) -> str:
    """Format accessible HTML card for bin disposal recommendation."""
    border_color = res.bin_color if res.bin_color else "#9E9E9E"
    badge_bg = res.bin_color if res.bin_color else "#616161"

    return f"""
    <div style="background: white; border: 1px solid #e0e0e0; border-left: 6px solid {border_color};
                border-radius: 8px; padding: 18px; margin-top: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
        <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
            <div style="font-size: 1.25rem; font-weight: 700; color: #111;">
                🗑️ {res.bin_name}
            </div>
            <div style="background-color: {badge_bg}; color: white; padding: 4px 12px;
                        border-radius: 14px; font-size: 0.85rem; font-weight: 700; text-transform: uppercase;">
                {res.bin_label}
            </div>
        </div>
        <div style="margin-top: 12px; color: #444; font-size: 0.95rem; line-height: 1.5;">
            <strong>Instructions:</strong> {res.bin_instructions}
        </div>
    </div>
    """


def format_guidance_card(res: PredictionResult) -> str:
    """Format contextual user guidance notification for single images."""
    if res.is_confident:
        bg = "#E8F5E9"
        border = "#A5D6A7"
        text_color = "#2E7D32"
        icon = "✅"
    else:
        bg = "#FFF8E1"
        border = "#FFE082"
        text_color = "#F57F17"
        icon = "⚠️"

    return f"""
    <div style="background-color: {bg}; border: 1px solid {border}; border-radius: 6px;
                padding: 12px; margin-top: 10px; color: {text_color}; font-size: 0.92rem; font-weight: 500;">
        <span>{icon}</span> <strong style="margin-left: 4px;">Guidance:</strong> {res.guidance}
    </div>
    """


def format_live_guidance_card(res: LiveFrameResult) -> str:
    """Format contextual live status badge and guidance card."""
    return f"""
    <div style="background-color: {res.badge_color}15; border: 1px solid {res.badge_color}60; border-radius: 6px;
                padding: 12px; margin-top: 10px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="background-color: {res.badge_color}; color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.82rem; font-weight: 700;">
                {res.badge_label}
            </span>
            <span style="color: #222; font-size: 0.92rem; font-weight: 500;">
                {res.guidance}
            </span>
        </div>
        <div style="font-size: 0.80rem; color: #666; font-weight: 600;">
            Target ~4 FPS Throttled
        </div>
    </div>
    """


def classify_image_handler(
    image: Image.Image | None,
) -> tuple[str, dict[str, float], str, str]:
    """Execute classification on single image and format UI outputs.

    Args:
        image: User-provided PIL Image from Upload or Webcam snapshot.

    Returns:
        Tuple of (header string, probabilities dict for gr.Label, bin card HTML, guidance HTML).
    """
    if image is None:
        empty_bin = """
        <div style="padding: 20px; border-radius: 8px; background: #fafafa; border: 1px dashed #ccc;
                    color: #777; text-align: center; font-size: 0.95rem;">
            No image provided. Please upload an image file or take a camera snapshot.
        </div>
        """
        empty_guide = """
        <div style="padding: 10px; border-radius: 6px; background: #f0f0f0; color: #666; font-size: 0.9rem;">
            Awaiting image input...
        </div>
        """
        return "No Image", {}, empty_bin, empty_guide

    predictor = get_predictor()
    res = predictor.predict(image)

    header = f"{res.top_class.upper()} ({res.confidence * 100:.1f}%)"
    prob_dict = {item.class_name: item.probability for item in res.predictions}
    bin_html = format_bin_card(res)
    guidance_html = format_guidance_card(res)

    return header, prob_dict, bin_html, guidance_html


def live_frame_handler(
    image: Image.Image | np.ndarray | None,
) -> tuple[str, dict[str, float], str, str]:
    """Execute live frame processing with throttling, quality check, and temporal smoothing.

    Args:
        image: Live frame from webcam stream.

    Returns:
        Tuple of (status header, probabilities dict for gr.Label, bin card HTML, guidance HTML).
    """
    pipeline = get_live_pipeline()
    res = pipeline.process_frame(image, enforce_throttle=True)

    prob_dict = {item.class_name: item.probability for item in res.top_predictions}
    bin_html = format_bin_card(res)
    guidance_html = format_live_guidance_card(res)

    return res.status, prob_dict, bin_html, guidance_html


def reset_live_handler() -> tuple[str, dict[str, float], str, str]:
    """Reset live pipeline state and temporal smoother."""
    pipeline = get_live_pipeline()
    pipeline.reset()
    res = pipeline.process_frame(None)

    prob_dict: dict[str, float] = {}
    bin_html = format_bin_card(res)
    guidance_html = format_live_guidance_card(res)

    return "Stabilizer Reset - Ready", prob_dict, bin_html, guidance_html


_DEFAULT_EXPLAINER: GradCAMExplainer | None = None


def get_explainer() -> GradCAMExplainer:
    """Retrieve or initialize cached GradCAMExplainer instance."""
    global _DEFAULT_EXPLAINER
    if _DEFAULT_EXPLAINER is None:
        _DEFAULT_EXPLAINER = GradCAMExplainer()
    return _DEFAULT_EXPLAINER


def explain_image_handler(
    upload_img: Image.Image | None,
    camera_img: Image.Image | None,
    target_choice: str,
) -> tuple[Image.Image | None, str]:
    """Execute on-demand Grad-CAM computation for the active user image."""
    active_img = upload_img if upload_img is not None else camera_img
    if active_img is None:
        return (
            None,
            "⚠️ **No image provided.** Please upload an image or take a camera snapshot first.",
        )

    target_cls = None if target_choice == "Top Prediction" else target_choice.lower().strip()
    explainer = get_explainer()
    res = explainer.explain(active_img, target_class=target_cls)

    info_text = (
        f"**Explained Target:** `{res.target_class.title()}` | "
        f"**Top Prediction:** `{res.predicted_class.title()}` ({res.predicted_prob * 100:.1f}%)\n\n"
        f"*{res.disclaimer}*"
    )
    return res.overlay_image, info_text


CUSTOM_CSS = """
.app-header { text-align: center; margin-bottom: 1.5rem; }
.privacy-notice { font-size: 0.85rem; color: #666; margin-top: 0.5rem; }
.status-badge { font-size: 1.5rem; font-weight: bold; }
"""


def create_app() -> gr.Blocks:
    """Build and assemble the complete Gradio web application."""
    summary_metrics = load_report_summary()
    examples = get_available_examples()

    with gr.Blocks(title="Smart Waste Classifier") as demo:
        with gr.Column(elem_classes=["app-header"]):
            gr.Markdown(
                """
                # ♻️ Smart Waste Classifier
                ### Camera-Ready Automated Waste Sorting & Recycling Guidance
                """
            )
            gr.Markdown(
                "🔒 **Privacy Guarantee:** Images and video frames are processed strictly in-memory "
                "and are **never saved to disk**.",
                elem_classes=["privacy-notice"],
            )

        with gr.Row():
            # Left Column: Inputs (Upload, Camera Snapshot, or Live Camera)
            with gr.Column(scale=5):
                with gr.Tabs():
                    # Tab 1: Upload (FR-10)
                    with gr.TabItem("📁 Upload Image", id="tab_upload"):
                        upload_input = gr.Image(
                            type="pil",
                            sources=["upload"],
                            label="Select or Drop Waste Photo (.jpg, .png)",
                        )
                        with gr.Row():
                            upload_btn = gr.Button("🔍 Classify Uploaded Image", variant="primary")
                            clear_upload_btn = gr.Button("🗑️ Clear")

                        if examples:
                            gr.Examples(
                                examples=examples,
                                inputs=upload_input,
                                label="Sample Dataset Items (Click to Test)",
                            )

                    # Tab 2: Camera Snapshot (FR-11)
                    with gr.TabItem("📸 Camera Snapshot", id="tab_camera"):
                        gr.Markdown(
                            "> **Camera Permission Notice:** Click 'Allow' when your browser requests webcam access. "
                            "If your camera is unavailable or blocked, switch to the **Upload Image** tab."
                        )
                        camera_input = gr.Image(
                            type="pil",
                            sources=["webcam"],
                            label="Capture Frame with Webcam",
                        )
                        with gr.Row():
                            camera_btn = gr.Button("📸 Classify Camera Snapshot", variant="primary")
                            clear_camera_btn = gr.Button("🗑️ Retake")

                    # Tab 3: Live Camera Stream (FR-13 to FR-16)
                    with gr.TabItem("🎥 Live Camera (Real-Time)", id="tab_live"):
                        gr.Markdown(
                            "> **Live Streaming Mode:** Real-time throttled inference (~4 FPS) with automated blur/brightness "
                            "quality checks, center-crop targeting, and multi-frame temporal stabilization. "
                            "If camera permission is denied, use the **Upload Image** tab."
                        )
                        live_input = gr.Image(
                            sources=["webcam"],
                            streaming=True,
                            type="pil",
                            label="Live Camera Stream (Keep Item in Center)",
                        )
                        with gr.Row():
                            reset_live_btn = gr.Button("🔄 Reset Stabilizer State")

            # Right Column: Results & Bin Recommendation (FR-12, FR-14, FR-15, FR-16)
            with gr.Column(scale=6):
                gr.Markdown("### 📊 Classification Result")
                status_output = gr.Textbox(
                    label="Identified Waste Category / Pipeline State",
                    placeholder="Result will appear here...",
                    interactive=False,
                )
                guidance_output = gr.HTML(
                    value="<div style='color: #888;'>Upload an image, snap a photo, or start live camera to begin.</div>",
                    label="Status Guidance",
                )
                chart_output = gr.Label(
                    num_top_classes=3,
                    label="Top Predictions (Confidence)",
                )
                gr.Markdown("### 🗂️ Disposal Recommendation")
                bin_output = gr.HTML(
                    value="<div style='color: #888;'>Disposal advice will appear here.</div>",
                    label="Bin Advice",
                )

                # Grad-CAM Explainability (FR-17) - On-Demand Only
                with gr.Accordion("🔍 Explain Prediction (Grad-CAM Overlay)", open=False):
                    gr.Markdown(
                        "> **On-Demand Visual Interpretability:** Compute Grad-CAM convolutional saliency to inspect which "
                        "visual regions contributed to the prediction. *Note: Grad-CAM is an interpretability visual aid, not proof of causal reasoning.*"
                    )
                    with gr.Row():
                        explain_target_dropdown = gr.Dropdown(
                            choices=["Top Prediction"] + [c.title() for c in CANONICAL_CLASSES],
                            value="Top Prediction",
                            label="Explain Class Target (Predicted or Counterfactual)",
                        )
                        explain_btn = gr.Button("🔬 Generate Grad-CAM Heatmap", variant="secondary")
                    cam_output = gr.Image(
                        type="pil",
                        label="Grad-CAM Saliency Overlay",
                        interactive=False,
                    )
                    cam_info = gr.Markdown(
                        "*Click 'Generate Grad-CAM Heatmap' above to explain the currently selected or uploaded image.*"
                    )

        # Bottom Section: About and Transparency Tab
        with gr.Accordion("ℹ️ Model Details, Benchmark Metrics & Real-World Domain Gap", open=False):
            gr.Markdown(
                f"""
                #### Architectural & Performance Transparency
                - **Model Architecture:** Fine-tuned `efficientnet_b0` CNN with 20% unfrozen trailing layers (Phase 2).
                - **Clean Benchmark Test Accuracy:** `{summary_metrics["test_accuracy"]}` (Macro-F1: `{summary_metrics["test_macro_f1"]}`).
                - **Authentic Camera Field-Set Accuracy:** `{summary_metrics["field_accuracy"]}` (Macro-F1: `{summary_metrics["field_macro_f1"]}`).
                - **Negative Non-Waste Uncertainty Rejection Rate:** `{summary_metrics["negative_rejection"]}` (rejection threshold = 0.60).
                - **Live Stabilization:** Sliding window probability averaging (N=5) and stability confirmation (K=3 consecutive frames) suppress camera noise.
                - **Live Quality Gate:** Computes Laplacian edge variance (min blur score 60) and mean intensity (min brightness 50), providing *'Hold steady'* or *'Too dark'* hints instead of guessing.
                - **Domain Gap Note:** Real-world camera images exhibit specular highlights on plastics, shadows, and crumpled cardboard geometry. When confidence drops below 60%, the classifier flags the item as uncertain and instructs manual inspection.
                """
            )

        # Wire Up Event Listeners
        # 1. Upload Tab Actions
        upload_btn.click(
            fn=classify_image_handler,
            inputs=[upload_input],
            outputs=[status_output, chart_output, bin_output, guidance_output],
        )
        upload_input.change(
            fn=classify_image_handler,
            inputs=[upload_input],
            outputs=[status_output, chart_output, bin_output, guidance_output],
        )
        clear_upload_btn.click(
            fn=lambda: (None, "No Image", {}, "", ""),
            inputs=[],
            outputs=[upload_input, status_output, chart_output, bin_output, guidance_output],
        )

        # 2. Camera Snapshot Tab Actions
        camera_btn.click(
            fn=classify_image_handler,
            inputs=[camera_input],
            outputs=[status_output, chart_output, bin_output, guidance_output],
        )
        camera_input.change(
            fn=classify_image_handler,
            inputs=[camera_input],
            outputs=[status_output, chart_output, bin_output, guidance_output],
        )
        clear_camera_btn.click(
            fn=lambda: (None, "No Image", {}, "", ""),
            inputs=[],
            outputs=[camera_input, status_output, chart_output, bin_output, guidance_output],
        )

        # 3. Live Camera Tab Actions (FR-13 to FR-16)
        live_input.stream(
            fn=live_frame_handler,
            inputs=[live_input],
            outputs=[status_output, chart_output, bin_output, guidance_output],
            stream_every=0.25,
            show_progress="hidden",
        )
        reset_live_btn.click(
            fn=reset_live_handler,
            inputs=[],
            outputs=[status_output, chart_output, bin_output, guidance_output],
        )

        # 4. Grad-CAM Explain Action (FR-17) - On-Demand
        explain_btn.click(
            fn=explain_image_handler,
            inputs=[upload_input, camera_input, explain_target_dropdown],
            outputs=[cam_output, cam_info],
        )

    return demo


def launch_app(
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    share: bool = False,
) -> None:
    """Launch the Gradio web application."""
    logger.info("Initializing Waste Classifier Predictor...")
    get_predictor()
    logger.info("Initializing Live Camera Pipeline...")
    get_live_pipeline()
    app = create_app()
    logger.info(f"Starting server on http://{server_name}:{server_port}")
    app.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    launch_app()
