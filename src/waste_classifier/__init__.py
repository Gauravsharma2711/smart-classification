"""Smart Waste Classification package."""

__version__ = "0.1.0"

from waste_classifier.explainability import GradCAMExplainer, GradCAMResult
from waste_classifier.live import FrameSmoother, LiveCameraPipeline, QualityGate

__all__ = [
    "__version__",
    "FrameSmoother",
    "QualityGate",
    "LiveCameraPipeline",
    "GradCAMExplainer",
    "GradCAMResult",
]
