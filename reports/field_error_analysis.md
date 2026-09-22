# Field-Set Evaluation & Camera Domain Gap Analysis

## Executive Summary

The field-set evaluation assesses model generalization under authentic camera-capture conditions, featuring varied lighting, complex backgrounds, realistic perspectives, sensor noise, and negative non-waste objects. This camera domain test is kept strictly held-out from model training and hyperparameter tuning.

### Benchmark Test Set vs. Field-Set Performance

| Metric | Benchmark Test Set | Real Camera Field Set | Domain Shift Gap (Delta) |
|---|---|---|---|
| **Overall Accuracy** | 83.16% | 60.00% | **-23.16 pp** |
| **Macro-F1 Score** | 80.90% | 58.40% | **-22.50 pp** |
| **Macro Precision** | 80.86% | 58.94% | **-21.92 pp** |
| **Macro Recall** | 84.01% | 60.00% | **-24.01 pp** |
| **Weighted F1 Score** | 84.01% | 58.40% | **-25.61 pp** |

## Per-Class Performance & Domain Sensitivity

| Class | Test Recall | Field Recall | Recall Delta | Test F1 | Field F1 | F1 Delta | Support |
|---|---|---|---|---|---|---|---|
| **cardboard** | 83.61% | 65.00% | -18.61% | 89.47% | 66.67% | -22.80% | 20 |
| **glass** | 84.00% | 55.00% | -29.00% | 85.71% | 62.86% | -22.85% | 20 |
| **metal** | 82.26% | 70.00% | -12.26% | 80.95% | 63.64% | -17.31% | 20 |
| **paper** | 82.22% | 70.00% | -12.22% | 85.06% | 71.79% | -13.27% | 20 |
| **plastic** | 81.94% | 85.00% | +3.06% | 86.13% | 66.67% | -19.46% | 20 |
| **trash** | 90.00% | 15.00% | -75.00% | 58.06% | 18.75% | -39.31% | 20 |

**Weakest Performing Class in Field Domain:** `trash` (F1-Score: 18.75%)

## Representative Error Patterns & Confusion Modes

Analysis of misclassifications reveals specific domain-shift mechanisms in real camera images:
1. **Specular Highlights and Transparency (Plastic vs. Glass):** Transparent plastic containers under direct ambient illumination produce harsh reflections resembling clear glass bottles.
2. **Deformation and Geometry (Crumpled Paper vs. Cardboard):** Crushed cardboard boxes or corrugated craft paper exhibit wrinkles that the convolutional filters confuse with crumpled printer paper.
3. **Heterogeneous Composite Materials (Trash Ambiguity):** Complex items with multiple material textures (e.g. snack wrappers with foil lining, mixed packaging) trigger uncertainty between trash, plastic, and metal.
4. **Shadow and Exposure Variance:** Handheld camera captures introduce uneven directional lighting and shadows not present in white-backdrop lab photography.

## Negative Sample & Uncertainty Rejection Analysis

- **Total Non-Waste Negative Samples Tested:** 15
- **Confidence Rejection Threshold:** 0.60
- **Successfully Rejected / Flagged Uncertain:** 13 / 15 (86.67%)
- **Mean Negative Confidence Score:** 44.70%

When presented with unrelated non-waste objects (e.g., clothing, footwear, biological items, empty frames), the model's softmax confidence remained below the 0.60 threshold for 86.67% of samples, triggering safe 'Uncertain / Please re-frame' fallback behavior instead of erroneous bin placement.

## Operational Recommendations for Production Deployment

- **Temporal Frame Smoothing:** Maintain a rolling average of prediction probabilities across 5-10 consecutive video frames (FR-11) to eliminate single-frame transient classification flickers.
- **User Quality Hints:** Provide dynamic guidance in the live UI (FR-12) detecting underexposure, blur, or distant objects, instructing the user to bring the object closer.
- **Uncertainty Gating:** Refuse classification with explicit guidance if the top-1 softmax probability is below 0.60 or if the top-2 margin is under 0.15.

---
*Report generated automatically on 2026-09-23 00:50:24 (Git commit: `21cc117`).*