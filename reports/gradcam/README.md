# Grad-CAM Visual Explainability Report (FR-17)

## Overview & Interpretability Scope
This directory contains on-demand **Grad-CAM (Gradient-weighted Class Activation Mapping)** visualizations produced by [`GradCAMExplainer`](file:///c:/Gaurav's%20Den/crazy-shits/smart-classification/src/waste_classifier/explainability.py) for the fine-tuned `efficientnet_b0` classifier.

> **Methodological Disclaimer:**
> Grad-CAM highlights discriminative convolutional feature regions associated with class activations; it is an interpretability visual aid and does not constitute proof of causal reasoning.
> Regions highlighted in red/yellow indicate high gradient-weighted convolutional feature activations in the final feature map (`conv_head`), pointing to visual evidence the network relied upon to project class logits.

## Key Observations
1. **Geometric Localization on Solid Waste**:
   - For structured items (cardboard boxes, metal cans), the model primarily focuses on rigid contours, structural edges, and labels.
2. **Transparent Material Confusion (Glass vs. Plastic)**:
   - Transparent bottles exhibit high activations along reflections and specular highlights.
   - When counterfactually querying the `plastic` class for a `glass` item, activations often focus on reflective highlights or caps rather than glass bottle necks.
3. **Background Invariance**:
   - High activations remain localized on the foreground waste object, with minimal spurious activation on plain neutral backgrounds.

## Generated Figures
- **Glass**: Target `glass` -> Predicted `glass` (87.2%) -> `[gradcam_glass_glass.png](./gradcam_glass_glass.png)`
- **Cardboard**: Target `cardboard` -> Predicted `cardboard` (98.4%) -> `[gradcam_cardboard_cardboard.png](./gradcam_cardboard_cardboard.png)`
- **Metal**: Target `metal` -> Predicted `metal` (90.6%) -> `[gradcam_metal_metal.png](./gradcam_metal_metal.png)`
- **Plastic**: Target `plastic` -> Predicted `glass` (47.5%) -> `[gradcam_plastic_plastic.png](./gradcam_plastic_plastic.png)`

- **Counterfactual Comparison**: `[gradcam_counterfactual_glass_vs_plastic.png](./gradcam_counterfactual_glass_vs_plastic.png)`
