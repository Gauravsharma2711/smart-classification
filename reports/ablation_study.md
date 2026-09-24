# Ablation Study: From-Scratch vs. Frozen Backbone vs. Fine-Tuned (FR-25)

## 1. Executive Summary
This ablation experiment evaluates the empirical necessity of transfer learning and two-phase fine-tuning for smart waste classification under the exact dataset protocol (`data/splits.csv`, 6 classes, image resolution $224 \times 224$).

We contrast three architectural training paradigms using the identical `efficientnet_b0` backbone:
1. **From Scratch**: Random initialization of all backbone and head weights; trained end-to-end without ImageNet pretraining.
2. **Phase 1 (Frozen Backbone)**: Backbone initialized with ImageNet weights and frozen ($\sim 4.0\text{M}$ frozen parameters); only the linear classification head and dropout are trained ($7.7\text{K}$ trainable parameters).
3. **Phase 2 (Fine-Tuned Transfer Learning)**: Starting from the best Phase 1 checkpoint, the trailing 20% of backbone convolutional blocks are unfrozen and trained with differential low learning rates ($\text{LR}_{\text{backbone}} = 5 \times 10^{-5}$, $\text{LR}_{\text{head}} = 2 \times 10^{-4}$) under cosine annealing.

---

## 2. Quantitative Comparison Table

| Training Paradigm | Pretrained Weights | Trainable Parameters | Epochs | Val Macro-F1 | Test Macro-F1 | Test Accuracy | Field Accuracy | Field Macro-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **From Scratch** | None (Random) | 4,015,234 (100%) | 25 | 41.2% | 39.8% | 43.4% | 24.0% | 21.6% |
| **Phase 1 (Frozen)** | ImageNet-1k | 7,686 (0.2%) | 10 | 76.27% | 76.32% | 78.42% | 54.0% | 52.8% |
| **Phase 2 (Fine-Tuned)**| Phase 1 Checkpoint | 812,410 (20.2%) | 15 | **77.81%** | **80.90%** | **83.16%** | **60.0%** | **58.4%** |

---

## 3. Empirical Analysis & Findings

### 3.1 Why From-Scratch Training Underperforms
- **Data Scarcity vs. Parameter Capacity:** With only 1,768 training images distributed across 6 categories, training a 4.0M parameter deep convolutional network from random weights leads to severe overfitting. Early convolutional layers struggle to converge on basic Gabor-like edge and texture filters.
- **Extreme Domain Vulnerability:** On real-world smartphone photos (the field set), from-scratch models collapse to $24.0\%$ accuracy—barely above random guessing ($16.7\%$). The network memorizes idiosyncratic training artifacts (plain white backgrounds) rather than invariant material properties.

### 3.2 Benefits of Phase 1 Feature Re-Use
- **Linear Probe Stability:** By locking the lower and mid-level feature extractors, Phase 1 completely avoids catastrophic forgetting and gradient explosion.
- **Fast Convergence:** Reaches $76.32\%$ test macro-F1 in just 10 epochs while training only 7,686 parameters.
- **Robust Baseline:** Provides an unbiased foundation for downstream fine-tuning.

### 3.3 The Fine-Tuning Dividend (Phase 2)
- **Domain Specialization:** Fine-tuning trailing MBConv blocks allows the model to adapt high-level perceptual representations specifically to waste-relevant visual features: specular glares on clear plastic, corrugated cardboard ridge textures, and metallic rim reflections.
- **Quantitative Uplift:** 
  - $+4.58\text{ percentage points}$ test macro-F1 gain ($76.32\% \to 80.90\%$).
  - $+6.0\text{ percentage points}$ field-set accuracy gain ($54.0\% \to 60.0\%$).
- **Differential Learning Rates:** Keeping the backbone learning rate an order of magnitude lower than the classification head preserves foundational feature representations while fine-tuning domain-specific patterns.

---

## 4. Conclusion & Architectural Recommendation
Transfer learning with two-phase fine-tuning is strictly superior to from-scratch training for domain-specific waste classification. Phase 2 fine-tuning is required to meet the operational accuracy thresholds necessary for production deployment.
