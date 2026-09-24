# Model Export & Numerical Parity Verification Report (FR-21)

## Executive Summary
- **Status:** **`PASS`**
- **Model Architecture:** `efficientnet_b0`
- **Source Checkpoint:** `C:\Gaurav's Den\crazy-shits\smart-classification\checkpoints\phase2\phase2-efficientnet_b0-epoch=11-val_f1=0.781.ckpt`
- **Exported ONNX Binary:** `models\model.onnx`
- **File Size:** `15.30 MB`
- **ONNX Opset:** `17`
- **Target Parity Tolerance:** `1.0e-04`
- **Observed Max Absolute Difference:** `7.87e-06`
- **Observed Max Softmax Difference:** `9.24e-07`
- **Target Latency Requirement:** `<= 100 ms`
- **Measured CPU Latency (Mean):** `12.86 ± 1.34 ms` (77.8 FPS)

---

## Numerical Parity Test Matrix

| Test Input Case | Batch | Max Logit Diff | Max Prob Diff | Class Match | Result |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `standard_normal_b1` | 1 | 1.55e-06 | 8.94e-08 | True | PASS |
| `standard_normal_b2` | 2 | 6.14e-06 | 5.22e-07 | True | PASS |
| `standard_normal_b4` | 4 | 2.38e-06 | 2.09e-07 | True | PASS |
| `uniform_0_1` | 1 | 1.31e-06 | 3.58e-07 | True | PASS |
| `zeros_tensor` | 1 | 7.87e-06 | 9.24e-07 | True | PASS |
| `ones_tensor` | 1 | 6.50e-06 | 6.56e-07 | True | PASS |
| `imagenet_synthetic_normalized` | 2 | 2.50e-06 | 3.13e-07 | True | PASS |

### Parity Conclusion
The PyTorch reference implementation and ONNX Runtime execution engine produce numerically identical predictions across all tested distribution regimes (standard normal, uniform, zero/one edge cases, and synthetic normalized images). With a maximum observed logit discrepancy of **7.87e-06** (well below the threshold of 1.0e-04), zero class divergence was detected.

---

## ONNX Runtime CPU Latency & Throughput Benchmark

| Latency Percentile | Value (ms) |
| :--- | :---: |
| **Mean ± Std** | **12.86 ± 1.34 ms** |
| **Median (p50)** | **13.02 ms** |
| **95th Percentile (p95)** | **14.45 ms** |
| **99th Percentile (p99)** | **14.71 ms** |
| **Throughput** | **77.8 frames/sec** |

The measured mean CPU latency of **12.86 ms** satisfies the production SLA requirement of <= 100 ms per frame.

---
*Report generated automatically by `waste-classifier export` on 2026-09-24T16:30:44.157472 (git commit `5172407`).*
