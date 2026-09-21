---
description: Export PyTorch model checkpoint to ONNX with numerical parity checks
---
1. Run lint and tests; stop if they fail.
2. Load checkpoint and construct model graph with static dummy input (1, 3, 224, 224).
3. Export model graph to ONNX format.
4. Execute parity check verifying max absolute difference between PyTorch and ONNX Runtime is < 1e-4.
5. Benchmark CPU inference latency with ONNX Runtime (target <= 100 ms).
6. Save export artifact and record latency benchmarks.
