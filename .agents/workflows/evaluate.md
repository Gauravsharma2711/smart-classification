---
description: Run deliberate evaluation on held-out test set and field set
---
1. Run lint and unit tests; confirm they pass.
2. Load the best trained model checkpoint or exported ONNX model.
3. Evaluate on the test set (`data/splits.csv` test manifest).
4. Evaluate on the field set (`data/field_set`).
5. Output metrics JSON, classification reports, and confusion matrices into `reports/`.
6. Record and analyze domain gap between clean test set and real-world camera field set.
