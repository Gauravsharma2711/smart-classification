---
description: Run a full two-phase training run and record results
---
1. Run lint and tests; stop if they fail.
2. Run Phase 1 with the given config; save the best checkpoint.
3. Run Phase 2 from the Phase 1 checkpoint.
4. Compute validation metrics and append a row to reports/runs.csv
   (config name, seed, val macro-F1, git commit).
5. Summarize results and flag anything unusual (overfitting, a class with very low recall).
