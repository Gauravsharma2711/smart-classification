---
name: ml-experiment-hygiene
description: Checklist for any change to data, training, evaluation, or live-inference code in this project.
---

# ML experiment hygiene
Before finishing a change:
1. Splits are created once, saved as a manifest, and disjoint (a test exists).
2. Augmentation is applied to the train split only.
3. Seeds are set; config and git commit are logged.
4. Metrics include macro-F1 and per-class recall, not only accuracy.
5. The test set and field set were not used for tuning.
6. Inference preprocessing equals evaluation preprocessing.
7. No camera frames or uploads are persisted.
8. Run `ruff` and `pytest` and report the results.
