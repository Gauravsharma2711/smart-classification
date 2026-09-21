# AGENTS.md

## Project

Smart Waste Classification: software-only waste classification using upload, camera snapshot, and live camera.

**Source of truth:** `PRD_software_only.md`

Follow the PRD. Do not invent requirements, features, datasets, architecture, or results.

---

## 1. Work Loop

For every task:

1. Read the relevant PRD requirement.
2. Inspect the existing code before changing it.
3. Make a small implementation plan.
4. Implement only the required change.
5. Add/update tests.
6. Run tests and lint.
7. Inspect the result and fix failures.
8. Repeat until the acceptance criteria are satisfied.
9. Commit the completed work.
10. Push the commit to GitHub.

Never declare a task complete while known tests or acceptance criteria are failing.

Do not make unrelated changes.

---

## 2. ML Safety

* Never fabricate experiment results.
* Never use the test set for training or tuning.
* Never use the field set for training or tuning in the main experiment.
* Keep train/validation/test splits reproducible.
* Training augmentation only; validation/test preprocessing must be deterministic.
* Inference preprocessing must match evaluation preprocessing.
* Keep configuration in the project's YAML/config files rather than hardcoding it.

---

## 3. Code Quality

Use the project's existing stack and dependencies.

Do not add a dependency unless it is required by the PRD or clearly necessary.

Every new reusable module must have appropriate tests.

Before committing:

```bash
uv run ruff check .
uv run pytest -q
```

Fix failures before committing.

---

## 4. Git and GitHub

After a task is verified, commit and push it directly to the configured GitHub repository.

Use professional commit messages with this format:

`<TYPE>: <short description>`

Allowed types:

* `Add:` new functionality
* `Fix:` bug or error correction
* `Test:` tests or validation
* `Refactor:` code restructuring without behavior change
* `Docs:` documentation
* `Config:` configuration/dependency changes
* `Experiment:` ML experiment changes or experiment infrastructure
* `Deploy:` deployment changes
* `Chore:` maintenance

Examples:

```text
Add: implement dataset ingestion
Test: add leak-free split validation
Fix: correct inference preprocessing mismatch
Experiment: add Phase 2 fine-tuning
Config: add EfficientNet training configuration
Docs: update training instructions
Deploy: configure HTTPS Gradio deployment
```

For a task that both implements functionality and adds tests, use the primary purpose of the change.

Do not create vague commits such as:

```text
update
changes
fixed stuff
final
done
```

Before pushing:

1. Check `git status`.
2. Review `git diff`.
3. Run tests and lint.
4. Confirm no datasets, checkpoints, secrets, camera images, or uploaded images are being committed.
5. Commit using the professional format above.
6. Push to GitHub.
7. Report the commit hash and push status.

Do not rewrite or delete previous commits unless explicitly instructed.

---

## 5. Reporting

After each completed task, report briefly:

* What changed
* Tests run
* Result
* Commit message
* Commit hash
* GitHub push status
* Any remaining issue

Keep reports factual and concise.
