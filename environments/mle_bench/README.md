# mle-bench

V1 taskset/harness environment for MLE-Bench competition submissions.

```python
from mle_bench import load_environment

env = load_environment()
```

The taskset represents each MLE-Bench competition as a sandboxed machine-learning
engineering task. The model receives the benchmark-level instructions plus the
competition description and must create:

```text
/home/submission/submission.csv
```

When run in an image that has MLE-Bench data and the validation server/script
available, the reward calls:

```bash
/home/validate_submission.sh /home/submission/submission.csv
```

and gives reward `1.0` only when the benchmark validator accepts the submission.
This keeps the environment aligned with the official MLE-Bench submission
contract without downloading Kaggle data during import or local unit tests.

For handoff to the benchmark grader, `grading_submission_row(task)` returns the
JSONL row expected by `mlebench grade`:

```json
{"competition_id": "spaceship-titanic", "submission_path": "/home/submission/submission.csv"}
```

By default, the environment uses the low-complexity/lite split. If the
`mlebench` Python package is installed, descriptions are loaded from its
registry. Otherwise the built-in competition IDs are still exposed so the
environment can be imported and tested without the upstream repo or Kaggle
credentials.
