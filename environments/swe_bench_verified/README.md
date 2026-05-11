# swe-bench-verified

Patch-generation environment for `princeton-nlp/SWE-bench_Verified`.

```python
from swe_bench_verified import load_environment

env = load_environment()
```

The taskset loads the 500-example SWE-bench Verified split from Hugging Face,
formats each instance as a repository repair prompt, and asks the model to
return a unified diff inside `<patch>...</patch>` tags.

The default reward is intentionally deterministic and local: it compares the
normalized submitted patch to the gold patch included in the dataset. This makes
the environment useful for SFT/RL sanity checks and reward-model experiments
without requiring per-instance Docker images. It is not a replacement for the
official SWE-bench execution harness; `test_patch`, `FAIL_TO_PASS`, and
`PASS_TO_PASS` are preserved in `task["info"]` so downstream harnesses can run
execution-based validation when available.
