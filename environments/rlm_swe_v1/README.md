# rlm-swe-v1

v1 RLM coding environment using the R2E-Gym SWE taskset and `vf.RLM` harness.

```python
from rlm_swe_v1 import load_environment

env = load_environment()
```

The taskset is fully implemented in this environment package on the v1 stack.
It loads the full `R2E-Gym/R2E-Gym-Subset` train split by default, converts each
row into a v1 task, creates the per-instance sandbox config from the dataset
image, stages hidden tests for scoring, runs `run_tests.sh`, and parses pytest
output for reward.

`RLM` owns the CLI program, intercepted endpoint config, RLM installation, and
trajectory filtering. Harbor is not used here because the R2E setup is dataset
and image backed rather than a Harbor task directory corpus.
