# rlm-swe-v1

v1 RLM coding environment using the R2E-Gym SWE taskset and `vf.RLM` harness.

```python
import verifiers as vf

env = vf.load_environment("rlm-swe-v1")
```

Tune the taskset and harness through typed v1 config objects:

```python
import verifiers as vf
from rlm_swe_v1 import RlmSweTasksetConfig, load_environment

env = load_environment(
    config=vf.EnvConfig(
        taskset=RlmSweTasksetConfig(timeout_minutes=90),
        harness=vf.RLMConfig(rlm_repo_ref="main", rlm_tools=["bash", "edit"]),
    )
)
```

The taskset is fully implemented in this environment package on the v1 stack.
It loads the full `R2E-Gym/R2E-Gym-Subset` train split by default, converts each
row into a v1 task, creates the per-instance sandbox config from the dataset
image, stages hidden tests for scoring, runs `run_tests.sh`, and parses pytest
output for reward.

`RLM` owns the CLI program, intercepted endpoint config, RLM installation, and
trajectory filtering. Harbor is not used here because the R2E setup is dataset
and image backed rather than a Harbor task directory corpus.
