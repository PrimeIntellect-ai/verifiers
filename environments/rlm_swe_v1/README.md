# rlm-swe-v1

v1 RLM coding environment using the R2E-Gym SWE taskset and packaged `RLM`
harness.

```python
import verifiers.v1 as vf

env = vf.load_environment("rlm-swe-v1")
```

Tune the taskset and harness through typed v1 config objects:

```python
import verifiers.v1 as vf
from harnesses import RLMConfig
from rlm_swe_v1 import RlmSweTasksetConfig, load_environment

env = load_environment(
    config=vf.EnvConfig(
        taskset=RlmSweTasksetConfig(timeout_minutes=90),
        harness=RLMConfig(
            tools=["bash", "edit"],
            cwd="/testbed",
        ),
    )
)
```

The taskset is fully implemented in this environment package on the v1 stack.
It loads the full `R2E-Gym/R2E-Gym-Subset` train split by default, converts each
row into a v1 task, records the per-instance container image as task runtime
config, stages hidden tests for scoring, runs `run_tests.sh`, and parses pytest
output for reward.

`RLM` owns the CLI command and protocol endpoint wiring. Harbor is not used here
because the R2E setup is dataset and image backed rather than a Harbor task
directory corpus.
