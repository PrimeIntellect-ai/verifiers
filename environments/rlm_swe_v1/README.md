# rlm-swe-v1

v1 RLM coding environment using the R2E-Gym SWE taskset and packaged `RLM`
harness.

```python
import verifiers.v1 as vf

env = vf.load_environment("rlm-swe-v1")
```

Tune the taskset, harness, and runtime through TOML-shaped v1 config data:

```python
import verifiers.v1 as vf

env = vf.load_environment(
    "rlm-swe-v1",
    config={
        "taskset": {"timeout_minutes": 90},
        "harness": {
            "tools": ["bash", "edit"],
            "cwd": "/testbed",
        },
        "runtime": {"type": "subprocess"},
    },
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
