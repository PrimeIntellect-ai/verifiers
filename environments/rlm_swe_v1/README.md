# rlm-swe-v1

v1 RLM coding environment using the reusable SWE taskset adapter and `vf.RLM`
harness.

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

The taskset is implemented by `vf.SWETaskset`. It wraps the existing SWE
tasksets on the v1 stack, converts each row into a v1 task, creates per-instance
sandbox and program config, keeps hidden tests for scoring, and bridges the
legacy SWE setup/reward hooks into v1 lifecycle handlers. The default backend is
R2E-Gym; set `task_type` to select other SWE backends such as `swebench`,
`multiswe`, `openswe`, `swelego-real`, `swerebench-v2`, or SWE-Smith variants.

`RLM` owns the CLI program, intercepted endpoint config, RLM installation, and
trajectory filtering. Harbor is not used here because the R2E setup is dataset
and image backed rather than a Harbor task directory corpus.
