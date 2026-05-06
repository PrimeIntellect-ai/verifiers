# rlm-swe-v1

v1 RLM coding environment using the packaged `vf.HarborTaskset` and `vf.RLM`
harness.

```python
import verifiers.v1 as vf

env = vf.Env(
    taskset=vf.HarborTaskset(tasks="/path/to/harbor/tasks"),
    harness=vf.RLM(workdir="/app", rlm_tools=["bash", "edit"]),
)
```

`load_environment()` uses the packaged smoke task when `tasks` is omitted.
Pass `tasks="/path/or/harbor/registry/id"` to run a real Harbor taskset.

`HarborTaskset` owns task loading, sandbox configuration, task uploads, test
scoring, and cleanup. `RLM` owns the CLI program, intercepted endpoint config,
RLM installation, optional RLM skill upload, and trajectory filtering.
