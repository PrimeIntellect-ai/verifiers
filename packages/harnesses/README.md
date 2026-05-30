# harnesses

Reusable v1 `vf.Harness` implementations for Verifiers.

Harnesses own rollout execution: programs, command agents, framework adapters,
endpoint interception, primary sandbox placement, execution setup, and execution
artifacts. Task data, task-owned tools, users, rewards, and task-specific config
belong to tasksets.

## Install

```bash
uv add harnesses
```

From `verifiers`, use:

```bash
uv add "verifiers[harnesses]"
uv add "verifiers[packages]"
```

## Golden Loader Shape

Environment packages should expose a typed child loader and let Verifiers coerce
the `[env.harness]` config through that annotation:

```python
import verifiers as vf
from harnesses import OpenCode, OpenCodeConfig


def load_harness(config: OpenCodeConfig) -> OpenCode:
    return OpenCode(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    """Loader pattern for all Taskset/Harness environments."""
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
    )
```

Use `vf.load_harness(config=config.harness)` when the environment does not
own a reusable execution mechanism.

## Included Harnesses

| Harness | Purpose |
| --- | --- |
| `OpenCode` | OpenCode CLI agent. |
| `Pi` | Pi Coding Agent. |
| `MiniSWEAgent` | mini-swe-agent. |
| `Terminus2` | Harbor Terminus agent. |
| `RLM` | Recursive language model command harness. |
| `NeMoGymHarness` | NeMo Gym rollout collection. |

Harness implementations resolve to one `ProgramConfig` shape. Command harness
configs may expose task-relevant execution knobs, but the harness owns command
construction, channel wiring, sandbox placement, and artifacts.
