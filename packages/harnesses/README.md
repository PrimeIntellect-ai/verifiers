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
| `ReplayHarness` | Replays stored assistant messages into trajectory steps without model calls. |
| `NeMoGymHarness` | NeMo Gym rollout collection. |

Harness implementations resolve to one `ProgramConfig` shape. Command harness
configs may expose task-relevant execution knobs, but the harness owns command
construction, channel wiring, sandbox placement, and artifacts.

## Replay Stored Transcripts

Use `ReplayHarness` when each task row already contains a top-level `messages`
chat transcript and each assistant message should become one trajectory step:

```python
from harnesses import ReplayHarness, ReplayHarnessConfig
from tasksets import ReplayTaskset, ReplayTasksetConfig


class MyReplayTaskset(ReplayTaskset):
    pass


def load_taskset(config: ReplayTasksetConfig) -> MyReplayTaskset:
    return MyReplayTaskset(config=config)


def load_harness(config: ReplayHarnessConfig) -> ReplayHarness:
    return ReplayHarness(config=config)
```

`messages` must be a JSON array of message objects with string `role` fields.
Non-assistant messages may appear before, between, or after assistant messages.
`ReplayHarnessConfig` defaults to replaying every assistant message; set
`max_turns` only when the replay should be capped.

The replayed trajectory keeps `tokens=None`; token IDs and logprobs remain the
responsibility of the trainer or renderer that consumes the final transcript.
