# tasksets

Reusable v1 `vf.Taskset` implementations for Verifiers.

Tasksets own task data, task controls, task-owned tools, user behavior, rewards,
metrics, and task-specific setup/cleanup. They are sibling packages to
`harnesses`: a taskset should run with the base `vf.Harness` unless a reusable
execution harness is actually needed.

## Install

```bash
uv add tasksets
```

Install only the backend extras you need:

```bash
uv add "tasksets[openenv,openreward,ta,nemogym]"
```

From `verifiers`, the matching extras are:

```bash
uv add "verifiers[tasksets]"
uv add "verifiers[openenv]"
uv add "verifiers[openreward]"
uv add "verifiers[ta]"
uv add "verifiers[nemogym]"
```

## Golden Loader Shape

Environment packages should expose a typed child loader and let Verifiers coerce
the `[env.taskset]` config through that annotation:

```python
import verifiers as vf
from tasksets import HarborTaskset, HarborTasksetConfig


def load_taskset(config: HarborTasksetConfig) -> HarborTaskset:
    return HarborTaskset(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    """Loader pattern for all Taskset/Harness environments."""
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
    )
```

Do not mutate config objects in loaders. Put defaults on the config class or pass
the intended config from Python/TOML.

## Included Tasksets

| Taskset | Purpose |
| --- | --- |
| `HarborTaskset` | Harbor task directories and Harbor Hub datasets. |
| `OpenEnvTaskset` | Upstream OpenEnv projects with out-of-the-box task/tool use. |
| `OpenRewardTaskset` | Upstream OpenReward environments and rollout-local session tools. |
| `ReplayTaskset` | HF or env-local JSON chat transcripts for replay data. |
| `TextArenaTaskset` | Compatible TextArena single-player games with a taskset-owned `vf.User`. |
| `NeMoGymTaskset` | NeMo Gym JSONL task rows. |

Taskset implementations follow the same rules as environment-local tasksets:
config classes are `XXXConfig`, lifecycle logic lives on the class, task rows are
serializable, and utilities exist only for shared messy internals.

## Replay Transcript Data

Use `ReplayTaskset` with `ReplayHarness` when each training example is already a
chat transcript row and each assistant message should become one trajectory
step.

For env-local data, put one JSON object per file under `data/` in the env
package:

```json
{
  "messages": [
    {"role": "user", "content": "Reverse abc."},
    {"role": "assistant", "content": "cba"}
  ]
}
```

`messages` must be a JSON array of message objects. Each message must have a
string `role`; all other message fields are preserved. Assistant messages may
appear anywhere in the transcript, and every assistant message is replayed as
one trajectory step.

Make the env-local taskset an empty subclass so the `data/` directory belongs to
the environment package:

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

For HF data, set `dataset` to a dataset whose `train` split has a top-level
`messages` field in the same format.
