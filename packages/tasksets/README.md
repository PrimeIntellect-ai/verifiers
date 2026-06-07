# tasksets

Reusable `verifiers.v1` taskset implementations.

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
import verifiers.v1 as vf
from tasksets import HarborTaskset, HarborTasksetConfig


def load_taskset(config: HarborTasksetConfig) -> HarborTaskset:
    return HarborTaskset(config=config)
```

Do not mutate config objects in loaders. Put defaults on the config class or pass
the intended config from Python/TOML.

## Included Tasksets

| Taskset | Purpose |
| --- | --- |
| `HarborTaskset` | Harbor task directories and Harbor Hub datasets. |
| `OpenEnvTaskset` | Upstream OpenEnv projects with rollout-local user simulation. |
| `OpenRewardTaskset` | Upstream OpenReward environments with rollout-local user simulation. |
| `ReplayTaskset` | HF datasets or explicit local JSONL chat transcripts for replay data. |
| `TextArenaTaskset` | Compatible TextArena single-player games with an MCP user server. |
| `NeMoGymTaskset` | NeMo Gym JSONL task rows. |

Taskset implementations follow the same rules as environment-local tasksets:
config classes are `XXXConfig`, lifecycle logic lives on the class, task rows are
serializable, and utilities exist only for shared messy internals.

## Replay Transcript Data

Use `ReplayTaskset` with `ReplayHarness` when each training example is already a
chat transcript row and each assistant message should become one transcript
turn.

For local data, put one JSON object per line in `.jsonl` files under a directory
owned by the env package:

```jsonl
{"messages":[{"role":"user","content":"Reverse abc."},{"role":"assistant","content":"cba"}]}
```

`messages` must be a JSON array of message objects. Each message must have a
string `role`; all other message fields are preserved. Assistant messages may
appear anywhere in the transcript, and every assistant message is replayed as
one transcript turn.

Set that local directory explicitly, either through `[env.taskset].data_dir` or
on the env-local taskset subclass:

```python
from pathlib import Path

import verifiers.v1 as vf
from harnesses import ReplayHarness
from tasksets import ReplayTaskset, ReplayTasksetConfig


class MyReplayTaskset(ReplayTaskset):
    data_dir = str(Path(__file__).parent / "data")


def load_taskset(config: ReplayTasksetConfig) -> MyReplayTaskset:
    return MyReplayTaskset(config=config)


def load_harness(config: vf.HarnessConfig) -> ReplayHarness:
    return ReplayHarness(config=config)
```

For HF data, set `dataset` to a dataset whose `train` split has a top-level
`messages` field in the same format. Set either `dataset` or `data_dir`, not
both.
