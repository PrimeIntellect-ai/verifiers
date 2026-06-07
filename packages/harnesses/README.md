# harnesses

Reusable `verifiers.v1` harness implementations.

Harnesses own reusable execution mechanisms: command agents, framework adapters,
runtime calls, and execution artifacts. Task data, task tools, users, rewards,
and task-specific config belong to tasksets.

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
import verifiers.v1 as vf
from harnesses import OpenCode, OpenCodeConfig


def load_harness(config: OpenCodeConfig) -> OpenCode:
    return OpenCode(config=config)
```

Omit `harness.py` when the environment does not own a reusable execution
mechanism; the component loader will use the base harness.

## Included Harnesses

| Harness | Purpose |
| --- | --- |
| `OpenCode` | OpenCode CLI agent. |
| `Pi` | Pi Coding Agent. |
| `MiniSWEAgent` | mini-swe-agent. |
| `Terminus2` | Harbor Terminus agent. |
| `RLM` | Recursive language model command harness. |
| `ReplayHarness` | Replays stored assistant messages into transcript turns without model calls. |
| `NeMoGymHarness` | NeMo Gym rollout collection. |

Command harness configs may expose task-relevant execution knobs, but the
harness owns command construction and records command output in
`state.artifacts`.

## Replay Stored Transcripts

Use `ReplayHarness` when each task row already contains a top-level `messages`
chat transcript and each assistant message should become one transcript turn:

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

`messages` must be a JSON array of message objects with string `role` fields.
Non-assistant messages may appear before, between, or after assistant messages.
`vf.HarnessConfig` defaults to replaying every assistant message; set
`max_turns` only when the replay should be capped.

The replayed transcript keeps `tokens=None`; token IDs and logprobs remain the
responsibility of the trainer or renderer that consumes the final transcript.

## Agent Versions

Command agents use `name@version` specs where their installer supports a
versioned package or release. Use `@latest` for a moving latest install:

```toml
[eval.harness]
id = "harnesses.opencode"
version = "PrimeIntellect-ai/opencode@latest"
```

```toml
[eval.harness]
id = "harnesses.mini_swe_agent"
version = "mini-swe-agent@2.2.8"
```
