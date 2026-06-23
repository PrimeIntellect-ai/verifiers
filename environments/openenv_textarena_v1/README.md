# openenv-textarena-v1

### Overview

- **Environment ID**: `openenv-textarena-v1`
- **Short description**: OpenEnv TextArena Wordle through the reusable v1 gym adapter.
- **Tags**: openenv, gym, textarena, wordle, v1

### Datasets

- **Primary dataset(s)**: Seeded `Wordle-v0` episodes from the public OpenEnv image.
- **Split sizes**: `num_tasks` episodes, default 100.

### Task

- **Type**: Multi-turn game with a colocated user simulator.
- **Rubric overview**: Cumulative game-authoritative reward returned by OpenEnv.

### Quickstart

```bash
uv pip install -e environments/openenv_textarena_v1
uv run eval openenv-textarena-v1 -n 3 --harness.runtime.type docker
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | --- | --- | --- |
| `num_tasks` | int | `100` | Number of seeded episodes. |
| `seed` | int | `0` | First episode seed. |
| `system_prompt` | str | Wordle prompt | Model behavior prompt. |

### Metrics

| Metric | Meaning |
| --- | --- |
| `openenv_reward` | Cumulative reward returned by OpenEnv. |

## How It Works

The taskset pins OpenEnv's public TextArena image and selects the gym contract.
The reusable adapter starts it over a Unix socket and drives Wordle through a
colocated v1 user simulator. This package contains no server or Docker build.
