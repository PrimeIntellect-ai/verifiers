# openenv-textarena-v1

<a href="https://github.com/PrimeIntellect-ai/verifiers/tree/main/environments/openenv_textarena_v1">
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="Source Code">
</a>

### Overview

- **Environment ID**: `openenv-textarena-v1`
- **Short description**: OpenEnv TextArena gym integration (default game: `Wordle-v0`) via `OpenEnvTaskset`.
- **Tags**: openenv, gym, textarena, wordle, example

### Datasets

- **Primary dataset(s)**: Seed-generated episodes (one seed per rollout).
- **Source links**: Bundled OpenEnv TextArena project in `proj/` (copied from OpenEnv).
- **Split sizes**: 100 train / 50 eval by default (configurable).

### Task

- **Type**: Multi-turn gym interaction.
- **Output format**: Game actions.
- **Rubric overview**: The OpenEnv taskset sums per-step rewards from the upstream environment.

### Quickstart

Build and register the bundled OpenEnv Docker image in the Prime registry:

```bash
uv run vf-build openenv-textarena-v1
```

Run an evaluation with default settings:

```bash
prime eval run openenv-textarena-v1
```

### Taskset Config

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_train_examples` | int | `100` | Number of training seeds to generate. |
| `num_eval_examples` | int | `50` | Number of eval seeds to generate. |
| `seed` | int | `0` | Base seed for episode generation. |

### Notes

- Upstream TextArena app defaults to `TEXTARENA_ENV_ID=Wordle-v0`.
- To use another game, set environment variables in the OpenEnv project/server config before building.
- `openenv_textarena_v1/taskset.py` defines `render_textarena_prompt` and passes it via `prompt_renderer` so observations are rendered as useful game messages.
