# openenv-echo

<a href="https://github.com/PrimeIntellect-ai/verifiers/tree/main/environments/openenv_echo">
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="Source Code">
</a>

### Overview

- **Environment ID**: `openenv-echo`
- **Short description**: OpenEnv Echo environment via `OpenEnvEnv`, demonstrating MCP tool-calling in Prime Sandboxes.
- **Tags**: openenv, mcp, tools, example

### Datasets

- **Primary dataset(s)**: Seed-generated episodes (one seed per rollout).
- **Source links**: Bundled OpenEnv Echo project in `proj/` (copied from OpenEnv).
- **Split sizes**: 100 train / 50 eval by default (configurable).

### Task

- **Type**: Tool use, multi-turn.
- **Parser**: Default `Parser` (no special formatting).
- **Rubric overview**: `OpenEnvEpisodicSumRubric` sums per-step rewards; `MultiTurnMonitorRubric` tracks turn count.

### Quickstart

Build and register the bundled OpenEnv Docker image in the Prime registry:

```bash
uv run vf-build openenv-echo
```

This writes `environments/openenv_echo/proj/.build.json` with the fully qualified image reference and runtime metadata.

Verify the image is ready (status **Ready** or **Completed**):

```bash
prime images list
```

Run an evaluation with default settings:

```bash
prime eval run openenv-echo
```

Configure model and sampling:

```bash
prime eval run openenv-echo \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7
```

Notes:
- If your environments directory is not `./environments`, run:
`uv run vf-build openenv-echo -p /path/to/environments`
- If you customize the bundled OpenEnv project, rerun `uv run vf-build openenv-echo` (the `proj/.build.json` manifest is updated).
- `openenv_echo.py` defines `render_echo_prompt()` and passes it via `prompt_renderer`
to keep the initial MCP prompt concise.

### Troubleshooting

If you see errors like `waiting to start: trying and failing to pull image`, it means the image is not available to the sandbox. Common causes:
- The image build is still running or failed (`prime images list` should show **Ready** or **Completed**).
- The image reference in `proj/.build.json` is stale or invalid.
- The image is private or not accessible to your team.

If `prime images list` shows **Ready** but the sandbox still cannot pull the image, escalate to the platform team with:
- Image name/tag
- Build status/output from `prime images list`
- Sandbox ID and timestamp from the error log

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_train_examples` | int | `100` | Number of training seeds to generate. |
| `num_eval_examples` | int | `50` | Number of eval seeds to generate. |
| `seed` | int | `0` | Base seed for episode generation. |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Sum of per-step rewards from the OpenEnv environment. |
| `num_turns` | Number of turns taken in the rollout. |
