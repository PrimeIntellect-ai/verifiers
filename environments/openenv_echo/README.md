# openenv-echo

### Overview
- **Environment ID**: `openenv-echo`
- **Short description**: OpenEnv Echo environment via `OpenEnvEnv`, demonstrating MCP tool-calling in Prime Sandboxes.
- **Tags**: openenv, mcp, tools, example

### Datasets
- **Primary dataset(s)**: Seed-generated episodes (one seed per rollout).
- **Source links**: Bundled OpenEnv Echo project in `openenv_project/` (copied from OpenEnv).
- **Split sizes**: 1000 train / 100 eval by default (configurable).

### Task
- **Type**: Tool use, multi-turn.
- **Parser**: Default `Parser` (no special formatting).
- **Rubric overview**: `OpenEnvEpisodicSumRubric` sums per-step rewards; `MultiTurnMonitorRubric` tracks turn count.

### Quickstart
Build and register the bundled OpenEnv Docker image in the Prime registry:

```bash
vf-openenv-build \
  --path environments/openenv_echo/openenv_project \
  --image openenv-echo:latest
```

This writes `environments/openenv_echo/openenv_project/.openenv_image` with the fully qualified image reference from `prime images list`.

Verify the image is ready (status **Ready**):

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
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"openenv_project": "environments/openenv_echo/openenv_project"}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.
- If you customize the bundled OpenEnv project, rebuild the image with the same `--image` value (the `.openenv_image` marker is updated for you).

### Troubleshooting

If you see errors like `waiting to start: trying and failing to pull image`, it means the image is not available to the sandbox. Common causes:
- The image build is still running or failed (`prime images list` should show **Ready**).
- The image reference in `.openenv_image` is not the fully qualified image from `prime images list`.
- The image is private or not accessible to your team.

If `prime images list` shows **Ready** but the sandbox still cannot pull the image, escalate to the platform team with:
- Image name/tag
- Build status/output from `prime images list`
- Sandbox ID and timestamp from the error log

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `openenv_project` | str | `"environments/openenv_echo/openenv_project"` | OpenEnv project path, git URL, or Hugging Face space. |
| `num_train_examples` | int | `1000` | Number of training seeds to generate. |
| `num_eval_examples` | int | `100` | Number of eval seeds to generate. |
| `seed` | int | `0` | Base seed for episode generation. |
| `max_turns` | int | `-1` | Max turns per rollout (`-1` = no limit). |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Sum of per-step rewards from the OpenEnv environment. |
| `num_turns` | Number of turns taken in the rollout. |
