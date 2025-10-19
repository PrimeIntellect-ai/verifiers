# arc-agi-3

### Overview

- **Environment ID**: `arc-agi-3`
- **Short description**: ARC-AGI-3 sandbox that proxies the official competition API through the Verifiers agent loop.
- **Tags**: ARC-AGI, tools, multi-turn

### Datasets

- **Primary dataset(s)**: Live ARC-AGI-3 games chosen via `game_id`.
- **Source links**: [ARC Prize](https://three.arcprize.org/)
- **Split sizes**: N/A (games are fetched on demand from the official API).

### Task

- **Type**: multi-turn tool use
- **Parser**: JSON-only responses (custom extractor)
- **Rubric overview**: single reward that awards `1.0` when the run finishes in `WIN`, `0.0` otherwise.

### Quickstart

1. Export your competition key:

```bash
export ARC_API_KEY="sk-..."
```

2. Run an evaluation with the default starter puzzle:

```bash
uv run vf-eval arc-agi-3 -n 1 -r 1
```

3. Target specific games and tweak limits:

```bash
uv run vf-eval arc-agi-3 \
  -a '{"games": ["ls20-fa137e247ce6"], "max_actions": 60}' \
  -m gpt-4.1-mini -n 1 -r 1
```

Notes:

- The environment streams real game states from the ARC servers; runs consume live attempts.
- Actions must be emitted as JSON per the system prompt. Invalid JSON will be rejected with an error turn.
- Scorecards are closed automatically after you emit the final summary JSON.
- The environment requires the official [`arc-agi-3-agents`](https://github.com/arcprize/ARC-AGI-3-Agents) package for data models and scorecard validation.

### Environment Arguments

| Arg              | Type                      | Default                  | Description |
| ---------------- | ------------------------- | ------------------------ | ----------- |
| `games`          | list[str \| dict]         | `["ls20-fa137e247ce6"]` | Game IDs (or dicts with `game_id`, optional `prompt`, `tags`). Full game IDs like `ls20-fa137e247ce6` are required (see API docs). |
| `base_url`       | str                       | `https://three.arcprize.org`       | ARC API root (`http(s)://host[:port]`). |
| `api_key`        | str                       | `ARC_API_KEY` env        | Competition API token. Required to issue actions. |
| `max_actions`    | int                       | `80`                     | Maximum number of action turns before forcing a summary. |
| `request_timeout`| float                     | `10.0`                   | HTTP timeout (seconds) for ARC API calls. |
| `tags`           | list[str]                 | `[]`                     | Extra scorecard tags appended when opening runs. |

### Metrics

| Metric        | Meaning |
| ------------- | ------- |
| `reward`      | `1.0` if the final ARC state is `WIN`, else `0.0`. |
| `final_score` | Latest score reported by the API (mirrors scorecard). |
| `action_count`| Number of API actions submitted in the run. |

For more detail on the agent protocol and scorecard endpoints, see the official [ARC-AGI-3 Agents](https://github.com/arcprize/ARC-AGI-3-Agents) reference implementation.

