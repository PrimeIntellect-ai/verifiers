# mcp-search-env

<a href="https://github.com/PrimeIntellect-ai/verifiers/tree/main/environments/mcp_env">
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="Source Code">
</a>

### Overview

- **Environment ID**: `mcp-search-env`
- **Short description**: Example environment using `vf.MCPEnv` for MCP server integration
- **Tags**: MCP, Tools

This environment demonstrates MCP server integration. The v1 variant uses a
package-local documentation server backed by markdown files bundled under
`docs/`, so installed environment packages do not need access to the source
repository checkout.

### Datasets

- **Primary dataset(s)**: bundled Verifiers documentation snippets
- **Source links**: package-local `docs/*.md`
- **Split sizes**: 10 v1 examples

### Task

- **Type**: <multi-turn | tool use>
- **Rubric overview**: N/A

### Quickstart

Run an evaluation with default settings:

```bash
prime eval run mcp-search-env
```

Configure model and sampling:

```bash
prime eval run mcp-search-env   -m openai/gpt-4.1-mini   -n 1 -r 1
```

Notes:

- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments

Document any supported environment arguments and their meaning. Example:

| Arg            | Type | Default | Description                            |
| -------------- | ---- | ------- | -------------------------------------- |
| `max_examples` | int  | `-1`    | Limit on dataset size (use -1 for all) |

### Metrics

| Metric         | Meaning                                       |
| -------------- | --------------------------------------------- |
| `reward`       | Main scalar reward (weighted sum of criteria) |
| `judge_reward` | LLM judge score (1.0 if correct, 0.0 if not)  |
