# `mcp-fetch`

Deterministic MCP environment that exposes a single `fetch` tool wired through the
shared `verifiers.envs.mcp_env.MCPEnv` wrapper. The tool talks to a local stdio MCP
server (`tools/fetch_mcp_server.py`) which in turn can only reach the fixtures
hosted by `utils/mini_httpd.py` unless explicitly configured for online hosts.

Key components:

- **Offline fixtures** – `utils/mini_httpd.py` serves HTML/JSON/text, redirects,
  auth checks, query endpoints, and gzipped responses at `http://127.0.0.1:31415`.
- **MCP server** – `tools/fetch_mcp_server.py` doubles as a CLI (`--url ...`) and
  an MCP stdio process (`--run-server`) returning structured JSON and plaintext.
- **Tasks** – `tasks/qa.jsonl` now includes 40 prompts covering direct lookups,
  multi-step pointer puzzles, ledger math, and short-form judge summaries. Each
  row includes metadata and a verifier definition used by scripts/tests.
- **Judge rubrics** – `tasks/judge_rubrics.yaml` defines four LLM-graded
  summaries (poem, fruits, ledger, manifest).

The expanded fixture set introduces chained lookups (pointer → directive → HTML),
numeric reasoning over the ledger JSON, and rubric-graded summarization to ensure
frontier models have to plan multiple tool calls instead of memorising a single
endpoint.

## Installation

```bash
cd environments/mcp_fetch
uv pip install -e .
```

This registers two console scripts:

- `mcp-server-fetch` – stdio MCP server used by the environment.
- `mcp-fetch-mini-httpd` – helper for manually serving fixtures.

## Running the environment

```bash
uv run vf-eval mcp-fetch -n 5 -m gpt-4.1-mini
```

Arguments exposed via `load_environment(...)`:

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `allow_online` | bool | `False` | Allow a curated list of public hosts in addition to localhost fixtures. |
| `allow_any_host` | bool | `False` | Disable allowlist enforcement entirely (use with caution). |
| `allowed_hosts` | Iterable[str] | `None` | Custom host allowlist (`host` or `host:port`). Overrides both flags above. |
| `server_cmd` | str \| Sequence[str] | `None` | Override the `mcp-server-fetch` launch command. |
| `fixture_port` | int | `31415` | Port for the deterministic fixture host. |
| `auto_start_fixtures` | bool | `True` | Automatically launch the mini HTTP daemon (skipped when online mode is enabled). |
| `task_path` | str \| Path | internal | Path to the QA JSONL file (swap for ablations). |

## Testing

Offline verification mirrors the PI requirements:

```bash
uv run pytest tests/environments/test_mcp_fetch.py -q
uv run ruff check environments/mcp_fetch
```

The pytest suite spins up the fixtures, drives the MCP server via the same helper
functions the environment uses, and asserts each canonical verifier passes. This
ensures regressions in the HTTP fixtures, hashing, or truncation metadata are
caught during CI.
