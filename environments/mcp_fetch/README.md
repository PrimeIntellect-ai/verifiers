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
- **Tasks** – `tasks/qa.jsonl` now includes 84 prompts covering direct lookups,
  multi-step pointer puzzles, ledger math, and short-form judge summaries. The
  latest gauntlet (IDs `fetch_065`–`fetch_084`) leans hard on planner/workflow
  chains and poem character counts to separate small vs. strong models. Each
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

When invoking the Runner directly (outside `uv run`), make sure `PYTHONPATH`
includes both the repo root and `environments/` so `environments.mcp_fetch`
resolves correctly:

```bash
PYTHONPATH=.:environments vf-eval mcp-fetch -n 10 -m gpt-4.1-mini
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

## Why non-trivial?

The latest calibration run keeps smaller models honest while leaving headroom
for GPT-5–class evaluators. Scores are averaged over the full 84-task suite:

| Model | Correct / Total | Accuracy |
| --- | --- | --- |
| `gpt-4.1-mini` | 46 / 84 | **54.8%** |
| `gpt-4.1` | 36 / 46 | **78.3%** |
| `gpt-5` | 68 / 84 | **80.95%** |

Mini models routinely stall on the planner/workflow gauntlet and poem
character-count checks, while GPT-4.1 clears most but not all retries and
rubric-graded prompts. GPT-5 still needs deliberate tool planning to stay above
80%, which is exactly the intended difficulty band for the MCP Agents bounty.

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

## API keys & GPT-5 diagnostic

The calibration script and the GPT-5 sanity-check test automatically load envvars
from `environments/mcp_fetch/.env` (and the repo-level `.env`, if present) via
`python-dotenv`. Add your OpenAI key to the existing gitignored file:

```bash
echo "OPENAI_API_KEY=sk-..." >> environments/mcp_fetch/.env
```

Because the file never leaves your machine, secrets stay local while still being
picked up automatically by both scripts. Once populated you can run:

```bash
PYTHONPATH=.:environments .venv/bin/pytest environments/mcp_fetch/tests/test_gpt5_tool_call.py -s
PYTHONPATH=.:environments .venv/bin/python environments/mcp_fetch/scripts/calibrate_questions.py --model gpt-5 --max-turns 6
```

The pytest harness mirrors OpenAI’s Responses tool loop and fails fast if GPT-5
doesn’t return the expected “Mini Site” H1, which makes debugging empty-answer
cases easier.

## Calibration cadence

Run a single pass per reference model so token usage stays bounded while the new
gauntlet still lands:

```bash
PYTHONPATH=.:environments .venv/bin/python environments/mcp_fetch/scripts/calibrate_questions.py --model gpt-4.1-mini --max-turns 6 --include-judge
PYTHONPATH=.:environments .venv/bin/python environments/mcp_fetch/scripts/calibrate_questions.py --model gpt-5 --max-turns 6 --include-judge
```

Review `environments/mcp_fetch/reports/question_quality_<model>.json` after each
run; planner/workflow and poem-char categories should show the widest gap. If
mini creeps above ~50% again, extend the gauntlet with additional planner or
character-count prompts before re-running these commands.
