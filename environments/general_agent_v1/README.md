# general-agent-v1 (solver)

Multi-turn tool-use tasks from the self-growing **general-agent** toolbench. Each task ships its own
`tools.py` (a `TaskDB` world + `@tool` methods that mutate it) and a gold tool-call chain. The agent
gets the task's instruction and its tools (served per rollout over MCP by a `Toolset` that loads the
task's `tools.py` dynamically); the reward replays the gold chain and rewards an exact final-DB-hash
match **or** a passing `verify(db)`.

The **4,417-task corpus is not vendored** — `corpus.ensure_corpus()` sparse-checks-out `tasks/` from
`PrimeIntellect-ai/research-environments` at a pinned commit into `~/.cache/verifiers/general_agent/`
on first use (needs read access to that repo; override the location with `GENERAL_AGENT_CACHE_DIR`).

Run under any MCP-tool-capable harness (`bash`, `default`):

```bash
uv run eval general-agent-v1 -n 1 -r 3 --taskset.tasks calendar_scheduling_t0 \
  --harness.id bash -m gpt-5-mini
```

Filter the corpus with `--taskset.tasks <task|family> ...`, `--taskset.min-tier` / `--taskset.max-tier`,
or a recorded pass-rate band (`--taskset.min-pass-rate` / `--taskset.max-pass-rate`). Model-free gold
check: `uv run validate general-agent-v1 --runtime.type subprocess -n 5`.
