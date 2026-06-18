# general-agent-synth-v1

The synthesizer half of the general-agent loop (synth → solve): a coding agent authors a **new**
tool-use task for [`general-agent-v1`](../general_agent_v1). Each rollout asks the agent (the `bash`
harness, in a container) to write a task under `out/<name>/` — its `tools.py` world + tools, an
initial `db.json`, an `instruction.md`, and a gold `gold.json` chain. `setup` stages a small
`general_agent` shim + a `validate_task.py` self-check into the container so the agent iterates to a
valid task; the reward then pulls the produced task back out and gold-checks it (the same
`gold_check` the solver's `validate` hook uses).

No pass-rate gating (that needs running the solver per tier — see the source env); this rewards a
structurally-valid new task. Needs a container:

```bash
uv run eval general-agent-synth-v1 -n 1 \
  --harness.id bash --harness.runtime.type docker -m gpt-5-mini
```
