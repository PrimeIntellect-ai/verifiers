# Agent programs

Live-tested example programs for the `vf.Agent` facade (see `docs/agent-programs.md`).
Each is a plain script: `PRIME_API_KEY` in the environment, then

```bash
uv run python examples/agent_programs/smoke.py            # subprocess, ~5s, <$0.01
uv run python examples/agent_programs/proposer_solver.py  # subprocess, ~1min, <$0.01
uv run python examples/agent_programs/same_box_judge.py   # one prime sandbox, ~3min, <$0.01 + sandbox time
uv run python examples/agent_programs/world_verified.py   # one prime sandbox, ~3min, <$0.01 + sandbox time
```

- `smoke.py` — the primitive: one agent, one task, one trace.
- `same_box_judge.py` — provision a sandbox, run the solver in it (borrowed), write its
  trace into the box, place the judge into the *same* box to audit world + trajectory.
- `proposer_solver.py` — proposer mints a task (lineage-stamped), solvers fan out via
  plain `asyncio.gather` with a taskset attached (`@reward` per rollout).
- `world_verified.py` — a third-party harness (`mini-swe-agent`) plus a runtime-aware
  `@reward` that re-executes the agent's artifact in the live box.
