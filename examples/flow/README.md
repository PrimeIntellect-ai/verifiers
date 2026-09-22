# Flow

A `Flow` subclass owns shared services and scheduling policy. Each `Unit` is work that can
be paused or resumed independently. A `@stage` method receives that unit and returns a
`Transition`: its next stage, status, and optional typed data/files, committed together in Git.
`setup()` runs on every launch, including resume. Persistent changes must be safe to repeat;
`create_unit()` preserves existing checkpoints.

Start with these short, runnable plugins:

- [single_agent.py](../../verifiers/v1/flows/single_agent.py): one agent per prompt.
- [parallel.py](../../verifiers/v1/flows/parallel.py): concurrent agents; retain successes if a sibling fails.
- [draft_review.py](../../verifiers/v1/flows/draft_review.py): two stages exchange a Git artifact revision.
- [partial_calls.py](../../verifiers/v1/flows/partial_calls.py): try partial failure and resume entirely offline.

Export exactly one Flow subclass in the installed package's `__all__`. Its `Flow[Config]`
specialization selects its Pydantic config. Prime-RL handles launch paths and logging:

```sh
uv run flow --flow.id single-agent --flow.model MODEL --run.name answers
uv run flow --flow.id partial-calls --run.name demo --dashboard false
uv run flow inspect --root outputs/demo
uv run flow steer --root outputs/demo --unit task --status ready
uv run flow --flow.id partial-calls --run.name demo --flow.available true --dashboard false
```

The offline example executes eight calls on its first launch, then only the two missing
results on the second. Its `available` flag describes an outage, so it is deliberately absent
from reuse inputs. Real pipelines declare their own meaningful task/model/grading inputs.

## Calls and files

`self.agents.solver.run(task, key=..., inputs=...)` uses native agent execution, retries and
traces. Without a key it runs every time. A key and explicit inputs reuse successful results
from disk; failures remain retryable. Agent `run()` returns a trace or raises; `attempt()`
returns success or failure. Host functions use `self.call(func, ..., output=ResultType)`
or `self.attempt(...)` with the same contract.

`self.gather(...)` waits for every child before propagating failures and settles children
on cancellation. Use `attempt()` to inspect individual outcomes, or `call()`/agent `run()`
when failure should raise. Successful keyed calls remain reusable either way.

Native Agent owns retries. Flow saves only the final trace (or the current partial trace on
cancellation), and token totals count only that trace. The live snapshot uses one path per
call, switching to each new trace and disappearing when the call ends.

For custom turn loops, pass an async `interact(interaction)` function to the agent's `run`
or `attempt`. It drives native `Interaction.turn`; the complete interaction is one recorded
call. To share a sandbox, use `async with solver.provision(task) as box`, then pass
`runtime=box` to agents that borrow it. The caller owns that box's lifetime.

Reuse restores a value or trace, **never sandbox side effects**. `GitArtifacts(unit)`
optionally preserves work products independently of workflow Git checkpoints; publish the
chosen revision in `Transition(data=unit.data)`. External effects before returning a
transition are not rolled back by a hold.

## Agent control

A monitoring coding agent reads committed unit state, `transitions.jsonl`, call records and
traces (including `live/` snapshots). `inspect` exposes current state and executing stages.
`steer` changes the next stage/status or appends a note; it does not interrupt a model's
conversation. Live controls survive stage publication. `unit.before` and `unit.notes` are
stage-start snapshots; `unit.state()` reads current committed state. Publish edits to the
stage's `unit.data` explicitly with a transition. Operator data edits require a settled unit
and its inspected revision (`--data patch.json --expected SHA`).

`admit(unit)` sees accepted executions in `self.active`, including their original executing
stage after a live route. Pipelines define barriers and success policy; `run()` returns unit
states and a quiescent/draining reason. Use `flow.stay_alive = true` to wait for new work.

`flow drain --root outputs/demo` finishes running calls and stops new work. Remove the
`drain` file to launch again. Ctrl-C drains once and cancels on a second signal.
`pools.json` holds the current named limits; replace it atomically to resize them:

```sh
printf '%s\n' '{"units": 2, "runtimes": 4}' > outputs/demo/pools.json.tmp
mv outputs/demo/pools.json.tmp outputs/demo/pools.json
```

Flow checks it about every two seconds. Keep the same pool names and positive integer
limits. Lowering a limit lets accepted work finish. Invalid edits leave the previous limits
in effect and log the error; recovery decisions remain with the monitor.
