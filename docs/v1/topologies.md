# Topologies — multi-agent environments

A **topology** composes agent invocations: which agents exist, how one agent's trace becomes
the next agent's task, and how rewards flow backwards once downstream agents have run. Each
agent invocation consumes one task and produces one trace. Every environment invocation,
including taskset × harness syntax lowered to the built-in single-agent topology, produces
one `AgentGraph`. Run an explicit topology with `--topology.id`:

```bash
uv run eval --topology.id llm-judge --topology.taskset.id gsm8k-v1 -n 4
uv run eval --topology.id proposer-solver-v1 -n 3
```

## Agents

An agent is **pure routing** — a name + the harness driving its runs + how its model calls
are routed; it carries nothing task-side. Declare agents as typed `AgentConfig` fields on your
config — the field *name* is the agent's name, and every agent is CLI-addressable
(`--topology.solver.harness.id rlm`, `--topology.judge.model <id>`):

```python
class ProposerSolverConfig(vf.TopologyConfig):
    proposer: vf.AgentConfig = vf.AgentConfig()              # default: bash + edit + task tools
    solver: vf.DirectAgentConfig = vf.DirectAgentConfig()    # in-process `direct` (tool-less)
    num_solvers: int = 4
```

An `AgentConfig` binds a harness (and where it runs — `harness.runtime`) plus per-agent
routing: `model` / `client` / `sampling` overrides and a `trainable` flag (stamped onto every
trace the agent produces, so a trainer can drop e.g. judge traces without the topology
config). Subclass it to pin typed per-agent defaults — `vf.DirectAgentConfig` /
`vf.NullAgentConfig` are the shared common pins, and the `llm-judge` topology's judge pins
the `direct` harness plus `trainable=False` (a pin must live on the subclass's field default,
e.g. `harness: SerializeAsAny[vf.HarnessConfig] = vf.HarnessConfig(id="direct")`, so partial
overrides deep-merge into it). The tasks an agent consumes (each carrying its own behavior)
arrive per invocation, from the topology's seeds or constructed in `run`.

An `AgentConfig` field is the one declaration form, and a `list[AgentConfig]` field
declares one **role with several seats** (`editors: list[vf.DirectAgentConfig] = [...]`) —
every seat shares the role name on its traces, and per-seat config makes mixed-model
line-ups a TOML change (`[[topology.editors]]` array-of-tables, or
`--topology.editors.<i>.model` on the CLI). The framework builds one executable
`vf.Agent` per declared field for every instance — with the run's model context, serving
resources, budgets, and graph recording already bound — and hands them to `run` as an
`Agents` namespace mirroring the config (`agents.judge`, `agents.editors[i]`). Loading
also validates the topology's declared judgement (`@reward(agent=...)` /
`@metric(agent=...)`) against the declared names, so a typo'd or missing agent scope
fails at load time, not mid-eval.

## Seed tasks

One topology instance runs per seed task (× `-r`). Seeds come from the config's `taskset`
slot — any taskset, plugged by id (`--topology.taskset.id gsm8k-v1`; its knobs validate
typed, e.g. `--topology.taskset.split train`; the slot is optional and defaults to unset) —
or from a `load_tasks` override for a self-seeding topology. **Exclusive-or, enforced at
load**: when the slot can be set it IS the seed source, verbatim; a topology that overrides
`load_tasks` is refused the flag (rather than silently ignoring it), and a custom
`load_tasks` wanting a config-driven source declares its own factory field. A
start-from-nothing topology (no outside data at all) returns identity-only stubs
(`vf.Task(vf.TaskData(idx=i))`) — the seed is each instance's identity for `-n`, resume,
and dispatch, not necessarily content. A topology may also pin a specific taskset, as
`proposer-solver-v1` pins AIME 2026:

```python
class ProposerSolverTopology(vf.Topology[ProposerSolverConfig]):
    def load_tasks(self) -> list[vf.Task]:
        config_type = vf.taskset_config_type("aime26-v1")
        return vf.load_taskset(config_type(id="aime26-v1")).load()
```

Per-role behavior lives on **task classes**, minted anywhere. In `proposer-solver-v1`,
`ProposeTask` judges its own trace with a format reward. The solver task is derived from
the AIME seed by replacing its prompt and answer while retaining its task class and config,
so the modified problem keeps AIME's math-verify reward:

```python
class ProposeTask(vf.Task):
    tools = (SubmitToolset,)

    @vf.reward(weight=0.1)
    async def parseable(self, trace: vf.Trace) -> float:
        return float("submission" in trace.info)


def solver_task(seed: vf.Task, trace: vf.Trace) -> vf.Task:
    submission = trace.info["submission"]
    data = seed.data.model_copy(update={
        "prompt": AIME_INSTRUCTION + submission["question"],
        "answer": submission["answer"],
    })
    return type(seed)(data, seed.config)
```

## The interaction pattern — `run`

`run` is plain imperative Python over the framework-built agents; interaction patterns are
code, not a DSL. `agents.<name>.run(task, parents=...)` runs one agent invocation — its
trace records onto the instance's agent graph automatically, and `parents=` names the
lineage right where the derived task is built; `asyncio.gather` fans out; loops are
rounds; awaiting several traces before building the next task is fan-in:

```python
    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        proposer = await agents.proposer.run(ProposeTask.from_task(task))
        if "submission" not in proposer.info:
            return
        derived = self.solver_task(task, proposer)
        await asyncio.gather(
            *(
                agents.solver.run(derived, parents=[proposer])
                for _ in range(self.config.num_solvers)
            )
        )
```

To share one live runtime across several runs, `provision()` it and pass the box
explicitly (`async with agents.solver.provision(task) as box: await
agents.solver.run(task, runtime=box)` — see `shared-runtime-v1`). A topology that owns
shared resources of its own overrides the `setup()` / `teardown()` hooks, which the
runner calls around the whole serving lifetime.

## Topology rewards — declared, cross-agent judgement

Per-trace judgement rides on task classes (the derived AIME task's `correct` reward above). Cross-agent
judgement is *not* written inline in `run`: declare it as `@vf.reward(agent=...)` /
`@vf.metric(agent=...)` methods on the topology — the same decorators tasks use, scoped
to an agent. Each runs once per matching trace **after the whole instance completes**, with
any of `task` / `trace` / `graph` injected by parameter name, and records under the method
name (weighted) exactly like a task reward:

```python
    @vf.metric(agent="proposer")
    async def solve_rate(self, trace: vf.Trace, graph: vf.AgentGraph) -> float:
        graded = [t for t in graph.children(trace, agent="solver") if not t.has_error]
        return sum(t.rewards.get("correct", 0.0) for t in graded) / len(graded) if graded else 0.0

    @vf.reward(agent="proposer")
    async def difficulty(self, trace: vf.Trace) -> float:
        return 1.0 - 2.0 * abs(trace.metrics["solve_rate"] - 0.5)   # rewards may read metrics
```

The contract, chosen to fail loudly and stay predictable:

- **Validated at load**: an `agent=` scope that doesn't exist — or a topology `@reward`
  with no scope — is refused when the topology loads, before anything runs.
- **Runs at instance end**, never earlier: nothing can observe a reward before the
  instance persists, so there is no "earliest possible" scheduling to reason about. Every
  trace in scope is scored — across all rounds and fan-outs — automatically.
- **Ordering**: methods run sequentially, metrics before rewards, each phase in
  (priority, name) order. A method may read task-recorded rewards (final since the
  agent run ended) and, in the rewards phase, any metric — but topology rewards must not
  read each other; derive shared inputs from the traces or a metric.
- **Failures**: a raise during instance scoring is classified `TopologyError` and recorded
  on the graph — completed traces stay as data, siblings unaffected.

`trace.record_reward(...)` inside `run` still works as the escape hatch for exotic shapes
(e.g. a mid-round adjustment), but the declared methods are the norm.

The forward arrow stays in `run`: construct the downstream agent's typed `Task` from an
upstream trace — its typed task, `last_reply`, `transcript`, or `trace.info`. This is pure
host-side code; only when peeling requires the agent's *live runtime* (scraping files,
running a build) does the upstream task class need a `finalize` hook, which runs before
teardown and parks results in `trace.info`. The backward arrow — cross-agent rewards — is
declared, not inlined (above).

Agent failures never raise into `run` — they come back as data on the trace
(`trace.has_error`), and `run` decides what a failed child means (drop it, count it against a
pass rate, retry the round). A crash in `run` itself is recorded on the instance's graph as a
`TopologyError` and doesn't touch sibling instances — traces already completed were
recorded as their agents finished, so nothing paid-for is lost.

## The agent graph

Running one instance produces an `AgentGraph` — the serialized instance artifact
`{id, topology, task, error, traces[]}`: the seed task plus every agent trace in completion order,
each stamped with `trace.agent`, `trace.parents` (upstream trace ids), and `trace.trainable`.
A topology run persists **one instance record per `traces.jsonl` line**, traces nested (an
instance's rewards are only final once the whole instance is done); `AgentGraph.load(dict)`
reads one back without the originating packages (task-specific fields ride in
`task.model_extra`). The links themselves are plain `Trace` fields, so the graph also
reconstructs from any flat trace dump (one instance = one connected component). Navigation —
`graph.roots()` / `graph.children(trace, agent=...)` / `graph.by_agent(name)` — is what
cross-agent scoring lives on, and `graph.error` records a crash in `run` itself. A `Trace` is
the per-agent view of one run; the agent graph is the global view of the interaction.

## The built-in judge topologies

Judging as a config-only pattern, two tiers, one verdict contract (a `SCORE: <0-10>` line,
recorded on the *solver's* trace as a weighted reward, `--topology.weight`):

- **`llm-judge`** — a `solver` (any taskset, via `--topology.taskset.id`) and a non-trainable
  `judge`, **fixed** to the in-process `direct` harness (a run ≈ one API call). `run`
  peels the judge's inputs off the finished solver trace — the seed task's framing, its
  ground truth (an `answer` field, when the taskset carries one), and the solver's final
  message — into a `JudgeTask` minted from the trace. Give the judge its own model
  (`--topology.judge.model`) or client routing; swapping its harness is refused and points
  here:
- **`agentic-judge`** — same shape, but the judge is a real agent: the solver's **entire
  serialized trace** is uploaded into the judge's own runtime (by the judge task's `setup`
  hook), and the judge investigates it with its tools before committing. Its harness is
  configurable (`--topology.judge.harness.id ...`, bash+edit `default` by default), as is its
  assignment (`--topology.prompt`, with `{path}` = where the trace landed).

For a verdict baked into a task's own grading, call `vf.Judge` from the task's
`@reward` instead — that's the cheap utility tier; these topologies are the tier where the
judge is itself an agent.

Topologies use the same in-process runner, env-server worker pool, resume behavior, and
dashboard path as taskset × harness configurations.

---
