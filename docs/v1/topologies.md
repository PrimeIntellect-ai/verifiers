# Topologies — multi-agent environments

A **topology** composes multiple agents over episodes: which agents exist, how one agent's
trace becomes the next agent's task, and how rewards flow backwards once downstream agents
have run. Each *episode* is an ordinary rollout — one agent consuming one task and producing
one trace — and tasks carry their own behavior, so nothing about a task or a harness is
topology-specific. Run one with `--topology.id` (it replaces the eval's own
`taskset` × `harness` pair):

```bash
uv run eval --topology.id llm-judge --topology.taskset.id gsm8k-v1 -n 4
uv run eval --topology.id proposer-solver-v1 -n 3
```

## Agents

An agent is **pure routing** — a name + the harness driving its episodes + how its model calls
are routed; it carries nothing task-side. Declare agents as typed `AgentConfig` fields on your
config — the field *name* is the agent's name, and every agent is CLI-addressable
(`--topology.solver.harness.id rlm`, `--topology.judge.model <id>`):

```python
class ProposerSolverConfig(vf.TopologyConfig):
    proposer: vf.NullAgentConfig = vf.NullAgentConfig()      # `null` chat loop (has MCP tools)
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
arrive per episode, from the topology's seeds or constructed in `go`.

An `AgentConfig` field is the one declaration form. The default `load_agents` builds one
`AgentBinding` per field, in declaration order; override it only to compose agents
programmatically. At serving time those bindings become executable `vf.Agent`s with the
active model context and shared run services. Loading also validates the topology's declared
judgement (`@reward(agent=...)` / `@metric(agent=...)`) against the agents, so a typo'd or
missing agent scope fails at load time, not mid-eval.

## Seed tasks

One topology instance runs per seed task (× `-r`). Seeds come from the config's `tasks`
factory — any taskset, plugged by id (`--topology.taskset.id gsm8k-v1`; its knobs validate
typed, e.g. `--topology.taskset.split train`) — or from a `load_tasks` override for a
self-seeding topology. **Exclusive-or, enforced at load**: when the slot can be set it IS
the seed source, verbatim; a topology that overrides `load_tasks` is refused the flag
(rather than silently ignoring it), and a custom `load_tasks` wanting a config-driven
source declares its own factory field:

```python
class ProposerSolverTopology(vf.Topology[ProposerSolverConfig]):
    def load_tasks(self) -> list[vf.Task]:
        """Self-seeding: the references are baked in, so no `--topology.taskset.id` needed."""
        return [
            ProposeTask(idx=i, prompt=PROPOSE_PROMPT.format(reference=reference))
            for i, reference in enumerate(REFERENCES)
        ]
```

Per-role behavior lives on **task classes**, minted anywhere. In `proposer-solver-v1`,
`ProposeTask` judges its own episode (a format reward), and the `SolverTask` built mid-`go`
carries the ground truth *and* the `correct` reward — question and verifier in one typed
object, serialized with each solver trace so the record shows exactly what was asked:

```python
class ProposeTask(vf.Task):
    @vf.reward(weight=0.1)
    async def well_formed(self, trace: vf.Trace) -> float:
        answer = parse_number(parse_labeled(trace, "ANSWER") or "")
        return float(bool(parse_labeled(trace, "QUESTION")) and answer is not None)


class SolverTask(vf.Task):
    answer: str   # the proposer's canonical numeric answer

    @vf.reward
    async def correct(self, trace: vf.Trace) -> float:
        return float(parse_number(parse_labeled(trace, "ANSWER") or "") == self.answer)
```

## The interaction pattern — `go`

`go` is plain imperative Python over a `TopologyRun`; interaction patterns are code, not a
DSL. `run.agent(name).run(task, parents=...)` runs one episode and links it into the
agent graph; `asyncio.gather` fans out; loops are rounds; awaiting several traces before
building the next task is fan-in:

```python
    async def go(self, task: vf.Task, run: vf.TopologyRun) -> None:
        proposer = await run.agent("proposer").run(task)
        # Forward arrow: read the proposal straight off the trace, pure host-side.
        question = parse_labeled(proposer, "QUESTION")
        answer = parse_number(parse_labeled(proposer, "ANSWER") or "")
        if not question or answer is None:
            return  # malformed proposal — `well_formed` scored it; nothing to solve
        derived = SolverTask(idx=task.data.idx, prompt=SOLVE_PROMPT.format(question=question), answer=answer)
        solver = run.agent("solver")
        await asyncio.gather(
            *(
                solver.run(derived, parents=[proposer])
                for _ in range(self.config.num_solvers)
            )
        )
```

## Sessions — agents talking *within* each other's episodes

`run(task)` composes **completed** episodes: an episode finishes before its trace feeds
anything downstream. For back-and-forth interaction — chess, negotiation, debate, any
game where each agent is effectively the other's user — hold episodes **open** with
`run.agent(name).interact(task)` and converse with the yielded `Session`:

```python
    async def go(self, task: vf.Task, run: vf.TopologyRun) -> None:
        board = chess.Board()                       # game rules live HERE, host-side
        async with (
            run.agent("white").interact(SeatTask(SeatData(color="white", prompt=None, system_prompt=...))) as white,
            run.agent("black").interact(SeatTask(SeatData(color="black", prompt=None, system_prompt=...))) as black,
        ):
            seats = {chess.WHITE: white, chess.BLACK: black}
            while not board.is_game_over():
                reply = await seats[board.turn].turn(render(board))   # user turn in, model turn out
                board.push(parse_move(reply, board))
        # scope exit ends both episodes cleanly (stop -> finalize -> task scoring);
        # stamp the outcome as data for the declared rewards to read:
        white.trace.info["chess"] = {"score": ...}
```

A `Session` has exactly three members: `turn(message) -> reply` (send the episode its
next user turn, get the model's turn back), `end()` (finish early — forfeits,
eliminations; idempotent, scope exit calls it), and `.trace` (the live trace: read
mid-game `state`, stamp outcome `info`). Everything else — who talks to whom, in what
order, with what *view* of the interaction — is imperative code in `go`, so N seats
compose into round-robins, simultaneous moves (`asyncio.gather` over several `turn`s),
moderated rooms, and hidden-information games with no further machinery. Each seat's
trace is ONE multi-turn trajectory with its counterparts' messages as user turns — the
training-sample shape self-play wants, not one episode per move.

Mechanically a session is a user simulator without the server: the interception layer
suspends the episode between turns awaiting the user seat, and here the "user" is `go`.
The safety contract is loud, not patient: `turn()` on a dead episode raises
`SessionEnded` (carrying the trace) instead of hanging; a second concurrent `turn()` on
one session is refused; a task that declares its own user simulator (`Task.user`) — or a prompted task
(sessions open on the first `turn()`; put framing in `system_prompt`), or a harness that
can't take injected user turns — is refused at the `interact()` call. No `retry=`: one
side of a half-played game can't be re-run; a dead seat is `go`'s decision. Budgets
(`--max-turns`, rollout timeout) span the whole interaction, including time a seat
spends suspended — size them for the game. Scripted counterparts stay `vf.User`
simulators (route 1, no topology needed); sessions are for counterparts that are
themselves agents. See `chess-v1` (two seats, host-side referee) and `debate-v1` (N
concurrent seats of one agent config, peer-voted).

## Topology rewards — declared, cross-agent judgement

Per-episode judgement rides on the task classes (`SolverTask.correct` above). Cross-agent
judgement is *not* written inline in `go`: declare it as `@vf.reward(agent=...)` /
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
  episode ended) and, in the rewards phase, any metric — but topology rewards must not
  read each other; derive shared inputs from the traces or a metric.
- **Failures**: a raise during instance scoring is classified `TopologyError` and recorded
  on the graph — episodes stay as data, siblings unaffected.

`trace.record_reward(...)` inside `go` still works as the escape hatch for exotic shapes
(e.g. a mid-round adjustment), but the declared methods are the norm.

The forward arrow stays in `go`: construct the downstream agent's typed `Task` from an
upstream trace — its typed task, `last_reply`, `transcript`, or `trace.info`. This is pure
host-side code; only when peeling requires the episode's *live runtime* (scraping files,
running a build) does the upstream task class need a `finalize` hook, which runs before
teardown and parks results in `trace.info`. The backward arrow — cross-agent rewards — is
declared, not inlined (above).

Episode failures never raise into `go` — they come back as data on the trace
(`trace.has_error`), and `go` decides what a failed child means (drop it, count it against a
pass rate, retry the round). A crash in `go` itself is recorded on the instance's graph as a
`TopologyError` and doesn't touch sibling instances.

## The agent graph

Running one instance produces an `AgentGraph` — the serialized instance artifact
`{id, topology, error, traces[]}`: every episode's trace in completion (= topological) order,
each stamped with `trace.agent`, `trace.parents` (upstream trace ids), and `trace.trainable`.
A topology run persists **one instance record per `traces.jsonl` line**, traces nested (an
instance's rewards are only final once the whole instance is done); `AgentGraph.load(dict)`
reads one back without the originating packages (task-specific fields ride in
`task.model_extra`). The links themselves are plain `Trace` fields, so the graph also
reconstructs from any flat trace dump (one instance = one connected component). Navigation —
`graph.roots()` / `graph.children(trace, agent=...)` / `graph.by_agent(name)` — is what
cross-agent scoring lives on, and `graph.error` records a crash in `go` itself. A `Trace` is
the per-agent view of one episode; the agent graph is the global view of the interaction.

## The built-in judge topologies

Judging as a config-only pattern, two tiers, one verdict contract (a `SCORE: <0-10>` line,
recorded on the *solver's* trace as a weighted reward, `--topology.weight`):

- **`llm-judge`** — a `solver` (any taskset, via `--topology.taskset.id`) and a non-trainable
  `judge`, **fixed** to the in-process `direct` harness (an episode ≈ one API call). `go`
  peels the judge's inputs off the finished solver episode — the seed task's framing, its
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

Not yet supported under a topology: `--server` (env-server serving), `--resume`, and the
`--rich` dashboard.

---

