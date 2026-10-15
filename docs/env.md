# The Env

An `Env` defines the control flow between `Agents`. In the simplest case, it is just a `SingleAgentEnv` where a single agent solves a task from a taskset.

Its core signature is `Env.run(task: Task, agents: Agents) -> BaseModel | None` — it is passed an initial task and pre-initialized agents and then programs the full multi-agent control flow; every finished agent run automatically joins the resulting `Episode`, which holds all the traces of all the agents.

```python
class Env(ABC):
    @abstractmethod
    async def run(self, task: Task, agents: Agents) -> BaseModel | None:
        """Run a single multi-agent episode."""
        ...
```

The other hooks an env may override:

- `run()` may return a pydantic model, the episode's outcome: the episode keeps it dumped
  as JSON in `episode.outcome`, and `episode.outcome_as(Model)` reads it back typed. Any
  other return value, a `Trace` included, is ignored. `finalize(task, episode)` may set
  or replace it, as a JSON dict (`model.model_dump(mode="json")`).
- `trace_labels(task)` returns labels every trace of the episode carries in
  `trace.info["labels"]` (`trace.labels`), stamped at mint, before any observer sees the
  trace: a job id, a stage, a rollout index.
- An episode whose `run()` ran no agent fails, unless the env sets
  `REQUIRES_AGENT = False` (a `run()` that may decide on host work alone).

verifiers comes with different pre-built `Env`s to use:

- The `AgenticJudgeEnv` defines the sequential interaction between a solver and judge agent. The solver’s `ScoringConfig` selects the workspace used for both task scoring and agentic judging.
- The `UserSimEnv` models users as agents, and the episode is a turn-by-turn conversation between the user and assistant agents.
- The `BestOfNEnv` runs n independent attempts at the same task, then marks which attempt achieved the highest reward (best) and whether any attempt crossed a success threshold (pass_at_n), which is useful for rejection sampling and pass@k evaluation.

## Verifier placement

Task scoring placement is independent of the Env's agent program. Set
`--env.agent.scoring.mode isolated` to restore declared artifacts into a fresh
runtime before running task metrics, rewards, and judge plugins. No extra agent
or harness is started. Tasks declare defaults through `TaskData.scoring`, using
`ScoringConfig`. Explicit agent fields override task fields; unset fields inherit.
Without either, scoring is `shared` in the workspace. This also works for each
attempt in `BestOfNEnv`.

```bash
uv run vf-eval my-task --env.agent.runtime.type docker --env.agent.scoring.mode isolated
```

Task authors declare every output needed for fresh grading in `TaskData.artifacts`.
Task `setup` initializes the fresh runtime, artifacts are restored at their original
paths, and `stage_verifier` installs trusted scoring inputs before `@vf.metric`,
`@vf.reward`, and configured judge plugins run. Each retry gets an independent task
controller and trace; only successful scoring updates the solver trace.

```python
class CodeTask(vf.Task[CodeData]):
    async def stage_verifier(self, runtime: vf.Runtime) -> None:
        await runtime.write("/tmp/test_solution.sh", PRIVATE_TESTS)

    @vf.reward
    async def tests(self, runtime: vf.Runtime) -> float:
        result = await runtime.run(["bash", "/tmp/test_solution.sh"], {})
        return float(result.exit_code == 0)


task = CodeTask(
    CodeData(
        prompt="Fix the implementation.",
        scoring=vf.ScoringConfig(mode="isolated"),
        artifacts=[vf.Artifact(source="src")],
    )
)
```

`agent.scoring.runtime` optionally chooses an independent provider, image,
resources, and network policy. Otherwise the agent's provider policy and the task's
scoring requirements resolve the fresh runtime. Selecting a scoring provider without
an image preserves the task's scoring image. An explicitly supplied image overrides
it, even when equal to that provider's default image. Serialization preserves this
distinction across env-server and trace roundtrips. Solver checkpoints are not inherited.
`agent.scoring.env` merges over `TaskData.scoring.env` by key, for shared and
isolated scoring. These variables apply to scoring setup, staging, rewards, and
judging. `${VAR}` and `${VAR:-default}` values resolve from the host only when
scoring starts; serialized configs retain the templates. Scoring bindings do not
change the solver's process environment. `agent.scoring.retries` uses the same `RetryConfig` as agent and episode retries.
Its `max_retries` defaults to 2; ordered `rules` can match exception type, HTTP
status, or message, including denying a retry with `max_retries = 0`. Each retry
recreates only the isolated scoring environment and reuses the completed
submission, including rerunning any required agentic judge. A valid zero reward
does not trigger retries. Shared scoring is not retried. Runtime overrides require isolated placement. Relative artifacts require matching solver and verifier workdirs;
absolute paths support different workdirs. Fresh verification requires an isolated
filesystem and cannot use the host subprocess runtime.

A judge plugin is part of task scoring and receives the selected scoring runtime.
`AgenticJudgeEnv` runs its separate judge agent through the solver's `on_score`
callback, before the selected verifier workspace is torn down. Set
`--env.solver.scoring.mode shared|isolated`, or leave it unset to honor the task default.
Task scores and the agentic verdict share the verifier's scoring budget; isolated
retries start in fresh workspaces and commit rewards only after judging succeeds.
The judge borrows that workspace. Judge model and harness settings live under
`env.judge`; workspace placement belongs to `env.solver.scoring`.

## Durable episodes

`Env.durable(task)` returns the `vf.Durable` the episode's calls resolve through: the
trace lookup over the run's trace store, and the scope (its `match`; left None, the
episode's `trace_labels(task)`). `run_episode` installs it around `setup()` and `run()`,
so every agent call in the episode is durable, keyed by its seat and position (see
[Durable calls](agent.md#durable-calls)). Run again after a crash, the episode starts
from the top: its calls return their finished traces without running (`trace.kept`;
they are this episode's traces too, since the attempt that made them never landed),
continue their cut-off runs from the latest checkpoint, and restore the boxes the seats
share. Without `durable()`, nothing changes.

The scope is the episode's identity, so it must be unique to the episode and the same
each time it runs: two rollouts of one task need labels that tell them apart (a rollout
index the runner puts in the task), or the second takes the first's calls as its own. An
episode whose scope is empty fails, and so does one whose scope another episode running
in the same process holds.

The rule, for everything `run()` does:

- **Anything expensive or nondeterministic is an agent run**, so it has a trace: a model
  call, a long build, a judge (an agent with the null harness and the verdict in its
  reply).
- **Host code between agent runs is cheap and deterministic given the recorded
  results.** It is run again on every attempt and must compute the same values from the
  same traces; what it computes from a reply is recomputed, never stored.
- **Randomness is seeded from the episode's identity**: `vf.episode_random()` returns a
  `random.Random` seeded from the durable's scope (and any salt), so a rerun draws the
  same numbers. Draw from it in a fixed order.
- **External side effects are idempotent, or deferred to the caller's commit**: a write
  to a database or a request to another service may happen once per attempt, so it must
  be safe to repeat, or be returned in the outcome for the caller to apply once.

A stage that continues a seat's conversation from an earlier one (a reviewer across
rounds) passes the earlier trace: `agents.reviewer.interaction(resume=round_one.trace)`
is durable like any other call, and a rerun of the earlier stage (a new trace) reruns
it. Host code that reads the box between the turns of an exchange reads it the same on
a rerun: while the exchange replays, `exchange.runtime` restores the box from the
checkpoint of the turn it is at, on first use.

Code changed between attempts is safe: a call whose inputs changed (its fingerprint) runs
fresh, and so does every call whose position shifted onto another's trace. A call whose
order is not stable (a call made only on some paths) takes a `key=`.

```python
class ReviewEnv(vf.Env[ReviewConfig]):
    def trace_labels(self, task):
        return {"job": task.data.job}

    def durable(self, task):
        return vf.Durable(MyTraceStore())  # scoped by trace_labels

    async def run(self, task, agents):
        async with agents.coder.provision(task) as box:          # coder/0
            draft = await agents.coder.run(task, runtime=box)    # coder/1
            rng = vf.episode_random()
            focus = rng.choice(FOCUS_AREAS)                       # the same on a rerun
            review = await agents.reviewer.run(                   # reviewer/0
                review_task(draft.last_reply, focus), runtime=box
            )
        return ReviewOutcome(approved=review.last_reply.startswith("APPROVE"))
```
