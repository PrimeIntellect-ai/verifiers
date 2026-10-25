# Environments

An `Env` defines the control flow across one or more `Agents` for a single task. Each completed agent run produces a `Trace`, and the environment collects every trace from the run into an `Episode`.

If your taskset package does not export a custom `Env` and you don't specify `--env.id`, verifiers defaults to `SingleAgentEnv`, which runs a single agent on each task. You can also select one of the built-in environments with `--env.id`:

- `best-of-n` (`BestOfNEnv`): runs `n` independent attempts on the same task, marks which attempt achieved the highest reward (`best`), and records whether any attempt crossed a success threshold (`pass_at_n`), which is useful for rejection sampling and pass@k evaluation.
- `agentic-judge` (`AgenticJudgeEnv`): runs a solver agent first, then a judge agent to evaluate the result. The solver's `ScoringConfig` (`--env.solver.scoring.mode shared|isolated`) selects the workspace used for both task scoring and agentic judging.
- `best-of-n-judge` (`BestOfNJudgeEnv`): runs `n` attempts like `best-of-n`, then gives `--env.judges` independent judge agents each a fresh container holding every attempt's trace and collected artifacts under shuffled labels, and has them rank the attempts best to worst. Each attempt's `judge` reward is its rank score (the share of other attempts it beats, ties counting half) averaged across the judges.
- `user-sim` (`UserSimEnv`): models the user as a second agent and runs a turn-by-turn conversation between the user simulator and the assistant.

## Custom control flow

To write a custom environment, subclass `vf.Env` and declare an `AgentConfig` field on your `EnvConfig` for each agent role. The field name becomes the CLI and TOML config prefix, such as `--env.solver.model` and `--env.solver.harness.id`:

```python
import verifiers.v1 as vf


class AttemptsConfig(vf.EnvConfig):
    solver: vf.AgentConfig = vf.AgentConfig()
    attempts: int = 2


class AttemptsEnv(vf.Env[AttemptsConfig]):
    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        for _ in range(self.config.attempts):
            await agents.solver.run(task)

    async def finalize(self, task: vf.Task, episode: vf.Episode) -> None:
        best = max(trace.reward for trace in episode.traces)
        for trace in episode.traces:
            trace.record_metric("best", float(trace.reward == best))
```

`Env.run(task, agents)` receives the task and pre-initialized agents and orchestrates the episode; it returns `None` because every completed `agent.run(task)` call automatically appends its trace to the episode (with `trace.agent.name` set to the role name). Unless overridden on the role, each agent inherits the evaluation's model, client and sampling, and the taskset's default harness. An agent that sets its own `client` uses only its own `sampling`, because the evaluation's sampling can be specific to its endpoint. Export your `Env` subclass alongside the taskset in your package's `__all__` so verifiers uses it automatically.

When you need episode-level metrics that compare traces across runs, such as marking the best attempt above, implement `finalize(task, episode)`. Because the agents' runtimes have already been released by the time `finalize` runs, any evidence needed for cross-trace scoring should be recorded on the traces during `run`. If an environment includes helper roles that shouldn't be trained (such as a judge or user simulator), mark them in `setup(agents)` with `agents.judge.trainable = False`.

### Role defaults

You can customize a role's default settings directly on its `AgentConfig` field. For example, an answer-only environment that does not need shell tools can default its agent to the `null` harness:

```python
class AnswerConfig(vf.EnvConfig):
    agent: vf.AgentConfig = vf.AgentConfig(harness={"id": "null"})


class AnswerEnv(vf.Env[AnswerConfig]):
    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        await agents.agent.run(task)
```

CLI flags and TOML settings merge into those defaults, so a user can still switch harnesses at runtime with `--env.agent.harness.id bash`. If a role's default harness depends on a taskset setting, type the config's `taskset` field with that taskset's config class and choose a harness in a Pydantic `after` model validator only when `agent.harness` was left unset, keeping explicit user overrides intact.

### Outcomes, labels, and required agents

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

### Judging a solution directly

`judge_solution(judge, solution, runtime, config, isolated=...)`
(`verifiers.v1.envs.agentic_judge`) is that judge run on its own: any `on_score`
callback can call it with its scoring workspace and get each criterion's score
from the judge's verdict, or an error when the judge gives none. `JudgeTaskConfig`
takes its criteria from a `rubric` file or inline as `criteria`.

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
