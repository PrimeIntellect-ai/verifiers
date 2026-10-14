# The Env

An `Env` defines the control flow between `Agents`. In the simplest case, it is just a `SingleAgentEnv` where a single agent solves a task from a taskset.

Its core signature is `Env.run(task: Task, agents: Agents)` -> None — it is passed an initial task and pre-initialized agents and then programs the full multi-agent control flow; every finished agent run automatically joins the resulting `Episode`, which holds all the traces of all the agents.

```python
class Env(ABC):
    @abstractmethod
    async def run(self, task: Task, agents: Agents) -> None:
        """Run a single multi-agent episode."""
        ...
```

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
The judge borrows that workspace with colocated deployment, including when the
solver uses split deployment. Judge model and harness settings live under
`env.judge`; workspace placement belongs to `env.solver.scoring`.
