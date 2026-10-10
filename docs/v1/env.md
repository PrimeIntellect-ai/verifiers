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

- The `AgenticJudgeEnv` defines the sequential interaction between a solver and judge agent. The judge can re-use the same runtime after the solver (`SharedAgenticJudgeEnv`) or use its own, new runtime `IsolatedAgenticJudgeEnv`.
- The `UserSimEnv` models users as agents, and the episode is a turn-by-turn conversation between the user and assistant agents.
- The `BestOfNEnv` runs n independent attempts at the same task, then marks which attempt achieved the highest reward (best) and whether any attempt crossed a success threshold (pass_at_n), which is useful for rejection sampling and pass@k evaluation.

## Verifier placement

Task scoring placement is independent of the Env's agent program. Set
`--env.agent.verifier.mode isolated` to restore declared artifacts into a fresh
runtime before running task metrics, rewards, and judge plugins. No extra agent
or harness is started. The default `task` follows `TaskData.verifier_mode`
(`shared` unless the task declares otherwise); `shared` explicitly scores in the
workspace. This also works for each attempt in `BestOfNEnv`.

```bash
uv run vf-eval my-task --env.agent.runtime.type docker --env.agent.verifier.mode isolated
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
        verifier_mode="isolated",
        artifacts=[vf.Artifact(source="src")],
    )
)
```

`agent.verifier.runtime` optionally chooses an independent provider, image,
resources, and network policy. Otherwise the agent's provider policy and the task's
scoring requirements resolve the fresh runtime. Solver checkpoints are not inherited.
`agent.verifier.env` overrides its process environment, and `agent.verifier.retries`
sets additional fresh attempts (default: 2). Runtime and environment overrides require
isolated placement. Relative artifacts require matching solver and verifier workdirs;
absolute paths support different workdirs. Fresh verification requires an isolated
filesystem and cannot use the host subprocess runtime.

A judge plugin is part of task scoring and receives the selected scoring runtime.
An agentic judge is a separate agent with its own trace, so its sequence and workspace
sharing remain the responsibility of an agentic-judge Env. The solver in that program
can independently use isolated task verification.
