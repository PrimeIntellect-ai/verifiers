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

- `IsolatedVerifierEnv` runs one solver, transfers only declared artifacts into a
  fresh configured runtime, and runs deterministic task scoring there.
- The `AgenticJudgeEnv` defines the sequential interaction between a solver and judge agent. The judge can re-use the same runtime after the solver (`SharedAgenticJudgeEnv`) or use its own, new runtime `IsolatedAgenticJudgeEnv`.
- The `UserSimEnv` models users as agents, and the episode is a turn-by-turn conversation between the user and assistant agents.
- The `BestOfNEnv` runs n independent attempts at the same task, then marks which attempt achieved the highest reward (best) and whether any attempt crossed a success threshold (pass_at_n), which is useful for rejection sampling and pass@k evaluation.

## Isolated deterministic verification

Select `--env.id isolated-verifier` when the task's score must not run in the
solver's sandbox. It is still a one-agent run: the environment records one solver
trace and starts no verifier agent, model, or harness.

```bash
uv run eval my-task --env.id isolated-verifier --env.agent.runtime.type docker
```

Task authors use the existing task API. Declare every solver output the verifier
needs in `TaskData.artifacts`, then implement deterministic `@vf.reward` and
`@vf.metric` methods. A runtime parameter makes the fresh verifier box available:

```python
from pathlib import Path

import verifiers.v1 as vf


PRIVATE_TESTS = Path("tests/test_solution.sh").read_bytes()


class CodeTask(vf.Task[CodeData]):
    @vf.reward
    async def tests(self, runtime: vf.Runtime) -> float:
        # Stage private verifier inputs here, then execute and parse them.
        await runtime.write("/tmp/test_solution.sh", PRIVATE_TESTS)
        result = await runtime.run(["bash", "/tmp/test_solution.sh"], {})
        return float(result.exit_code == 0)


task = CodeTask(
    CodeData(
        prompt="Fix the implementation.",
        artifacts=[vf.Artifact(source="src")],
    )
)
```

The lifecycle is fixed:

1. The task and harness run normally, including task `finalize` and harness metrics,
   but task metrics and rewards are deferred.
2. The environment collects the declared paths and `/logs/artifacts`, then destroys
   the solver runtime.
3. It creates a fresh task controller and provisions either the same resolved
   container/runtime policy or the independently configured verifier runtime, runs
   task `setup`, restores the artifacts at their original paths, reapplies the
   execution network policy, and runs task metrics and rewards onto the solver trace.

The verifier runtime must be Docker, Prime, or another container runtime; absolute
artifact restoration is intentionally refused on the host subprocess runtime.
By default the verifier uses the solver's resolved runtime policy. Set
`--env.verifier-runtime.*` to independently choose its runtime type, image, workdir,
resources, and network policy; `--env.verifier-env` can independently set its process
environment. Configured model-backed task judges are rejected: use deterministic
metrics/rewards here, or an agentic/judge environment when a model must judge the
result. `--env.verifier-retries` controls fresh-box infrastructure retries (default:
2).
