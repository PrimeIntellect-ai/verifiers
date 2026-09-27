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

## Task attempts

`agent.run(task)` opens a task attempt, runs the agent, grades the result, and
closes the attempt. To control the world lifetime yourself, pass an entered
attempt to `run()` or `interaction()`:

```python
async with task.open(placement=agents.solver.runtime_config) as attempt:
    solution = await agents.solver.run(attempt)
    await attempt.grade(solution)
```

An attempt owns its services (`attempt.runtime` is `attempt.services["main"]`).
Several agents may use the same attempt, each with an independent trace and
`trace.state`. `runtime=` can place an agent in another live runtime; that runtime
remains owned by its caller. Close every agent session before grading the selected
trace. Grading finalizes the world once and records task scores on that trace;
the environment decides how to assign credit to the other agents. Overlapping
agent sessions need distinct runtimes when networking is restricted: trusted setup
must not reopen egress underneath another running agent.

`Task.prepare(runtime)` prepares the world once on entry, before any agent trace
exists. `Task.setup(trace, runtime)` initializes each agent session. Harness metrics
and cleanup run when that agent closes; task `finalize` and scoring run when the
attempt is graded. Exiting an attempt always frees its owned services, including
after an exception or cancellation. An entered attempt is not retried by an agent:
its caller controls retries of the whole shared world.

## Isolated deterministic verification

Set `TaskConfig.verifier = vf.VerifierConfig()` to grade in a fresh runtime under
any environment strategy, including best-of-N. An explicit `task.open()` can also
take `verifier=vf.VerifierConfig(...)`. The `--env.id isolated-verifier` preset
configures this for a single solver. It is still a one-agent run: the environment records one solver
trace and starts no verifier agent, model, or harness.

```bash
uv run vf-eval my-task --env.id isolated-verifier --env.agent.runtime.type docker
```

Task authors use the existing task API. Declare every solver output the verifier
needs in `TaskData.artifacts`, then implement deterministic `@vf.reward` and
`@vf.metric` methods. A runtime parameter makes the fresh verifier box available:

```python
from pathlib import Path

import verifiers.v1 as vf


PRIVATE_TESTS = Path("tests/test_solution.sh").read_bytes()


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
        artifacts=[vf.Artifact(source="src")],
    )
)
```

The lifecycle is fixed:

1. The agent runs, records harness metrics, and closes its harness session.
2. The task attempt runs task `finalize`, collects the declared paths and
   `/logs/artifacts`, then releases its owned solver services.
3. It creates a fresh task controller and provisions either the same resolved
   container/runtime policy or the independently configured verifier runtime, runs
   task `prepare` and session `setup`, restores the artifacts at their original paths, runs task
   `stage_verifier`, reapplies the execution network policy, and runs task metrics
   and rewards onto the solver trace.

The verifier runtime must be Docker, Prime, or another container runtime; absolute
artifact restoration is intentionally refused on the host subprocess runtime.
By default the verifier uses the solver's resolved runtime policy. Set
`--env.verifier.runtime.*` to independently choose its runtime type, image, resources,
and network policy; `--env.verifier.env` can independently set its process environment.
Relative artifacts require matching solver and verifier workdirs because artifacts are
restored without path translation; absolute artifacts permit different workdirs.
Configured model-backed task judges are rejected: use deterministic metrics/rewards
here, or an agentic/judge environment when a model must judge the result.
`--env.verifier.retries` in the preset (or `TaskConfig.verifier.retries`) retries fresh verifier attempts after setup, restoration,
staging, or scoring failures (default: 2).
