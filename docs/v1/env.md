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

1. The task and harness run normally, including task `finalize` and harness metrics,
   but task metrics and rewards are deferred.
2. The environment collects the declared paths and `/logs/artifacts`, then destroys
   the solver runtime.
3. It creates a fresh task controller and provisions either the same resolved
   container/runtime policy or the independently configured verifier runtime, runs
   task `setup`, restores the artifacts at their original paths, runs task
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
`--env.verifier.retries` retries fresh verifier attempts after setup, restoration,
staging, or scoring failures (default: 2).

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
