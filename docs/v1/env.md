# Environments

An `Env` decides which agents run and in what order. Each completed agent run
produces a trace. The environment collects those traces in an `Episode`.

Choose a built-in environment with `--env.id`:

| ID | What it does |
| --- | --- |
| `best-of-n` | Runs several independent attempts, marks the best reward, and records whether any attempt passed. |
| `agentic-judge` | Runs a solver, then a judge agent in a fresh runtime. |
| `shared-agentic-judge` | Runs a solver, then a judge agent in the same runtime. |
| `user-sim` | Runs a conversation between an assistant and a model playing the user. |
| `isolated-verifier` | Runs one solver, then scores its saved files in a fresh runtime without a judge model. |

Without an explicit ID, verifiers uses the environment exported by the taskset
package. If none is exported, `SingleAgentEnv` runs one agent on each task.

## Custom control flow

Add an `AgentConfig` field for each agent role. Its name becomes the config path,
such as `--env.solver.model` and `--env.solver.harness.id`:

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

Export the environment alongside the taskset in the package's `__all__`.
`run(task, agents)` returns nothing: completed agent runs are added to the
episode automatically. Unless configured, roles use the evaluation's model and
client and the taskset's default harness. Read the role from `trace.agent.name`.

Use `finalize(task, episode)` for scores that compare several traces. The agents'
runtimes have already been released, so save any needed evidence in the traces.

For scripted user messages, call `turn()` on `agents.solver.interaction(task)`
inside `run`; see [Agent](agent.md). For a model playing the user, use `user-sim`.
To exclude a role from training, set it in `setup(agents)`, for example
`agents.judge.trainable = False`.

## Isolated deterministic verification

Use `--env.id isolated-verifier` to grade in a container the solver has never
touched. It runs one agent and records one trace; grading uses code, not a model.

```bash
uv run vf-eval my-task --env.id isolated-verifier --env.agent.runtime.type docker
```

List outputs in `TaskData.artifacts` and prepare private tests in
`Task.stage_verifier`. The solver runtime is removed before task metrics and
rewards run against the verifier runtime. By default, the verifier uses the
solver's resolved runtime settings; `env.verifier.runtime` selects independent
settings. Model-backed task judges are not supported.

See [building environments](building-environments.md#separate-agent-and-grading-sandboxes)
for a complete package, lifecycle diagram, separate images and resources,
artifact requirements, and validation commands.
