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

To exclude a role from training, set it in `setup(agents)`, for example
`agents.judge.trainable = False`.

### Role defaults

Set a role's default harness on its `AgentConfig`. For example, an answer-only
environment can use the `null` harness by default:

```python
class AnswerConfig(vf.EnvConfig):
    agent: vf.AgentConfig = vf.AgentConfig(harness={"id": "null"})


class AnswerEnv(vf.Env[AnswerConfig]):
    async def run(self, task: vf.Task, agents: vf.Agents) -> None:
        await agents.agent.run(task)
```

Export `AnswerEnv` alongside the taskset. Declare roles with default instances,
as above. CLI and TOML overrides merge into those defaults; a user can select
another harness with `--env.agent.harness.id bash`. If the default depends on a
task setting, type the config's `taskset` field with that taskset's config class
and choose a harness in an `after` model validator only when `agent.harness`
is unset. An explicit selection should remain under the user's control.

### Scripted conversations

Keep all messages in one interaction when a task has a fixed sequence of user
turns. Store the first message in `prompt` and the rest in a task data field:

```python
class ConversationData(vf.TaskData):
    followups: list[str] = []


class ConversationEnv(vf.SingleAgentEnv):
    async def run(self, task: vf.Task[ConversationData], agents: vf.Agents) -> None:
        async with agents.agent.interaction(task) as interaction:
            segment = await interaction.turn()
            for message in task.data.followups:
                if segment.terminated:
                    break
                segment = await interaction.turn(message)
```

Use a harness that supports conversation resume. The initial bare `turn()` uses
the task prompt; for `prompt=None`, supply the first message yourself. Each call
runs one harness segment, which can include several model and tool calls. A
later call can report termination without consuming its message. Leaving the
context ends this rollout and runs finalization and scoring, even when the
script ends before the agent's limits. All segments share one trace and the
same token and turn budgets.

Record any results needed by task hooks in `interaction.trace.state` or
`interaction.trace.info` before leaving the context. Use `user-sim` when a model
should generate the user messages. See [Agent](agent.md) for interaction details.

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
