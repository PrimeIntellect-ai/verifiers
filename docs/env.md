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
