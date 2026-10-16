# Environments

An `Env` defines the control flow across one or more `Agents` for a single task. Each completed agent run produces a `Trace`, and the environment collects every trace from the run into an `Episode`.

If your taskset package does not export a custom `Env` and you don't specify `--env.id`, verifiers defaults to `SingleAgentEnv`, which runs a single agent on each task. You can also select one of the built-in environments with `--env.id`:

- `best-of-n` (`BestOfNEnv`): runs `n` independent attempts on the same task, marks which attempt achieved the highest reward (`best`), and records whether any attempt crossed a success threshold (`pass_at_n`), which is useful for rejection sampling and pass@k evaluation.
- `agentic-judge` (`AgenticJudgeEnv`): runs a solver agent first, then a judge agent to evaluate the result. The solver's `ScoringConfig` (`--env.solver.scoring.mode shared|isolated`) selects the workspace used for both task scoring and agentic judging.
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

`Env.run(task, agents)` receives the task and pre-initialized agents and orchestrates the episode; it returns `None` because every completed `agent.run(task)` call automatically appends its trace to the episode (with `trace.agent.name` set to the role name). Unless overridden on the role, each agent inherits the evaluation's model and client and the taskset's default harness. Export your `Env` subclass alongside the taskset in your package's `__all__` so verifiers uses it automatically.

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

### Scripted conversations

When a task has a fixed sequence of follow-up user messages (as opposed to an LLM-driven user simulator, where `user-sim` fits better), you can keep the entire exchange inside a single rollout with `agent.interaction(task)`. Store the opening message in `TaskData.prompt` and the follow-up turns in a field on your `TaskData` subclass:

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

This pattern requires a harness that supports conversation resume. Calling `interaction.turn()` with no arguments sends the task's initial `prompt` (or pass a string if `prompt=None`), and each subsequent `turn(message)` runs the next harness segment within the same trace, token budget, and turn budget. Exiting the `async with` block finalizes and scores the rollout even if the script finishes before hitting the agent's turn or token limits. If your task hooks need extra state from the conversation, store it in `interaction.trace.state` or `interaction.trace.info` before leaving the block. See [Agent](agent.md) for more details on driving interactions.

## Verifier placement

Task scoring placement is configured per agent rather than by switching `Env` classes, so it works under `SingleAgentEnv`, `BestOfNEnv`, and `AgenticJudgeEnv` alike. Set `--env.agent.scoring.mode isolated` (or set `scoring=vf.ScoringConfig(mode="isolated")` on `TaskData` as the task default) to restore declared artifacts into a fresh container runtime before running `@vf.metric`, `@vf.reward`, and configured judge plugins:

```bash
uv run vf-eval my-task --env.agent.runtime.type docker --env.agent.scoring.mode isolated
```

Explicit `agent.scoring` fields override `TaskData.scoring`, and unset fields inherit from the task (defaulting to `shared` scoring in the agent's workspace when neither sets a mode).

For isolated grading, declare every solver output the grader needs in `TaskData.artifacts` and write any private test files in `Task.stage_verifier`:

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

When isolated scoring runs, task `setup` initializes the fresh container, declared artifacts are restored at their original paths, and `stage_verifier` installs private grading inputs before metrics, rewards, and judge plugins execute. You can customize the verifier environment and retry behavior under `agent.scoring`:

- `agent.scoring.runtime` optionally selects a separate runtime provider, image, resource allocation, and network policy for the fresh verifier (when omitted, the agent's runtime policy and the task's scoring requirements are used; solver checkpoints are not inherited). Relative artifact paths require matching solver and verifier working directories, while absolute paths work across different `workdir` settings. Isolated verification requires a container runtime and cannot use `subprocess`.
- `agent.scoring.env` merges over `TaskData.scoring.env` by key for both `shared` and `isolated` scoring, applying to scoring setup, staging, rewards, and judging without altering the solver's process environment (`${VAR}` and `${VAR:-default}` templates resolve from the host when scoring starts).
- `agent.scoring.retries` uses `RetryConfig` (defaulting to `max_retries = 2`) to retry failed isolated scoring attempts in a fresh verifier container while reusing the completed solver submission. A valid `0.0` reward is treated as a completed grade and is not retried, and shared scoring is not retried.

In `AgenticJudgeEnv`, the separate judge agent runs through the solver's `on_score` callback inside the selected verifier workspace before that workspace is torn down. Use `--env.solver.scoring.mode shared|isolated` (or leave it unset to follow the task's default) to control where the judge inspects the submission, and configure the judge's model and harness under `env.judge`. See [building environments](building-environments.md#separate-agent-and-grading-sandboxes) for a complete package walkthrough and validation commands.
