---
name: create-environments
description: Build verifiers.v1 tasksets, environments, and harnesses. Use to add a benchmark, task tools, a simulated user, several cooperating agents, or a custom agent program.
---

# Create Environments

Build an installable package that runs with verifiers. Read the current code
before copying an external example: a `prime-envs` package may use a different
Verifiers version.

## Choose what to build

Use the user's requirements and the source benchmark to establish:

- Which dataset, split, prompts, and reference answers to use.
- How to score answers, including partial credit and failures.
- Which tools the task needs beyond those already in the harness.
- Whether the task needs a container, files, services, or network restrictions.
- Whether it needs several agents or follow-up user messages.

For a benchmark port, preserve the source prompts, task selection, allowed tools,
and scoring behavior. Ask about missing choices that would change the benchmark;
link the source that explains the choice.

Use built-in tasksets, judges, environments, and harnesses where possible.
`HarborTaskset` handles Harbor datasets with a small wrapper. It also needs
`HarborEnv` exported from the package for Compose and separate verification.
See [Harbor](../../docs/v1/harbor.md).

## Create the package

Always start with the CLI:

```bash
uv run vf-init my-task-v1
```

Add `-T` for custom MCP tools or `-H` for a custom harness. Prefer a built-in
harness unless the task needs a different agent program.

Export one `vf.Taskset` class through the package's `__all__`. Also export a
`vf.Env` or `vf.Harness` class if needed. The loader reads these classes and their
config types; do not add `load_environment()`, `load_taskset()`, or
`load_harness()` functions.

## Write the taskset

Use `import verifiers.v1 as vf` and its typed interfaces:

```python
import verifiers.v1 as vf


class AdditionData(vf.TaskData):
    answer: int


class AdditionTask(vf.Task[AdditionData]):
    @vf.reward
    async def exact_match(self, trace: vf.Trace) -> float:
        return float(trace.last_reply == str(self.data.answer))


class AdditionTaskset(vf.Taskset[AdditionTask, vf.TasksetConfig]):
    def load(self) -> list[AdditionTask]:
        return [
            AdditionTask(
                AdditionData(prompt=f"What is {i} + {i}?", answer=2 * i),
                self.config.task,
            )
            for i in range(100)
        ]


__all__ = ["AdditionTaskset"]
```

Keep each part small:

| Part | What belongs there |
| --- | --- |
| `TaskData` | One task's prompt, answers, image, workdir, resource requests, and other saved data. No live clients or runtime objects. |
| `Task` | Setup, finalization, validation, tools, stop conditions, and scoring. |
| `TaskConfig` | Settings used during a task, accessed through `self.config`. |
| `Taskset` | `load()` and tools shared by a worker. Implement `load()`, not `__init__`. |
| `TasksetConfig` | Dataset, split, seed, and selection settings. Store task settings under `task`. |
| `Harness` | Installing and running the agent program. Use `setup()`, not `__init__`. |
| `Env` | Which agents run, their interactions, and scores that compare several traces. |

See [tasksets](../../docs/v1/tasksets.md) for config, generators, image prompts,
scoring helpers, and hooks. Preserve source image order and use typed message
content; converting images to text loses them. Check the model and harness
support image input.

## Set up and score the task

The usual order is task `setup`, harness setup, agent execution, task `finalize`,
then scoring. Use `finalize` to save outputs before the runtime is removed.

- Prefer checks against the actual answer or output files.
- Reuse the built-in scoring helpers. For coding tasks, capture the patch before tests change the repository.
- Use a model judge only when code cannot decide correctness. Start with `ReferenceJudgeConfig` or `RubricJudgeConfig`; subclass `Judge` for a custom prompt or response format.
- Judge settings are separate from the evaluated model's settings. Pass `trace=trace` to custom judge calls to save their requests, verdicts, and usage.
- Rewards sum `score * weight`. Metrics record measurements without adding to reward. A scoring method can return a dictionary of named scores.
- Metrics run first, then rewards, then configured judges. Methods in each group run concurrently. Prepare shared inputs in `finalize`.
- Raise an exception when setup or scoring fails. Do not turn a failed check into a zero reward.

Store evidence needed after the run in `trace.info`, using JSON-compatible
values. Use a typed `vf.State` subclass for temporary state. It is not saved.
Implement `Task.validate(runtime)` when a known answer or solution can be checked
without a model.

`vf-validate` also checks the untouched task: setup, finalization, then scoring
without an agent. Make sure correct work passes and an untouched task does not.
This check can call configured judges.

## Use the runtime

Set `TaskData.image` to a pullable image with the task's dependencies. Ordinary
runtimes do not build task Dockerfiles; Prime prepares and caches pullable images
on first use. Set `Task.NEEDS_CONTAINER = True` for tasks that need isolated files
or run untrusted code.

Use `runtime.run`, `read`, and `write` to work inside the sandbox. Plain Python
file operations run on the evaluator. Check command exit codes.

Set `network_allow` and `network_block` when the task restricts network access.
Restrictions start after setup and apply inside the runtime. They do not cover
Python hooks on the evaluator or tools in separate runtimes.

For grading in a fresh container, use `isolated-verifier`, list outputs in
`TaskData.artifacts`, and prepare private tests in `stage_verifier`. Do not create
a second sandbox manager inside a reward. See the
[building guide](../../docs/v1/building-environments.md) for a complete example,
artifact transfer, independent grader settings, and validation limits. Check the
[runtime matrix](../../docs/v1/runtimes.md#capability-matrix) before choosing a backend.

## Add tools only when needed

Use `vf.Toolset` and `@vf.tool` for task-specific MCP tools. The harness must
support MCP. Start with `vf-init -T` and keep its module entry point:
`if __name__ == "__main__": YourToolset.run()`.

Choose where the tool runs and how long it lives:

- `Task.toolsets` with `ToolsetConfig`: one server per rollout. By default it runs in a host subprocess.
- `colocated = true`: run that task server inside the agent's runtime to share files and processes.
- `Taskset.toolsets` with `SharedToolsetConfig`: share a server across a worker's rollouts.
- `url`: connect to an existing streamable-HTTP MCP service.

Use `setup()` to prepare the server and `setup_task(task)` to receive task data
in a server created per task. See [tools](../../docs/v1/tasksets.md#adding-tools).

## Add agents or user messages

Check `best-of-n`, `agentic-judge`, and `user-sim` before writing a custom `Env`.
For a custom environment:

1. Add an `AgentConfig` field for each role on `EnvConfig`.
2. Implement `run(task, agents)`. Completed agent runs join the episode automatically; return nothing.
3. Use `setup(agents)` to set role properties, such as `agents.judge.trainable = False`.
4. Use `finalize(task, episode)` to compare traces and record scores. Agent runtimes are already gone, so use saved evidence and identify roles with `trace.agent.name`.

Role settings use `env.<role>.*`. Unset models and clients use the evaluation's
settings; unset harnesses use the taskset's default.

For scripted users or games, call `agents.solver.interaction(task)` inside
`run()` and send messages with `turn()`. A task with a prompt starts with bare
`turn()`; a task with `prompt=None` starts with `turn(message)`. One turn can
include several model and tool calls. To keep a scenario hidden from the
assistant, pass a task copy with `prompt=None` and keep scoring data in other
fields. For a model playing the user, prefer `user-sim`.

The assistant's harness must support resuming the conversation. `bash` and
`null` can restart from the transcript; programs with saved sessions can resume
those instead. See [Agent](../../docs/v1/agent.md) and [Env](../../docs/v1/env.md).

## Custom harnesses

Every model request must use the supplied `endpoint` and `secret`, so verifiers
can record it. Reuse `vf.ACPHarness` for programs that use ACP.

Set capability flags only for behavior the harness implements:
`SUPPORTS_MCP`, `SUPPORTS_RESUME`, `APPENDS_SYSTEM_PROMPT`, `SUPPORTS_SKILLS`, and
`SUPPORTS_TOOL_INTERCEPTION`. Flags alone do not implement these features.

Return the main program's `ProgramResult` from `runtime.run_program()` or
`runtime.run_uv_script()`. Do not create trace nodes manually. If the harness
creates files or sessions outside its temporary runtime, remove them in
`cleanup(trace, runtime)`. Cleanup runs after scoring and must be safe to repeat,
including when the environment owns the runtime. See [harnesses](../../docs/v1/harnesses.md).

## Dependencies and credentials

Add dependencies to the package's own `pyproject.toml` using `uv`; do not change
the repository's root dependencies or lockfile for a taskset. Check required
credentials where they are first needed and report missing values clearly.

Task data and config are saved. Use `Task.runtime_env()` or harness `forward_env`
for credentials needed in the runtime. Keep judge-only credentials on the
evaluator. Require a separately managed server only when the task explicitly
uses a remote URL.

## Check the result

1. Install with `uv pip install -e environments/<package_dir>` in the CLI's Python environment.
2. Run `uv run vf-eval <id> --dry-run` to check config and imports.
3. Run `vf-validate` on the intended runtime. `unchecked` means no gold check exists, not that it passed.
4. Run the agreed small evaluation. Inspect a success, wrong answer, and failure. For a benchmark port, compare prompts, tools, and scores with the source, including partial credit.

Keep dataset revision, split, filters, and image references explicit. Use
[debug-environments](../debug-environments/SKILL.md) for failures and
[evaluate-environments](../evaluate-environments/SKILL.md) for evaluation runs.

Publish only when requested, after the package works. Use the requested Hub
owner and visibility; ask only if a required choice is missing. For example:

```bash
prime env push my-task-v1 --visibility PRIVATE
```
