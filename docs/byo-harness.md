# v1 Taskset/Harness Environments

BYO Harness is the v1 authoring path for reusable Verifiers environments. Use it
when the environment has a taskset, a custom harness, tools, users, sandboxes,
programs, packaged benchmark adapters, or config that should work the same way
from Python, TOML, eval, GEPA, RL, and Hosted Training.

For short legacy environments built directly from `SingleTurnEnv`, `ToolEnv`, or
`MultiTurnEnv`, see [Environments](environments.md). For new reusable
environments, the v1 Taskset/Harness shape is the golden path.

## Golden Shape

Every v1 environment has the same outer shape:

```python
import verifiers as vf


class MyTasksetConfig(vf.TasksetConfig):
    system_prompt: vf.SystemPrompt = "Answer exactly."
    split: str = "train"


class MyTaskset(vf.Taskset[MyTasksetConfig]):
    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        return [
            {
                "prompt": [{"role": "user", "content": "Reverse abc."}],
                "answer": "cba",
                "split": split,
                "max_turns": 1,
            }
        ]

    @vf.reward(weight=1.0)
    async def exact(self, task: vf.Task, state: vf.State) -> float:
        messages = vf.get_messages(state.get("completion") or [], role="assistant")
        response = str(messages[-1].content or "") if messages else ""
        return float(response.strip() == task["answer"])


def load_taskset(config: MyTasksetConfig) -> MyTaskset:
    return MyTaskset(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.Harness(config=config.harness),
    )
```

Add a custom harness only when the environment owns reusable execution behavior:

```python
async def run_agent(task: vf.Task, state: vf.State) -> vf.State:
    client = state.get_client(api="chat")
    response = await client.chat.completions.create(
        model=state.get_model(),
        messages=[*state.get("system_prompt", []), *task["prompt"]],
    )
    message = response.choices[0].message
    state["completion"] = [
        {"role": "assistant", "content": message.content or ""}
    ]
    return state


class MyHarnessConfig(vf.HarnessConfig):
    program: vf.ProgramConfig = vf.ProgramConfig(fn="my_env:run_agent")


class MyHarness(vf.Harness[MyHarnessConfig]):
    pass


def load_harness(config: MyHarnessConfig) -> MyHarness:
    return MyHarness(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
    )
```

The loader annotations are load-bearing. `load_taskset(config:
MyTasksetConfig)` tells Verifiers how to validate `[env.taskset]`.
`load_harness(config: MyHarnessConfig)` does the same for `[env.harness]`.
`load_environment` stays typed as `vf.EnvConfig` and should not accept
environment-specific keyword arguments.

## Non-Negotiable Rules

These rules are intentionally strict so environments look and behave the same
across the ecosystem:

- Import the public API as `import verifiers as vf`.
- Define `XXXConfig` Pydantic config classes for structured settings.
- Put environment-specific fields on the owner config:
  `TasksetConfig` for task behavior, `HarnessConfig` for execution behavior.
- Expose `load_taskset(config: MyTasksetConfig)` for custom taskset config.
- Expose `load_harness(config: MyHarnessConfig)` only for custom harness config.
- Keep `load_environment(config: vf.EnvConfig)` tiny and explicit.
- Do not subclass `vf.Env` for normal environment packages.
- Do not subclass `vf.EnvConfig` just to narrow child config types.
- Do not override `Taskset.__init__`, `Harness.__init__`, or `User.__init__`;
  those constructors are final framework setup.
- Do not pass `None` to loaders or synthesize fallback config objects.
- Do not mirror taskset/harness fields as root loader kwargs.
- Do not put system messages inside `task["prompt"]`; use `system_prompt`.
- Do not use detached helper functions for one-line or single-use class logic.
- Do not hide core taskset/harness behavior at the bottom of a file. Put public
  lifecycle behavior on the class with standard names or decorators.
- Use `@vf.reward`, `@vf.metric`, `@vf.setup`, `@vf.update`, `@vf.cleanup`,
  `@vf.teardown`, and `@vf.stop`.

Utility modules are appropriate only when logic is reused in multiple places,
is several lines of genuine internal plumbing, or adapts a messy upstream
library that users should not have to think about.

## Ownership

![Task to Harness to State](assets/v1-task-harness-state.svg)

v1 has three composition objects:

| Object | Owns |
| --- | --- |
| `Taskset` | Task data, task prompts, task controls, task-owned tools, user behavior, task-specific setup/update/cleanup, metrics, rewards, advantages, and task-owned program/sandbox inputs. |
| `Harness` | Rollout execution, execution-level system prompts, model/client defaults, programs, command agents, framework adapters, endpoint interception, primary sandbox placement, harness-owned tools, and execution artifacts. |
| `Env` | The adapter from one taskset/harness pair to eval and training workers. |

If a tool or state transition defines the task's action space, observations, or
success condition, it belongs to the taskset. If a class only knows how a model
or external agent attempts arbitrary tasks, it belongs to the harness.

Examples:

- Wikispeedia link tools belong to the Wikispeedia taskset.
- TextArena game state and user responses belong to the TextArena taskset.
- Harbor task directories, task uploads, and tests belong to `HarborTaskset`.
- OpenCode, Pi, Mini SWE Agent, Terminus, and RLM command execution belong to
  harness classes.
- Model endpoint routing and interception belong to the harness/runtime, not to
  task rows.

## Config

Config objects must be serializable. Use import-ref strings such as
`"my_env.module:factory"` when a config has to name a callable across TOML, CLI,
or package boundaries. Python constructors may pass concrete objects only where
the constructor explicitly accepts runtime objects, such as `vf.Toolset(tools=[...])`
or standalone `vf.Harness(model=..., client=...)`.

Common config fields inherited by tasksets and harnesses:

| Field | Meaning |
| --- | --- |
| `system_prompt` | A string, message list, or `vf.SystemPromptConfig`. |
| `user` | A `UserConfig` subclass that materializes a registered `User`. |
| `toolsets` | Configured toolset collection. |
| `objects` | Private dependency factories owned by this object. |
| `bindings` | Hidden argument bindings for lifecycle handlers and tools. |
| `artifacts` | Text/JSON artifacts owned by this object. |
| `stops`, `setups`, `updates`, `metrics`, `rewards`, `advantages`, `cleanups`, `teardowns` | Import-ref lifecycle handlers. |
| `scoring` | Per-handler tuning or skipping by handler name. |

Taskset-specific config should describe data and task behavior. Harness-specific
config should describe execution. Avoid broad unions and untyped mappings unless
arbitrary JSON is the actual task payload.

## Tasks And Datasets

Tasksets load train and eval tasks through one method:

```python
class GSM8KTasksetConfig(vf.TasksetConfig):
    dataset_name: str = "gsm8k"
    train_split: str = "train"
    eval_split: str = "test"
    num_examples: int | None = None


class GSM8KTaskset(vf.Taskset[GSM8KTasksetConfig]):
    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        dataset_split = (
            self.config.train_split if split == "train" else self.config.eval_split
        )
        dataset = load_dataset(self.config.dataset_name, "main", split=dataset_split)
        if self.config.num_examples is not None:
            dataset = dataset.select(range(self.config.num_examples))
        return (
            {
                "prompt": [{"role": "user", "content": row["question"]}],
                "answer": row["answer"],
            }
            for row in dataset
        )
```

`vf.Tasks` may be a `datasets.Dataset`, an iterable of serializable task
records, or an iterable of `vf.Task` objects. During rollout, records are always
materialized as immutable `vf.Task`.

Task records are JSON-serializable. Use top-level fields for framework controls:

| Field | Meaning |
| --- | --- |
| `prompt` | User/developer/tool messages. No system messages. |
| `system_prompt` | Per-task system instructions. |
| `answer` | Reference answer or target data. |
| `info` | Extra serializable metadata. |
| `max_turns` | Per-task default-loop turn limit. |
| `toolsets` | Toolset visibility: `{"show": [...]}` or `{"hide": [...]}`. |
| `tools` | Per-toolset tool visibility: `{"search": {"show": ["lookup"]}}`. |
| `sandbox` | Per-task sandbox overrides. |
| `program` | Task-owned program files, dirs, setup, env, artifacts, bindings, and args. |
| `artifacts` | Task-owned artifacts collected after program execution. |

Users should not have to think about task/example IDs. Include upstream IDs only
when they are meaningful task metadata.

Priority for runtime controls is:

```text
explicit state.runtime > task top-level controls > harness defaults
```

`task.runtime` is not public schema. Runtime handles live on `state` while a
rollout is active and are stripped before serialization.

## System Prompts And GEPA

The normal pattern is a config field:

```python
class WordleTasksetConfig(vf.TasksetConfig):
    system_prompt: vf.SystemPrompt = (
        "Play Wordle. Submit guesses inside <guess>...</guess> tags."
    )
```

For GEPA or other file-backed prompt optimization, make the config file-backed:

```python
class WordleTasksetConfig(vf.TasksetConfig):
    system_prompt: vf.SystemPromptConfig = vf.SystemPromptConfig(
        path="system_prompt.txt"
    )
```

Override `load_system_prompt(config)` only when prompt loading is genuinely
computed from other config fields or package resources. Static prompt text and
file-backed prompt paths belong in config.

System prompt resolution happens per task during rollout setup. There are two
sides:

- The taskset side is `task["system_prompt"]` when the task provides one,
  otherwise `TasksetConfig.system_prompt`.
- The harness side is `HarnessConfig.system_prompt`.

The default strategy is `HT`, which preserves harness policy first and
then the resolved taskset side. Set `HarnessConfig.system_prompt_strategy` only
when the harness needs a different strategy:

| Strategy | Meaning |
| --- | --- |
| `HT` | Harness side followed by resolved taskset side. |
| `TH` | Resolved taskset side followed by harness side. |
| `H_OR_T` | Harness side when present, otherwise resolved taskset side. |
| `T_OR_H` | Resolved taskset side when present, otherwise harness side. |
| `H` | Harness side only. |
| `T` | Resolved taskset side only. |
| `REJECT` | Error if both sides are present. |

## Toolsets

Toolsets package model-visible tool schemas, hidden bindings, private objects,
artifacts, lifecycle hooks, and optional runtime scope.

```python
class SearchTasksetConfig(vf.TasksetConfig):
    objects: vf.ObjectsConfig = vf.ObjectsConfig.model_validate(
        {"index": "my_env.search:load_index"}
    )
    bindings: vf.BindingsConfig = vf.BindingsConfig.model_validate(
        {"search.query.index": "objects.index"}
    )


async def query(index, q: str) -> str:
    return index.search(q)


class SearchTaskset(vf.Taskset[SearchTasksetConfig]):
    def load_toolsets(self, config: SearchTasksetConfig) -> vf.Toolsets:
        return {"search": vf.Toolset(tools=[query])}
```

Bindings inject hidden arguments that the model does not see. Common binding
sources include `task.*`, `state.*`, `objects.*`, and framework-owned runtime
paths. Literal string values should come from callables or task/config fields so
misspelled binding paths fail loudly.

Tasks show all toolsets and tools by default. A task can hide or show toolsets
and individual tools without changing the global taskset:

```python
yield {
    "prompt": [{"role": "user", "content": "Use the calculator only."}],
    "toolsets": {"show": ["math"]},
    "tools": {"math": {"show": ["calculate"]}},
}
```

Use rollout-scoped toolsets for runtime resources that exist only during a
rollout, such as OpenReward sessions or sandbox-backed tool servers. Global
toolsets should discover stable tool names and schemas up front. If a tool's
backend state is rollout-local, keep the backend handle on `state` and expose
the tool through a normal `vf.Toolset`.

MCP servers are tool entries:

```python
class FetchTasksetConfig(vf.TasksetConfig):
    toolsets: dict[str, vf.ToolsetConfig] = {
        "fetch": vf.ToolsetConfig(
            tools=[
                vf.MCPToolConfig(command="uvx", args=["mcp-server-fetch"]),
            ],
            scope="rollout",
        )
    }
```

Custom harness programs should consume resolved tools from state:

```python
async def run_agent(task: vf.Task, state: vf.State) -> vf.State:
    tools = state.get_tools()
    result = await framework_agent(task["prompt"], tools=list(tools.values()))
    state["completion"] = [{"role": "assistant", "content": result}]
    return state
```

Wrap tools only at the boundary where a third-party framework requires its own
tool object type.

## Users

A `User` simulates environment/user responses between model turns. It is not a
callable; subclass `vf.User` and implement `get_response`.

```python
class GameUserConfig(vf.UserConfig):
    pass


class GameUser(vf.User[GameUserConfig]):
    async def get_response(
        self,
        task: vf.Task,
        state: vf.State,
        messages: list[vf.Message],
    ) -> list[vf.UserMessage]:
        observation = state["game"].observe(messages)
        return [{"role": "user", "content": observation}]


class GameTasksetConfig(vf.TasksetConfig):
    user: GameUserConfig = GameUserConfig()
```

Use a user when the environment naturally replies to the model after the model
speaks and before scoring. Use tools when the model chooses an explicit action
from a schema. Use setup/update handlers when state should change without adding
messages to the conversation.

## Programs, Harnesses, And Sandboxes

`HarnessConfig.program` is a `vf.ProgramConfig`. It defines the program the
harness runs after setup:

| Form | Meaning |
| --- | --- |
| `vf.ProgramConfig()` | Base endpoint-backed tool loop. |
| `vf.ProgramConfig(base=True)` | Explicit base loop, often with sandbox options. |
| `vf.ProgramConfig(fn="my_env:run")` | Importable Python program. |
| `vf.ProgramConfig(command=["agent", "run"])` | Local or sandboxed command. |

The preferred Python program signature is:

```python
async def program(task: vf.Task, state: vf.State) -> vf.State:
    state["answer"] = task["answer"]
    return state
```

Programs may call models, call tools, run deterministic solvers, replay cached
solutions, or adapt third-party agent frameworks. They should read immutable
task data, mutate serializable state, and let the standard lifecycle handle
artifacts, scoring, and cleanup.

Program config supports:

| Field | Meaning |
| --- | --- |
| `sandbox` | `False`, `True`, or `SandboxConfig`; controls program placement. |
| `files` | Remote file path to literal/task/state/callable value. |
| `dirs` | Remote directory path to local path or package resource. |
| `setup` | Setup commands or callable values. |
| `env` | Program environment variables. |
| `bindings` | Hidden args for callable program values. |
| `artifacts` | Files collected after program execution. |
| `channels` | Program-facing tool channels, currently `callable` and `mcp`. |
| `args` | Extra command args appended to command harness defaults. |

Tasksets can contribute task-local program data through `task["program"]`.
Harnesses still own the command/function/base kind, channel wiring, and primary
sandbox placement. Duplicate files, env vars, artifacts, or bindings fail fast.

Sandbox config belongs on the harness when it is part of the execution
mechanism:

```python
class PythonHarnessConfig(vf.HarnessConfig):
    sandbox: vf.SandboxConfig = vf.SandboxConfig(
        image="python:3.11-slim",
        scope="rollout",
    )
    program: vf.ProgramConfig = vf.ProgramConfig(
        fn="my_env.solver:solve",
        sandbox=True,
    )
```

Put sandbox overrides on tasks only when the taskset owns per-task images,
files, resource sizing, or setup.

## Lifecycle And Scoring

![v1 composition lifecycle](assets/v1-composition-lifecycle.svg)

Lifecycle decorators attach behavior to the owning class:

```python
class QAATaskset(vf.Taskset[QAATasksetConfig]):
    @vf.setup
    async def setup_question(self, task: vf.Task, state: vf.State) -> None:
        state["question_seen"] = True

    @vf.update
    async def extract_answer(self, task: vf.Task, state: vf.State) -> None:
        messages = vf.get_messages(state.get("completion") or [], role="assistant")
        state["answer"] = str(messages[-1].content or "") if messages else ""

    @vf.metric
    async def response_length(self, task: vf.Task, state: vf.State) -> float:
        return float(len(str(state.get("answer") or "")))

    @vf.reward(weight=1.0)
    async def exact(self, task: vf.Task, state: vf.State) -> float:
        return float(state.get("answer") == task["answer"])
```

Rollout lifecycle handlers can request `task`, `state`, `completion`, `prompt`,
and hidden bound args. Group handlers use `tasks` and `states` and must return
one value per state when scoring.

Use class methods with standard public names for lifecycle behavior. Avoid
private helper methods and detached helper functions unless the logic is
reused, nontrivial, and belongs in a named utility module.

## Objects, Bindings, And Artifacts

`objects` are private dependency factories owned by a taskset, harness, toolset,
or user. `bindings` connect those objects, task fields, state fields, or other
runtime values to hidden handler/tool/program arguments.

```python
class ExtractTasksetConfig(vf.TasksetConfig):
    objects: vf.ObjectsConfig = vf.ObjectsConfig.model_validate(
        {"extractor": "my_env.extractors:load_answer_extractor"}
    )
    bindings: vf.BindingsConfig = vf.BindingsConfig.model_validate(
        {"exact.extractor": "objects.extractor"}
    )


class ExtractTaskset(vf.Taskset[ExtractTasksetConfig]):
    @vf.reward(weight=1.0)
    async def exact(self, task: vf.Task, state: vf.State, extractor) -> float:
        return float(extractor(state.get("completion") or []) == task["answer"])
```

Use artifacts when the owner produces files that should be copied into the
serialized state:

```python
class AgentHarnessConfig(vf.HarnessConfig):
    artifacts: vf.ArtifactsConfig = vf.ArtifactsConfig.model_validate(
        {"agent_log": {"path": "/app/agent.log", "format": "text", "optional": True}}
    )
```

Artifacts can live on tasksets, harnesses, users, toolsets, programs, or tasks.
The owner determines which sandbox/filesystem is searched first.

## Nested Harnesses And Borrowed Runtime State

Nested harnesses are normal harness runs. Create a child task, create a child
state, and run the child harness.

```python
async def ask_child(name: str, state: vf.State) -> str:
    harness = vf.Harness(
        config=vf.HarnessConfig(
            program=vf.ProgramConfig(fn="my_env.children:greet")
        )
    )
    task = vf.Task(
        {
            "prompt": [{"role": "user", "content": f"Say hello to {name}."}],
            "name": name,
        }
    ).freeze()
    child_state = await harness.run(task, state.for_task(task))
    messages = vf.get_messages(child_state.get("completion") or [], role="assistant")
    return str(messages[-1].content or "") if messages else ""
```

Borrow runtime handles only when the child run intentionally reuses live parent
resources:

```python
child_state = state.for_task(child_task, borrow="model", tools=["search"])
```

Borrowed resources stay owned by the source runtime. They are process-local and
stripped before state is serialized.

## Packaged Tasksets And Harnesses

Reusable implementations live in standalone packages under `packages/`:

```bash
uv add "verifiers[packages]"
```

Install narrower extras when you only need one side:

```bash
uv add "verifiers[tasksets]"
uv add "verifiers[harnesses]"
uv add "verifiers[openenv]"
uv add "verifiers[openreward]"
uv add "verifiers[ta]"
uv add "verifiers[nemogym]"
```

Tasksets:

| Class | Use |
| --- | --- |
| `HarborTaskset` | Harbor task directories or Harbor Hub datasets. |
| `OpenEnvTaskset` | Upstream OpenEnv projects with no task rewrites. |
| `OpenRewardTaskset` | Upstream OpenReward environments and session tools. |
| `TextArenaTaskset` | TextArena single-player games with a taskset-owned `User`. |
| `NeMoGymTaskset` | NeMo Gym JSONL rows. |

Harnesses:

| Class | Use |
| --- | --- |
| `OpenCode` | OpenCode CLI agent. |
| `Pi` | Pi Coding Agent. |
| `MiniSWEAgent` | mini-swe-agent. |
| `Terminus2` | Harbor Terminus agent. |
| `RLM` | Recursive language model command harness. |
| `NeMoGymHarness` | NeMo Gym rollout collection. |

Package-backed environments still use the same loader shape:

```python
import verifiers as vf
from harnesses import OpenCode, OpenCodeConfig
from tasksets import HarborTaskset, HarborTasksetConfig


def load_taskset(config: HarborTasksetConfig) -> HarborTaskset:
    return HarborTaskset(config=config)


def load_harness(config: OpenCodeConfig) -> OpenCode:
    return OpenCode(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
    )
```

If a packaged taskset should load package-local assets, make that a typed config
field such as `bundle_package=__name__`. Do not patch config objects inside
`load_taskset`; pass the intended config from Python or TOML.

## TOML, CLI, Eval, And Training

Eval and training config owns the run: model, endpoint, sampling, examples, and
rollout count. v1 child config owns the environment behavior:

```toml
model = "openai/gpt-5.4-mini"
num_examples = 5
rollouts_per_example = 3

[[eval]]
env_id = "my-v1-env"

[eval.sampling]
max_tokens = 4096

[eval.taskset]
split = "eval"
system_prompt = "Answer exactly."

[eval.harness]
max_turns = 4
```

Hosted Training and RL use the same child section shape:

```toml
[[env]]
id = "primeintellect/my-v1-env"

[env.taskset]
split = "train"

[env.harness]
max_turns = 8
```

CLI overrides should target typed child config fields:

```bash
prime eval run my-v1-env --taskset.split eval --harness.max-turns 4
```

Callable config uses import refs:

```toml
[[env.taskset.rewards]]
fn = "my_env.rewards:exact"
weight = 1.0
priority = 0
```

Use `[...scoring.function_name]` to tune or skip an existing class-defined
metric/reward without creating a new signal:

```toml
[env.taskset.scoring.exact]
weight = 0.5
```

## Validation Checklist

Before publishing or asking for review:

1. `load_environment(config: vf.EnvConfig)` is the only root loader shape.
2. `load_taskset(config: MyTasksetConfig)` is present for custom tasksets.
3. `load_harness(config: MyHarnessConfig)` is present only for custom harnesses.
4. No `Taskset`, `Harness`, or `User` subclass overrides `__init__`.
5. No normal environment subclass of `vf.Env` or `vf.EnvConfig` exists.
6. Config fields are serializable and named `XXXConfig`.
7. Taskset-owned behavior is not hidden in the harness.
8. Harness-owned execution is not hidden in task rows.
9. Static prompts live in config; computed prompts use `load_system_prompt`.
10. Tools are exposed through `vf.Toolset`; task rows only show/hide them.
11. Runtime-only resources live on state or runtime-managed owners, not tasks.
12. Metrics/rewards/setup/update/cleanup are decorated with `@vf.*`.
13. One-off helper methods and bottom-of-file helper functions are absent.
14. Environment package install/load/eval has been validated with `prime eval run`
    or the relevant package-install test.

The deeper implementation reference lives in `verifiers/v1/README.md`.
