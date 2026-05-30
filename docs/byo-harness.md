# v1 Taskset/Harness Environments

Use the v1 Taskset/Harness path for reusable environments: dataset adapters,
tool environments, user simulators, sandboxed programs, command agents,
framework harnesses, packaged benchmark formats, and environments that need the
same config shape from Python, TOML, eval, GEPA, RL, and Hosted Training.

For short v0-style `SingleTurnEnv`, `ToolEnv`, or `MultiTurnEnv` examples, see
[Environments](environments.md). For API signatures, see
[Reference](reference.md).

## Start From The Template

Initialize the v1 Taskset/Harness template first:

```bash
prime env init my-env --v1
```

Use a custom harness template only when the environment owns reusable execution
behavior such as a command agent, framework adapter, sandboxed program, browser
loop, endpoint interceptor, or nested harness:

```bash
prime env init my-env --v1 --with-harness
```

Open the generated `my_env.py` and edit it in this order:

1. Add user-facing task settings to `MyTasksetConfig`.
2. Fill `MyTaskset.load_tasks()` with train/eval task records.
3. Add task-owned tools with `MyTaskset.load_toolsets()` when the task defines
   an action space.
4. Add task behavior with `@vf.setup`, `@vf.update`, `@vf.reward`, `@vf.metric`,
   `@vf.cleanup`, and related lifecycle methods on `MyTaskset`.
5. Add a `User` subclass and `load_user()` when the task owns simulated user
   behavior.
6. If `--with-harness` is used, put execution-level program, sandbox, endpoint,
   model, or harness lifecycle behavior on `MyHarness`.
7. Keep the generated loaders as the typed entrypoints: `load_taskset`,
   optional `load_harness`, and the root `load_environment`.

## Golden Shape

Every v1 environment has one root loader and typed child loaders:

```python
import verifiers as vf


class MyTasksetConfig(vf.TasksetConfig):
    system_prompt: vf.SystemPrompt = "Answer exactly."


class MyTaskset(vf.Taskset[MyTasksetConfig]):
    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        """Return serializable task records as a list, generator, or Dataset."""
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
    """Loader pattern for all Taskset/Harness environments."""
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
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
    state["completion"] = [{"role": "assistant", "content": message.content or ""}]
    return state


class MyHarnessConfig(vf.HarnessConfig):
    program: vf.ProgramConfig = vf.ProgramConfig(fn="my_env:run_agent")


class MyHarness(vf.Harness[MyHarnessConfig]):
    pass


def load_harness(config: MyHarnessConfig) -> MyHarness:
    return MyHarness(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    """Loader pattern for all Taskset/Harness environments."""
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
    )
```

The child loader annotations are load-bearing. `load_taskset(config:
MyTasksetConfig)` defines the `[env.taskset]` schema; `load_harness(config:
MyHarnessConfig)` defines the `[env.harness]` schema. Keep
`load_environment(config: vf.EnvConfig)` as-is: implement the config surface through taskset and harness configs, not root loader kwargs.

Start with a taskset and the base harness. Add a custom harness only when the
environment owns a reusable execution protocol such as a command agent,
third-party framework adapter, browser loop, endpoint interceptor, primary
sandbox placement, or program runner.

## Implementation Map

- Import as `import verifiers as vf`.
- Use `XXXConfig` classes for structured settings.
- Put task behavior on the taskset config/class.
- Put execution behavior on the harness config/class.
- Use `vf.Env` and `vf.EnvConfig` for ordinary environment packages.
- Let the base `Taskset`, `Harness`, and `User` constructors handle
  construction; customize with config fields and public methods.
- Put taskset/harness behavior on the owning class with standard public methods
  or `@vf.*` decorators.
- Use `system_prompt` for system messages.
- Keep reusable multi-line internals in utility modules with clear names.

Utility modules are appropriate only for reused, nontrivial internals or messy
upstream adapters that users should not think about.

## Ownership

| Object | Owns |
| --- | --- |
| `Taskset` | Task loading, task data, task prompts, task controls, task-owned tools, user behavior, task-specific lifecycle, metrics, rewards, advantages, and task-owned program/sandbox inputs. |
| `Harness` | Rollout execution, execution-level system prompts, model/client defaults, programs, command agents, framework adapters, endpoint interception, primary sandbox placement, harness-owned tools, and execution artifacts. |
| `Env` | The adapter that makes one taskset/harness pair usable by eval and training workers. |

If a tool or state transition defines the task action space, observations, or
success condition, it belongs to the taskset. If a class describes how a model
or external agent attempts arbitrary tasks, it belongs to the harness.

Examples:

- Wikispeedia link tools belong to the Wikispeedia taskset.
- TextArena game state and user responses belong to the TextArena taskset.
- Harbor task directories, uploads, and tests belong to `HarborTaskset`.
- OpenCode, Pi, mini-swe-agent, Terminus, and RLM execution belong to harness
  classes.
- Endpoint routing and interception belong to the harness/runtime, not task
  rows.

## Config

Config values must be serializable. Use import-ref strings such as
`"my_env.module:factory"` when config needs to name a callable across TOML, CLI,
or package boundaries. Python constructors may pass runtime objects only where
the constructor explicitly accepts them, such as `vf.Toolset(tools=[...])` or
standalone `vf.Harness(model=..., client=...)`.

Common owner config fields:

| Field | Meaning |
| --- | --- |
| `system_prompt` | String, system-message list, or `vf.SystemPromptConfig`. |
| `user` | `UserConfig` subclass that materializes a registered `User`. |
| `toolsets` | Configured toolset collection. |
| `objects` | Private dependency factories owned by this object. |
| `bindings` | Hidden argument bindings for handlers, tools, users, and programs. |
| `artifacts` | Text/JSON artifacts owned by this object. |
| lifecycle lists | Import-ref `setups`, `updates`, `metrics`, `rewards`, `cleanups`, etc. |
| `scoring` | Per-handler tuning or skipping by handler name. |

Put taskset fields on `TasksetConfig`; put harness fields on `HarnessConfig`.
Avoid broad unions and untyped mappings unless arbitrary JSON is the actual
task payload.

## Tasks

Tasksets load train and eval data through `load_tasks(split=...)`:

```python
class GSM8KTasksetConfig(vf.TasksetConfig):
    dataset_name: str = "gsm8k"
    train_split: str = "train"
    eval_split: str = "test"
    num_examples: int | None = None


class GSM8KTaskset(vf.Taskset[GSM8KTasksetConfig]):
    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        """Return serializable task records as a list, generator, or Dataset."""
        dataset_split = (
            self.config.train_split if split == "train" else self.config.eval_split
        )
        dataset = load_dataset(self.config.dataset_name, "main", split=dataset_split)
        if self.config.num_examples is not None:
            dataset = dataset.select(range(self.config.num_examples))
        return dataset
```

`vf.Tasks` may be a `datasets.Dataset`, an iterable of serializable records, or
an iterable of `vf.Task` objects. During rollout, records become immutable
`vf.Task` objects.

Return a `datasets.Dataset` directly when the source already has standard
columns such as `question` and `answer`; the framework derives `prompt` from
`question`. Use config fields like `train_split` and `eval_split` only to map
v1 split names to upstream source split names. Do not add generic split config
that duplicates `load_tasks(split=...)`.

Common task fields:

| Field | Meaning |
| --- | --- |
| `prompt` | User/developer/tool messages. No system messages. |
| `system_prompt` | Per-task taskset-side system prompt override. |
| `answer` | Reference answer or target data. |
| `info` | Serializable metadata. |
| `max_turns` | Per-task base-loop turn limit. |
| `toolsets` | Toolset visibility: `{"show": [...]}` or `{"hide": [...]}`. |
| `tools` | Per-toolset tool visibility: `{"search": {"show": [...]}}`. |
| `sandbox` | Per-task sandbox override. |
| `program` | Task-owned files, dirs, setup, env, artifacts, bindings, and args. |
| `artifacts` | Task-owned artifacts collected after program execution. |

Users should not manage task/example IDs. Preserve upstream IDs only as ordinary
metadata when they matter.

Do not copy config defaults into every row. Use `max_turns`, `sandbox`,
`program`, and tool visibility fields in task records only when they genuinely
vary by example.

## System Prompts

System prompt resolution happens per task during rollout setup.

There are two sides:

- `T`: the resolved taskset side. `task["system_prompt"]` wins for that task;
  otherwise the taskset uses `TasksetConfig.system_prompt`.
- `H`: the harness side from `HarnessConfig.system_prompt`.

`HarnessConfig.system_prompt_strategy` decides how those two sides resolve:

| Strategy | Meaning |
| --- | --- |
| `HT` | Harness side followed by resolved taskset side. Default. |
| `TH` | Resolved taskset side followed by harness side. |
| `H_OR_T` | Harness side when present, otherwise resolved taskset side. |
| `T_OR_H` | Resolved taskset side when present, otherwise harness side. |
| `H` | Harness side only. |
| `T` | Resolved taskset side only. |
| `REJECT` | Error if both sides are present. |

Static prompts belong in config:

```python
class WordleTasksetConfig(vf.TasksetConfig):
    system_prompt: vf.SystemPrompt = (
        "Play Wordle. Submit guesses inside <guess>...</guess> tags."
    )
```

For GEPA or other file-backed prompt optimization, use config too:

```python
class WordleTasksetConfig(vf.TasksetConfig):
    system_prompt: vf.SystemPromptConfig = vf.SystemPromptConfig(
        path="system_prompt.txt"
    )
```

Override `load_system_prompt(config)` only when prompt loading is computed from
other config fields or package resources.

## Toolsets

Toolsets package model-visible schemas, hidden bindings, private objects,
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

Bindings inject hidden arguments that the model does not see. Common sources
include `task.*`, `state.*`, `objects.*`, and `tools.*`.

Tasks show all toolsets and tools by default. Restrict visibility in task data:

```python
yield {
    "prompt": [{"role": "user", "content": "Use the calculator only."}],
    "toolsets": {"show": ["math"]},
    "tools": {"math": {"show": ["calculate"]}},
}
```

Use rollout-scoped toolsets for resources that exist only during a rollout,
such as OpenReward sessions or sandbox-backed servers. Keep live backend
handles on `state`; keep task rows serializable. Dynamic schemas use
`state.add_tool("toolset_name", vf.Tool(...))` during rollout setup against a
named rollout toolset.

MCP servers are normal tool entries:

```python
class FetchTasksetConfig(vf.TasksetConfig):
    toolsets: dict[str, vf.ToolsetConfig] = {
        "fetch": vf.ToolsetConfig(
            tools=[vf.MCPToolConfig(command="uvx", args=["mcp-server-fetch"])],
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

Use a user when the environment naturally replies after model turns. Use tools
when the model chooses an explicit schema action. Use setup/update handlers when
state should change without adding conversation messages.

## Programs, Harnesses, And Sandboxes

`HarnessConfig.program` is a `vf.ProgramConfig`:

| Form | Meaning |
| --- | --- |
| `vf.ProgramConfig()` | Base endpoint-backed tool loop. |
| `vf.ProgramConfig(base=True)` | Explicit base loop, usually with sandbox options. |
| `vf.ProgramConfig(fn="my_env:run")` | Importable Python program. |
| `vf.ProgramConfig(command=["agent", "run"])` | Local or sandboxed command. |

The preferred Python program signature is:

```python
async def program(task: vf.Task, state: vf.State) -> vf.State:
    state["answer"] = task["answer"]
    return state
```

Programs may call models, call tools, run solvers, replay cached solutions, or
adapt third-party frameworks. They should read immutable task data, mutate
serializable state, and let lifecycle handlers collect artifacts and score.

Tasksets can contribute task-local program data through `task["program"]`.
Harnesses still own the program kind, channel wiring, and primary sandbox
placement. Duplicate files, env vars, artifacts, or bindings fail fast.

Put sandbox config on the harness when it is part of the execution mechanism:

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

Lifecycle decorators attach behavior to the owning class:

```python
class QAATaskset(vf.Taskset[QAATasksetConfig]):
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

Rollout handlers can request `task`, `state`, `completion`, `prompt`, and bound
hidden args. Group handlers use `tasks` and `states` and must return one value
per state when scoring.

## Objects, Bindings, And Artifacts

`objects` are private dependency factories. `bindings` connect those objects,
task fields, state fields, or runtime values to hidden callable arguments.

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

Artifacts are text/JSON files copied into serialized state:

```python
class AgentHarnessConfig(vf.HarnessConfig):
    artifacts: vf.ArtifactsConfig = vf.ArtifactsConfig.model_validate(
        {"agent_log": {"path": "/app/agent.log", "format": "text", "optional": True}}
    )
```

Artifacts can live on tasksets, harnesses, users, toolsets, programs, or tasks.
The owner determines which sandbox/filesystem is searched first.

## Nested Harnesses

Nested harnesses are ordinary harness runs. Create a child task, create a child
state, and run the child harness.

```python
async def ask_child(name: str, state: vf.State) -> str:
    harness = vf.Harness(
        config=vf.HarnessConfig(program=vf.ProgramConfig(fn="my_env.children:greet"))
    )
    task = vf.Task(
        {"prompt": [{"role": "user", "content": f"Say hello to {name}."}]}
    ).freeze()
    child_state = await harness.run(task, state.for_task(task))
    messages = vf.get_messages(child_state.get("completion") or [], role="assistant")
    return str(messages[-1].content or "") if messages else ""
```

Borrow runtime handles only when the child intentionally reuses live parent
resources:

```python
child_state = state.for_task(child_task, borrow="model", tools=["search"])
```

Borrowed resources are process-local and stripped before state serialization.

## Packaged Tasksets And Harnesses

Reusable implementations live in standalone packages under `packages/`:

```bash
uv add "verifiers[packages]"
uv add "verifiers[tasksets]"
uv add "verifiers[harnesses]"
uv add "verifiers[openenv]"
uv add "verifiers[openreward]"
uv add "verifiers[ta]"
uv add "verifiers[nemogym]"
```

Package-backed environments use the same loader shape:

```python
import verifiers as vf
from harnesses import OpenCode, OpenCodeConfig
from tasksets import HarborTaskset, HarborTasksetConfig


def load_taskset(config: HarborTasksetConfig) -> HarborTaskset:
    return HarborTaskset(config=config)


def load_harness(config: OpenCodeConfig) -> OpenCode:
    return OpenCode(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    """Loader pattern for all Taskset/Harness environments."""
    return vf.Env(
        taskset=vf.load_taskset(config=config.taskset),
        harness=vf.load_harness(config=config.harness),
    )
```

Tasksets include Harbor, OpenEnv, OpenReward, TextArena, and NeMoGym. Harnesses
include OpenCode, Pi, mini-swe-agent, Terminus, RLM, and NeMoGymHarness.

## TOML And CLI

Eval and training config owns the run: model, endpoint, sampling, examples, and
rollout count. v1 child config owns environment behavior:

```toml
model = "openai/gpt-5.4-mini"
num_examples = 5
rollouts_per_example = 3

[[eval]]
env_id = "my-v1-env"

[eval.sampling]
max_tokens = 4096

[eval.taskset]
system_prompt = "Answer exactly."

[eval.harness]
max_turns = 4
```

CLI overrides target typed child fields:

```bash
prime eval run my-v1-env --taskset.system-prompt "Answer exactly." --harness.max-turns 4
```

For package-only composition, TOML can name the taskset and harness packages
directly:

```toml
[[eval]]

[eval.taskset]
id = "tasksets.harbor"
tasks_dir = "tasks"

[eval.harness]
id = "harnesses.opencode"
max_turns = 8
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

## Checklist

Before publishing or asking for review:

1. `load_environment(config: vf.EnvConfig)` is the only root loader shape.
2. Custom tasksets have `load_taskset(config: MyTasksetConfig)`.
3. Custom harnesses have `load_harness(config: MyHarnessConfig)`.
4. No `Taskset`, `Harness`, or `User` subclass overrides `__init__`.
5. No ordinary environment subclass of `vf.Env` or `vf.EnvConfig` exists.
6. Config fields are serializable and named `XXXConfig`.
7. Taskset-owned behavior is not hidden in the harness.
8. Harness-owned execution is not hidden in task rows.
9. Static prompts live in config; computed prompts use `load_system_prompt`.
10. Tools are exposed through `vf.Toolset`; task rows only show/hide them.
11. Runtime-only resources live on state or runtime-managed owners.
12. Metrics/rewards/setup/update/cleanup are decorated with `@vf.*`.
13. Generated component loaders remain the typed taskset/harness entrypoints.
14. One-off helper methods and bottom-of-file helper functions are absent.
15. Install/load/eval has been validated with `prime eval run` or the relevant
    package-install test.
