# Research Environments v1 Migration

This guide maps older research-environments packages onto the current v1
Taskset/Harness shape. The authoritative implementation guide is
[`docs/byo-harness.md`](../../docs/byo-harness.md); this file is a migration map
for choosing the right v1 pattern.

## Migration Contract

Every migrated v1 package should expose the same loader boundary:

```python
import verifiers as vf


class MyTasksetConfig(vf.TasksetConfig):
    system_prompt: vf.SystemPrompt = "Answer exactly."


class MyTaskset(vf.Taskset[MyTasksetConfig]):
    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        if split == "eval":
            return []
        return [
            {
                "prompt": [{"role": "user", "content": "Question?"}],
                "answer": "Answer",
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

Add `load_harness(config: MyHarnessConfig)` only when the package owns a
reusable execution mechanism such as a command agent, framework adapter,
sandbox placement policy, or endpoint interception behavior:

```python
class MyHarnessConfig(vf.HarnessConfig):
    program: vf.ProgramConfig = vf.ProgramConfig(fn="my_env.agent:run")


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

Do not subclass `vf.EnvConfig` to narrow child config types. The
`load_taskset` and `load_harness` annotations are the child config contract for
Python, TOML, CLI, eval, GEPA, RL, and Hosted Training.

## Pattern Map

Pick the row that matches the old package, copy the referenced v1 shape, and
then port the dataset and scoring logic.

| Old package shape | Reference | v1 pattern |
| --- | --- | --- |
| AIME, GPQA, MATH500, MMLU-Pro, SimpleQA | `environments/reverse_text/reverse_text_v1.py` | serializable task rows, base `vf.Harness`, taskset reward |
| instruction-following or string-matching tasks | `environments/reverse_text/reverse_text_v1.py` | prompt taskset plus class-local metrics/rewards |
| Python execution math/code tasks | `environments/math_python/math_python_v1.py` | sandbox-backed callable tool |
| web/search/browser tasks | `environments/wiki_search/wiki_search_v1.py` | taskset-owned `vf.Toolset` with bound private objects |
| BFCL-style task-provided schemas | `environments/bfcl_v3/bfcl_v3.py` | task-local tool visibility/schemas and state-recorded calls |
| games and simulators | `environments/tau2_bench_v1/tau2_bench_v1.py` | taskset-owned `vf.User` subclass |
| stdio MCP tasks | `environments/mcp_search_env/mcp_search_env.py` | `vf.MCPTool` entries in a `vf.Toolset` |
| helper agents or judges | `environments/hello_subagent_v1/hello_subagent_v1.py` | nested `vf.Harness.run(...)` from a tool/update/reward |
| shared sandbox helper agents | `environments/hello_parallel_sandbox_v1/hello_parallel_sandbox_v1.py` | borrowed runtime state through `state.for_task(...)` |
| Harbor task directories | `environments/opencode_harbor/opencode_harbor.py` | `HarborTaskset` plus a packaged command harness |
| OpenCode, Pi, mini-swe-agent, Terminus, RLM | `packages/harnesses/` | reusable `vf.Harness` subclasses with typed config |
| OpenEnv, OpenReward, TextArena, NeMoGym | `packages/tasksets/` | reusable `vf.Taskset` subclasses matching upstream formats |

## Task Data

Tasksets return serializable task records from `load_tasks(split=...)`. During
a rollout, the framework materializes them as immutable `vf.Task` objects.

```python
yield {
    "prompt": [{"role": "user", "content": question}],
    "answer": answer,
    "info": {"source_id": source_id},
    "max_turns": 8,
}
```

Use top-level fields for task controls:

- `prompt`: user/developer/tool messages, never system messages.
- `system_prompt`: per-task system instructions.
- `answer`: reference answer or target data.
- `info`: serializable metadata.
- `max_turns`: per-task base-loop turn limit.
- `toolsets`: toolset visibility with `{"show": [...]}` or `{"hide": [...]}`.
- `tools`: per-toolset tool visibility with `{"search": {"show": [...]}}`.
- `sandbox`: task-owned sandbox override.
- `program`: task-owned program files, dirs, setup, env, artifacts, bindings,
  and args.
- `artifacts`: task-owned artifacts collected after program execution.

Do not ask users to manage task IDs. Preserve upstream IDs only when they are
meaningful task metadata.

## System Prompts

Static system prompts belong in the config that owns the policy. Use taskset
config for task policy and harness config for execution or agent policy:

```python
class PromptTasksetConfig(vf.TasksetConfig):
    system_prompt: vf.SystemPrompt = "Answer concisely."
```

File-backed GEPA prompts should also be config:

```python
class PromptTasksetConfig(vf.TasksetConfig):
    system_prompt: vf.SystemPromptConfig = vf.SystemPromptConfig(
        path="system_prompt.txt"
    )
```

Override `load_system_prompt(config)` only when prompt construction is computed
from config fields or package resources. Do not put system messages in
`task["prompt"]`. System prompt resolution is per task: task prompt overrides
taskset prompt for the taskset side, then `HarnessConfig.system_prompt_strategy`
resolves the taskset side against the harness side. The available strategies are
`HT`, `TH`, `H_OR_T`, `T_OR_H`, `H`, `T`, and `REJECT`; the default is `HT`.

## Single-Turn QA And Instruction Following

Use the base harness unless the old environment owns a reusable execution
mechanism. Move dataset construction into `load_tasks(split=...)`, and move
each reward/metric onto the taskset class.

```python
class QATasksetConfig(vf.TasksetConfig):
    dataset_name: str = "gsm8k"


class QATaskset(vf.Taskset[QATasksetConfig]):
    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        dataset_split = "test" if split == "eval" else "train"
        return load_dataset(self.config.dataset_name, "main", split=dataset_split)

    @vf.reward(weight=1.0)
    async def exact(self, task: vf.Task, state: vf.State) -> float:
        messages = vf.get_messages(state.get("completion") or [], role="assistant")
        response = str(messages[-1].content or "") if messages else ""
        return float(str(task["answer"]).strip() in response)
```

Return the source dataset directly when it already has standard fields such as
`question` and `answer`; v1 derives `prompt` from `question`. Transform records
only when the source does not match the task contract.

Judge/extractor dependencies should be private objects with explicit bindings:

```python
class ExtractTasksetConfig(vf.TasksetConfig):
    objects: vf.ObjectsConfig = vf.ObjectsConfig.model_validate(
        {"extract_answer": "my_env.extractors:load_extractor"}
    )
    bindings: vf.BindingsConfig = vf.BindingsConfig.model_validate(
        {"exact.extract_answer": "objects.extract_answer"}
    )
```

The reward reads serializable `task`/`state` and receives the bound dependency
as an argument:

```python
@vf.reward(weight=1.0)
async def exact(self, task: vf.Task, state: vf.State, extract_answer) -> float:
    return float(extract_answer(state.get("completion") or []) == task["answer"])
```

## Callable Tools

Tools that define the task action space belong to the taskset. Expose them
through `vf.Toolset`; hide private dependencies behind `objects` and `bindings`.

```python
async def search(query: str, exa) -> str:
    return await exa.search(query)


async def open_page(url: str, exa) -> str:
    return await exa.open(url)


class SearchTasksetConfig(vf.TasksetConfig):
    objects: vf.ObjectsConfig = vf.ObjectsConfig.model_validate(
        {"exa": "my_env.search:load_exa"}
    )
    bindings: vf.BindingsConfig = vf.BindingsConfig.model_validate(
        {
            "search.search.exa": "objects.exa",
            "search.open_page.exa": "objects.exa",
        }
    )


class SearchTaskset(vf.Taskset[SearchTasksetConfig]):
    def load_toolsets(self, config: SearchTasksetConfig) -> vf.Toolsets:
        return {"search": vf.Toolset(tools=[search, open_page])}
```

Task rows show all toolsets/tools by default and can restrict visibility:

```python
yield {
    "prompt": [{"role": "user", "content": "Search only the docs."}],
    "toolsets": {"show": ["search"]},
    "tools": {"search": {"show": ["search"]}},
}
```

Use rollout-scoped toolsets for session-backed tools. Keep live backend handles
on `state`; keep task records serializable.

## Users

Use a `vf.User` subclass when the environment replies with user messages after
model turns. Users are not callables.

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

Use a user for simulator observations. Use tools when the model must select an
explicit schema action. Use setup/update handlers when state should change
without adding conversation messages.

## MCP Toolsets

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

The runtime materializes MCP tools as callable handles for Python programs and
can expose resolved toolsets through the generic `mcp` program channel for
command harnesses. `program.channels` names the program-facing channel, not a
specific tool.

## Programs, Harnesses, And Sandboxes

Use `HarnessConfig.program` for the executable behavior:

```python
class AgentHarnessConfig(vf.HarnessConfig):
    sandbox: vf.SandboxConfig = vf.SandboxConfig(image="python:3.11-slim")
    program: vf.ProgramConfig = vf.ProgramConfig(
        command=["agent", "run"],
        sandbox=True,
        channels={"mcp": {"setup": ["agent mcp add vf ${VF_MCP_URL}"]}},
    )
```

Tasksets can contribute task-owned files, dirs, setup, env, artifacts,
bindings, and command args through `task["program"]`. The harness still owns
the program kind (`base`, `fn`, or `command`), channel wiring, and primary
sandbox placement.

```python
yield {
    "prompt": [{"role": "user", "content": instruction}],
    "sandbox": {"image": "python:3.12-slim"},
    "program": {
        "files": {
            "/task/instruction.md": {"task": "instruction"},
        },
        "env": {"TASK_ID": {"task": "info.source_id"}},
        "artifacts": {
            "agent_log": {
                "path": "/workspace/agent.log",
                "format": "text",
                "optional": True,
            }
        },
    },
}
```

Use task sandbox overrides only when the taskset owns per-task images, files,
resource sizing, or setup. Put reusable execution policy on the harness.

## Nested Harnesses

Nested harnesses are regular harness runs. A tool, update, reward, or program
can create a child task and run a child harness. Bind the child harness through
the owning object config or construct it inside the owning Python object; do not
store live harness objects in task data.

```python
async def ask_child(name: str, child_harness: vf.Harness, state: vf.State) -> str:
    child_task = vf.Task(
        {"prompt": [{"role": "user", "content": f"Say hello to {name}."}]}
    ).freeze()
    child_state = await child_harness.run(child_task)
    messages = vf.get_messages(child_state.get("completion") or [], role="assistant")
    return str(messages[-1].content or "") if messages else ""
```

Borrow runtime handles only when the child intentionally reuses live parent
resources:

```python
child_state = state.for_task(child_task, borrow="model", tools=["search"])
```

Borrowed resources are process-local runtime handles and are stripped before
serialization.

## Packaged Tasksets And Harnesses

Prefer the sibling packages when an upstream format already matches:

```bash
uv add "verifiers[tasksets]"
uv add "verifiers[harnesses]"
uv add "verifiers[openenv]"
uv add "verifiers[openreward]"
uv add "verifiers[ta]"
uv add "verifiers[nemogym]"
```

```python
from harnesses import OpenCode, OpenCodeConfig
from tasksets import HarborTaskset, HarborTasksetConfig


def load_taskset(config: HarborTasksetConfig) -> HarborTaskset:
    return HarborTaskset(config=config)


def load_harness(config: OpenCodeConfig) -> OpenCode:
    return OpenCode(config=config)
```

`HarborTaskset` owns Harbor task loading, task sandbox overrides, task uploads,
and test scoring. Command harnesses own installation, endpoint wiring, config
generation, channel setup, and log artifacts. The two sides communicate only
through task controls, program config, sandbox config, state, and lifecycle
handlers.

## Migration Checklist

1. The root loader is `load_environment(config: vf.EnvConfig)`.
2. Custom tasksets have `load_taskset(config: MyTasksetConfig)`.
3. Custom harnesses have `load_harness(config: MyHarnessConfig)`.
4. `Taskset`, `Harness`, and `User` subclasses do not override `__init__`.
5. Static prompts are config fields; computed prompts use `load_system_prompt`.
6. Task data is serializable and does not contain live handles.
7. Runtime handles live on `state` or framework runtime owners.
8. Tasksets own task data, tools, users, metrics, rewards, and task behavior.
9. Harnesses own programs, endpoint routing, sandboxes, command agents, and
   execution artifacts.
10. Tools are exposed through `vf.Toolset`; tasks show/hide tools and toolsets.
11. Private dependencies use `ObjectsConfig` plus `BindingsConfig`.
12. Lifecycle logic is on the owning class with `@vf.*` decorators.
13. No one-off bottom-of-file helpers are needed for ordinary implementations.
14. The install/load/eval path has been validated with `prime eval run` or the
    relevant package-install test.
