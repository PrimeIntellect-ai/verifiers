# BYO Harness

BYO Harness is the preferred `verifiers.v1` Taskset/Harness authoring path for
new environments that need a clean separation between the task being attempted
and the way a model attempts it.

Use this path when you want to bring your own harness: a tool loop, CLI program,
third-party Python program, sandboxed program, user simulator, MCP server, or
nested sub-harness workflow. For simple one-off environments, the core
[Environments](environments.md) guide remains the shortest path.

## Core Shape

![Task to Harness to State](assets/v1-task-harness-state.svg)

v1 environments are composed from:

- `Taskset`: task rows, task-owned tools, user behavior, metrics, rewards, and
  cleanup;
- `Harness`: rollout behavior, model endpoint forwarding, program execution,
  harness-owned tools, sandboxes, and nested harness calls;
- `Env`: adapter that makes a taskset/harness pair usable by eval and training
  workers.

The smallest v1 environment only needs a taskset. If no harness is passed,
`vf.Env` uses the base endpoint-backed harness.

Keep the boundary strict: if a tool defines the task's action space,
observations, success condition, or domain state, put it on the `Taskset`.
Harnesses should own only execution adapters and framework-specific mechanics.
For example, a Wikispeedia taskset owns `click_link` and `go_back`; a
LangChain, OpenAI Agents, CLI, or base harness should consume those tools from
runtime state instead of constructing its own copy.

```python
import verifiers as vf


@vf.reward(weight=1.0)
async def contains_answer(task, state) -> float:
    return float(task["answer"] in str(state.get("completion") or ""))


class ReverseTasksetConfig(vf.TasksetConfig):
    split: str = "train"
    rewards: list[vf.CallableConfig] = [
        vf.CallableConfig(fn="my_env:contains_answer", weight=1.0)
    ]


class ReverseTaskset(vf.Taskset[ReverseTasksetConfig]):
    def rows(self) -> list[dict[str, object]]:
        rows = [
            {
                "system_prompt": "Reverse text exactly.",
                "prompt": [{"role": "user", "content": "Reverse abc."}],
                "answer": "cba",
                "split": "train",
                "max_turns": 1,
            }
        ]
        return [row for row in rows if row["split"] == self.config.split]


class ReverseEnvConfig(vf.EnvConfig):
    taskset: ReverseTasksetConfig = ReverseTasksetConfig()
    harness: vf.HarnessConfig = vf.HarnessConfig()


def load_taskset(config: ReverseTasksetConfig = ReverseTasksetConfig()):
    return ReverseTaskset(config=config)


def load_environment(config: ReverseEnvConfig = ReverseEnvConfig()):
    return vf.Env(taskset=load_taskset(config=config.taskset))
```

## Tasksets

Tasksets usually own row loading directly. Config should hold user-facing knobs,
such as dataset name, split, or size limits; the taskset class should turn those
values into rows.

```python
from datasets import load_dataset
import verifiers as vf


class GSM8KTasksetConfig(vf.TasksetConfig):
    dataset_name: str = "gsm8k"
    split: str = "train"


class GSM8KTaskset(vf.Taskset[GSM8KTasksetConfig]):
    def rows(self) -> list[dict[str, object]]:
        dataset = load_dataset(
            self.config.dataset_name,
            "main",
            split=self.config.split,
        )
        return [
            {
                "example_id": index,
                "prompt": [{"role": "user", "content": row["question"]}],
                "answer": row["answer"],
            }
            for index, row in enumerate(dataset)
        ]


def load_taskset(config: GSM8KTasksetConfig = GSM8KTasksetConfig()):
    return GSM8KTaskset(config=config)
```

Rows are JSON-serializable mappings. The base taskset normalizes each row into a
stable task payload for eval and training workers.

Do not use a top-level string `task` field for routing. v1 tasksets serialize
the full task payload through `info["task"]` for worker compatibility, and
environment routing uses `info["env_id"]`.

## Shared Dependencies

Shared dependencies live on the taskset and are injected into named lifecycle or
scoring functions through bindings:

```python
import re
import verifiers as vf


class AnswerExtractor:
    def __init__(self):
        self.pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

    def __call__(self, completion: list[dict[str, object]]) -> str:
        message = vf.get_messages(completion, role="assistant")[-1]
        text = str(message.content or "")
        match = self.pattern.search(text)
        return "" if match is None else match.group(1).strip()


@vf.reward
async def exact(task, state, extract_answer) -> float:
    response = extract_answer(state.get("completion") or [])
    return float(response == task["answer"])


def build_answer_extractor() -> AnswerExtractor:
    return AnswerExtractor()


class ExtractTasksetConfig(vf.TasksetConfig):
    rewards: list[vf.CallableConfig] = [vf.CallableConfig(fn="my_env:exact")]
    objects: dict[str, str] = {"extract_answer": "my_env:build_answer_extractor"}
    bindings: dict[str, str] = {"exact.extract_answer": "objects.extract_answer"}


class ExtractTaskset(vf.Taskset[ExtractTasksetConfig]):
    def rows(self) -> list[dict[str, object]]:
        return [
            {
                "prompt": [{"role": "user", "content": "What is 2 + 2?"}],
                "answer": "4",
            }
        ]


class ExtractEnvConfig(vf.EnvConfig):
    taskset: ExtractTasksetConfig = ExtractTasksetConfig()
    harness: vf.HarnessConfig = vf.HarnessConfig()


def load_environment(config: ExtractEnvConfig = ExtractEnvConfig()) -> vf.Env:
    return vf.Env(taskset=ExtractTaskset(config=config.taskset))
```

Config `objects` values are import refs to zero-argument factories. Runtime
mutation APIs can still accept live objects, but serialized configs should stay
on the import-ref side of the boundary.

## Message Access

Taskset/harness environments expose one transcript selector:

```python
messages = vf.get_messages(state.get("completion") or [], role="assistant")
response = str(messages[-1].content or "") if messages else ""

assistant_turns = len(vf.get_messages(state.get("completion") or [], role="assistant"))
```

Use `vf.get_messages(...)` to get the transcript as typed message objects,
optionally filtered by role. Index or slice the returned list with ordinary
Python. The helper does not parse answers; task-specific extraction belongs in
ordinary Python or a taskset-bound object.

Keep rollout-loop data manipulation explicit. A few lines that read
`state["completion"]`, select messages, inspect task fields, or build a prompt
should usually be written directly where they are used, not hidden behind a
library helper or a one-off private function. Helpers are appropriate when the
logic is reused in multiple places, when a taskset-bound object is part of the
environment contract, or when complex behavior belongs in a named secondary
module. Do not create buried `utils` imports just to avoid three clear lines in
a reward, update, setup, or program function.

## Task Controls

Tasks can request rollout behavior through top-level serializable fields:

- `max_turns`: per-rollout turn limit for the base harness loop;
- `tools`: tool visibility as `{"show": [...]}` or `{"hide": [...]}`;
- `toolsets`: toolset visibility or rollout-local toolsets;
- `sandbox`: per-task overrides for a sandboxed program;
- `program`: per-task files, dirs, env, setup, artifacts, bindings, and command
  args.

Priority is:

```text
explicit state.runtime > task top-level controls > harness defaults
```

Keep system instructions out of `prompt`. v1 resolves `system_prompt` from the
task, taskset, and harness as a separate field; the base harness concatenates
the resolved system messages with `prompt` only when it submits a model request.
If more than one source provides a system prompt, resolution fails unless the
harness explicitly sets a merge policy.

`state.runtime` comes from explicit standalone state passing, `Taskset.init_group`
customization, or eval/training model controls. For normal tasksets, use
top-level task controls:

```python
yield {
    "prompt": [{"role": "user", "content": "Use the search tool."}],
    "max_turns": 5,
    "tools": {"show": ["search"]},
}
```

`task.runtime` is not part of the public task schema. Runtime metadata lives on
`state.runtime` and is written by the harness, the taskset group initializer, or
the eval/training worker.

Use `task.program` when a taskset owns files or environment variables that a
reusable harness should consume. The taskset cannot change the harness command
or tool channel; duplicate keys across the taskset and harness fail.

## Toolsets

Tools are packaged as `Toolset` objects. A taskset can own tools directly:

```python
async def search(query: str, index) -> str:
    return index.search(query)


toolset = vf.Toolset(
    tools=[search],
    objects={"index": load_index},
    bindings={"search.index": "objects.index"},
)

class SearchTaskset(vf.Taskset):
    def rows(self) -> list[dict[str, object]]:
        return [
            {
                "prompt": [{"role": "user", "content": "Search for docs."}],
                "answer": "example",
            }
        ]


taskset = SearchTaskset(config=vf.TasksetConfig())
taskset.add_toolset(toolset)
```

Bindings inject hidden arguments that the model does not see. Common binding
roots are `task.*`, `state.*`, and `tools.*`. Tasksets, toolsets, and users can
also bind `objects.*` from their own private dependency factories.
String binding sources are always framework paths. Use a callable source for
literal string values so misspelled paths fail during setup.

Custom harness programs can adapt taskset-owned tools through `state.get_tools()`.
That keeps the same taskset reusable across the base harness, a third-party
agent framework, and CLI or sandbox harnesses:

```python
async def run_agent_framework(task: vf.Task, state: vf.State) -> vf.State:
    tools = state.get_tools()
    agent_tools = [tools[name] for name in ("search", "lookup") if name in tools]
    result = await framework_agent(task["prompt"], tools=agent_tools)
    state["completion"] = [{"role": "assistant", "content": result}]
    return state
```

Wrap the returned callables only at the framework boundary when a library
requires its own tool object type.

If the harness has to know domain-specific tool internals, the taskset/harness
boundary is probably in the wrong place. Move the toolset and hidden bindings
back to the taskset, then let the harness adapt the resolved callables.

MCP servers are also tools:

```python
class FetchTasksetConfig(vf.TasksetConfig):
    toolsets: list[dict[str, object]] = [
        {
            "tools": [
                {"command": "uvx", "args": ["mcp-server-fetch"]},
            ]
        }
    ]


class FetchTaskset(vf.Taskset[FetchTasksetConfig]):
    def rows(self) -> list[dict[str, object]]:
        return [
            {
                "prompt": [{"role": "user", "content": "Fetch example.com."}],
                "answer": "example",
            }
        ]


taskset = FetchTaskset(config=FetchTasksetConfig())
```

## Harnesses

Create a harness when rollout behavior is no longer just "call the model with
the resolved taskset tools."

```python
class AgentHarnessConfig(vf.HarnessConfig):
    program: vf.ProgramConfig = vf.ProgramConfig(fn="my_env.program:run")


def load_harness(config: AgentHarnessConfig = AgentHarnessConfig()):
    return vf.Harness(config=config)


def load_environment(config: vf.EnvConfig):
    return vf.Env(
        taskset=load_taskset(config=config.taskset),
        harness=load_harness(config=config.harness),
    )
```

`Harness.program` can be:

| Form | Meaning |
| --- | --- |
| `None` | default endpoint-backed tool loop |
| callable | Python program called in-process |
| `{"fn": "pkg.module:run"}` | importable Python program |
| `{"command": ["cmd", "arg"]}` | local or sandboxed command |
| `{"sandbox": True}` | sandboxed default loop |

All model calls go through the v1 interception endpoint so trajectory capture,
tool forwarding, and protocol translation share one path.

Sandbox command programs can request the resolved tools as an MCP server with
`program={"command": [...], "sandbox": True, "channels": "mcp"}`. Python programs
receive callable tool handles by default, or can set
`program={"sandbox": True, "channels": "callable"}` when the base loop is moved
into a sandbox. `program.channels` supports only the generic `callable` and `mcp`
channels. Harness-specific tool carriers, such as RLM skill uploads, should
live on the taskset upload directory contract or the harness config.

For sandboxed `program.fn` refs, v1 resolves the owning local package from the
resolved module root: single-file modules use `pyproject.toml` in the same
directory as the module file, and package modules use `pyproject.toml` inside
the package directory. v1 uploads that package and installs it in the program
sandbox. Package dependencies are normal `[project.dependencies]`.

Programs are also the right shape for LLM-free replay:

```python
async def replay_solution(task, state):
    state["answer"] = task["answer"]
    state.stop("replayed")
    return state


@vf.reward
async def exact(task, state) -> float:
    return float(state.get("answer") == task.get("answer"))


class ReplayTasksetConfig(vf.TasksetConfig):
    rewards: list[vf.CallableConfig] = [vf.CallableConfig(fn="my_env:exact")]


class ReplayTaskset(vf.Taskset[ReplayTasksetConfig]):
    def rows(self) -> list[dict[str, object]]:
        return [
            {
                "prompt": [{"role": "user", "content": "Say the answer."}],
                "answer": "done",
            }
        ]


class ReplayHarnessConfig(vf.HarnessConfig):
    program: vf.ProgramConfig = vf.ProgramConfig(fn="my_env:replay_solution")


class ReplayEnvConfig(vf.EnvConfig):
    taskset: ReplayTasksetConfig = ReplayTasksetConfig()
    harness: ReplayHarnessConfig = ReplayHarnessConfig()


def load_environment(config: ReplayEnvConfig = ReplayEnvConfig()):
    return vf.Env(
        taskset=ReplayTaskset(config=config.taskset),
        harness=vf.Harness(config=config.harness),
    )
```

Use this for cached completions, deterministic solvers, and gold-solution
validation. Subclass `Harness` only when packaging reusable behavior with a new
config surface; do not subclass `Env` just to bypass inference.

Packaged CLI harnesses should use the same boundary. These implementations live
under `verifiers.v1.packages` while the v1 surface stabilizes, and are
re-exported through `verifiers.v1`. `OpenCode`, `Pi`, `MiniSWEAgent`,
`Terminus2`, and `RLM` are bundled `Harness` leaf wrappers for common
command-line agents:

```python
class HarborEnvConfig(vf.EnvConfig):
    taskset: vf.HarborTasksetConfig = vf.HarborTasksetConfig()
    harness: vf.OpenCodeConfig = vf.OpenCodeConfig()


def load_environment(config: HarborEnvConfig = HarborEnvConfig()):
    return vf.Env(
        taskset=vf.HarborTaskset(config=config.taskset),
        harness=vf.OpenCode(config=config.harness),
    )
```

`HarborTaskset(config=vf.HarborTasksetConfig())` loads Harbor-format task
directories from the environment package's reserved `tasks/` directory. Set
`dataset = "owner/name"` on the config to fetch a Harbor Hub dataset. The
taskset owns Harbor task loading, sandbox overrides, task uploads, and test
scoring. CLI harnesses own CLI
installation/config/run behavior and work with any taskset that supplies a
prompt.
Tasksets can expose package-owned upload directories with `get_upload_dirs()`.
The base `Taskset` discovers a sibling `skills/` directory by default, and
`RLM` uploads that directory to `/rlm/skills` unless `skills=` is passed
explicitly to the harness.
Use `RLMConfig` in `env.harness` for RLM-specific settings such as
`rlm_repo_ref`, `rlm_tools`, `rlm_max_turns`, and `summarize_at_tokens`.

## Setup, Updates, Signals, And Cleanup

![v1 composition lifecycle](assets/v1-composition-lifecycle.svg)

Setup functions, update functions, metrics, rewards, and advantages are
lifecycle functions around program execution and the rollout/group scoring
boundary.

```python
@vf.metric
async def turns(task, state) -> float:
    return float(len(state["trajectory"]))


@vf.reward(weight=1.0)
async def correct(task, state) -> float:
    return float(task["answer"] in str(state.get("completion") or ""))


@vf.reward(stage="group")
async def best_of_n(tasks, states) -> list[float]:
    ...
```

Rollout signals can request framework args such as `task`, `state`,
`completion`, and `prompt`, plus hidden args supplied by taskset or toolset
bindings. Group signals can request `tasks`, `states`, and bound hidden args,
and must return one value per state. Setup functions use `@vf.setup` and run
before the program body; update functions use `@vf.update` and run before
scoring; cleanup functions use `@vf.cleanup` and run after scoring; teardown
functions use `@vf.teardown`.

For sandbox command/Python programs, program files, directories, setup commands,
state handoff, and channel setup are framework setup contributions with
fixed priorities. User `@vf.setup(priority=...)` handlers can intentionally run
before or after those built-ins without adding new lifecycle hooks.

`env.requires_group_rollouts` is true when group-stage updates, scoring,
cleanup, or group setup are part of the environment contract.
`env.provides_advantages` is true when the environment has explicit advantage
handlers.

## TOML Config

Eval and RL TOML own the outer run: model, endpoint, sampling, rollout count,
and trainer/eval settings. v1 config owns taskset and harness behavior inside
the environment package.

The recommended loader takes one `config` object and routes its `taskset` and
`harness` sections:

```python
def load_environment(config: vf.EnvConfig):
    return vf.Env(
        taskset=load_taskset(config=config.taskset),
        harness=load_harness(config=config.harness),
    )
```

Eval config passes v1 config through the `taskset`/`harness` sections:

```toml
model = "openai/gpt-5.4-mini"
num_examples = 5
rollouts_per_example = 3

[[eval]]
env_id = "my-v1-env"
sampling_args = { max_tokens = 4096 }

[eval.harness]
max_turns = 4

[eval.taskset.scoring.exact_answer]
weight = 0.5
```

For environment-specific settings, define leaf fields on the taskset or harness
config that owns them. An `EnvConfig` subclass only fixes the concrete taskset
and harness config types for the loader.

```python
class MyTasksetConfig(vf.TasksetConfig):
    split: str = "train"


class MyEnvConfig(vf.EnvConfig):
    taskset: MyTasksetConfig
    harness: vf.HarnessConfig


def load_taskset(config: MyTasksetConfig):
    ...


def load_harness(config: vf.HarnessConfig):
    ...


def load_environment(config: MyEnvConfig):
    return vf.Env(
        taskset=load_taskset(config=config.taskset),
        harness=load_harness(config=config.harness),
    )
```

RL and Hosted Training config uses the same shape under `env`:

```toml
model = "Qwen/Qwen3-30B-A3B-Instruct-2507"
max_steps = 100
batch_size = 256
rollouts_per_example = 8

[sampling]
max_tokens = 4096

[[env]]
id = "primeintellect/my-v1-env"

[env.harness]
max_turns = 8

[env.taskset]
split = "train"

[env.taskset.toolsets.search]
tools = ["my_env.tools:search"]
bindings = { "search.index" = "objects.index" }
```

Taskset and harness sections can import a base config with `config` and then
overlay local fields. Collection fields extend the imported config.

```toml
[env.harness]
config = "my_env.configs:load_another_harness_config"

[[env.harness.rewards]]
fn = "my_env.rewards:new_reward_func"
weight = 0
```

Callable config uses `fn = "module:callable"` when metadata is needed:

```toml
[[env.taskset.rewards]]
fn = "my_env.signals:exact_answer"
weight = 1.0
priority = 0
```

The callable name is always its Python function name. Use
`[...scoring.function_name]` to tune or skip an existing metric/reward without
creating a new signal.

For command harnesses, keep endpoint and tool registration under the requested
`program.channels` channel:

```toml
[env.harness.program]
command = ["my-cli", "run", "--config", "/tmp/my-cli.json"]
sandbox = true

[env.harness.program.channels]
mcp = { fn = "my_env.cli:write_cli_config" }

[env.harness.program.bindings]
"write_cli_config.endpoint_config" = { fn = "my_env.cli:endpoint_config" }
```

The implementation details for TOML refs, toolset tables, row loading, program
bindings, and custom config subclasses are in
`verifiers/v1/README.md`.

## When To Use Which Path

Use the core `SingleTurnEnv`, `ToolEnv`, and `MultiTurnEnv` docs when you want
the shortest path through the established environment classes.

Use BYO Harness when you want reusable tasksets, reusable harnesses, task-owned
or harness-owned toolsets, third-party Python programs, sandboxed programs,
stateful users, MCP tools, or nested harness calls.

The repository also includes a deeper implementation guide at
`verifiers/v1/README.md`.
