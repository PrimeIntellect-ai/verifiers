# v1 Taskset/Harness Environments

v1 is the active-development Taskset/Harness API. Use it for new reusable
tasksets, reusable harnesses, MCP tools, user simulators, framework adapters,
endpoint interception, command agents, and runtime-backed environments.

v1 is intentionally separate from v0. Author v1 files with:

```python
import verifiers.v1 as vf
from verifiers.utils.response_utils import parse_response_message
```

The top-level `verifiers` package remains the v0 authoring surface. The public
loader can still load v1 packages, but v1 environment code should import from
`verifiers.v1` so the boundary is explicit.

## Package Shape

Start from the v1 template:

```bash
prime env init my-env --v1
prime env init my-env --v1 --with-harness
```

A v1 package is component-first:

```text
my_env/
  pyproject.toml
  README.md
  my_env/
    __init__.py
    taskset.py
    harness.py        # only when the package owns a custom harness
    servers/
      __init__.py
      toolset.py      # optional toolset implementation
      user.py         # optional user implementation
```

Do not define a package-level `load_environment`. The loader imports the package
and discovers `taskset.py` and `harness.py`.

## Minimal Taskset

```python
import verifiers.v1 as vf


class ReverseTask(vf.Task):
    answer: str


class ReverseTasksetConfig(vf.TasksetConfig):
    system_prompt: vf.SystemPrompt = "Return only the reversed string."


class ReverseTaskset(vf.Taskset[ReverseTasksetConfig]):
    task_type = ReverseTask

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        if split == "eval":
            return []
        return [
            {
                "row_id": 0,
                "prompt": [{"role": "user", "content": "Reverse abc."}],
                "answer": "cba",
                "max_turns": 1,
            }
        ]

    @vf.reward
    async def exact(self, task: ReverseTask, state: vf.State) -> float:
        return float(state.completion[-1].content == task.answer)


def load_taskset(config: ReverseTasksetConfig) -> ReverseTaskset:
    return ReverseTaskset(config=config)
```

`load_taskset(config: MyTasksetConfig)` is the taskset entrypoint and defines
the `[taskset]` config schema for eval, RL, GEPA, and hosted workers.

## Custom Harness

Use the base `vf.Harness` until the package owns a reusable execution protocol.
Add `harness.py` when the environment owns a command/program runner, framework
adapter, browser loop, nested harness, interception protocol, runtime placement,
or execution-level lifecycle.

```python
import verifiers.v1 as vf


class MyHarnessConfig(vf.HarnessConfig):
    max_turns: int = 3


class MyHarness(vf.Harness[MyHarnessConfig]):
    async def run_with_context(self, context: vf.Context) -> None:
        task = context.task
        state = context.state
        toolsets = context.toolsets
        prompt = self.initial_messages(task)
        response = await context.model_client.get_response(
            prompt=prompt,
            model=context.model,
            sampling_args=context.sampling_args,
            tools=toolsets.tools() if toolsets is not None else None,
            state=state,
        )
        turn = vf.Turn(
            prompt=prompt,
            completion=await parse_response_message(response),
            tool_calls=list(response.message.tool_calls or []),
            response_id=response.id,
            model=response.model,
            created=response.created,
            finish_reason=response.message.finish_reason,
            usage=vf.TurnUsage.from_usage(response.usage),
            tokens=vf.TurnTokens.from_response(
                response.message.tokens,
                is_truncated=bool(response.message.is_truncated),
            ),
            is_truncated=bool(response.message.is_truncated),
        )
        state.transcript.append(turn)
        state.is_truncated = state.is_truncated or turn.is_truncated
        state.stop("assistant_completed")


def load_harness(config: MyHarnessConfig) -> MyHarness:
    return MyHarness(config=config)
```

`load_harness(config: MyHarnessConfig)` defines the `[harness]` config schema.
Packages without `harness.py` use the base harness.

## State

`vf.State` is a strict Pydantic model. It is not a dict and has no convenience
helpers for arbitrary mutation.

Use:

- `state.transcript: list[vf.Turn]` for model request/completion turns.
- `state.prompt`, `state.completion`, and `state.messages` for derived views of
  the current transcript.
- `state.extras` for user-owned per-rollout mutable JSON.
- `state.metrics`, `state.reward`, and `Turn.tokens.*_advantages` for scoring.
- `state.artifacts` for serializable outputs worth saving.

Do not store live clients, functions, runtimes, file handles, or other
non-serializable objects on `State`, `Task`, tool specs, user specs, or config.

## Tasks

Task rows become typed `vf.Task` instances. Define a task subclass when a field
matters to the environment:

```python
class SearchTask(vf.Task):
    query: str
    answer: str
```

Task records must be serializable. Common fields:

| Field | Meaning |
| --- | --- |
| `row_id` | Explicit upstream row/example identifier. |
| `prompt` | Initial non-system prompt messages. May be empty when a user server starts the rollout. |
| `system_prompt` | Per-task taskset system prompt override. |
| `answer` | Reference answer or target data. |
| `image` | Per-task runtime image selection. |
| `resources` | Per-task runtime CPU, memory, GPU, and disk requests. Runtime config wins when explicitly set. |
| `max_turns` | Per-task base-loop turn limit. |
| `toolsets` | Per-task toolset visibility. |
| `tools` | Per-tool visibility. |
| `extras` fields | Put custom mutable rollout data in `state.extras`, not task rows. |

## Tools

Tasksets declare toolsets through `ToolsetConfig`. A toolset implementation is
a `vf.Toolset` subclass with `@vf.tool` methods.

```python
class SearchToolsetConfig(vf.ToolsetConfig):
    scope: vf.Scope = "rollout"


class SearchToolset(vf.Toolset):
    @vf.tool(
        args={"case": "state.metadata.case"},
        extends={"events": "state.extras.search_events"},
        sets={"last_result": "state.extras.last_search_result"},
    )
    def search(self, query: str, case: str) -> dict:
        ...


class SearchTasksetConfig(vf.TasksetConfig):
    toolsets: vf.ToolsetConfigs = {"search": SearchToolsetConfig()}
```

The mapping key is the tool prefix exposed to the model. A config file can
override taskset-defined keys directly, and can add a new key by setting
`source` to a `ToolsetConfig` class path.

Use `scope="rollout"` for per-rollout servers and `scope="env"` for servers
that live for the environment lifetime. Group-shared resources should use
`state.group_id` plus env-scope server state.

`@vf.tool(args=..., sets=..., extends=...)` is the data-flow contract. `args`
are removed from the model-visible tool schema and injected from serialized
task/state paths at call time. `sets` consume return keys and replace state
paths. `extends` consume return keys and append lists to list paths. Multiple
same-path extends in one tool-call batch are allowed; their relative order is
not a contract.

Unbound result keys remain visible to the model. A `content` key is used as the
tool message content when it is the only unbound key.

## Users

Users follow the same pattern. A user implementation exposes a hidden
`respond` tool. The harness calls it after each assistant turn, and also before
the first model request when `task.prompt` is empty.

```python
class DialogueUserConfig(vf.UserConfig):
    pass


class DialogueUser(vf.User):
    @vf.user(
        args={"transcript": "transcript"},
        sets={"turn_count": "state.extras.turn_count"},
    )
    def respond(self, transcript: list[dict]) -> dict:
        return {
            "messages": [{"role": "user", "content": "continue"}],
            "turn_count": len(transcript),
        }


class DialogueTasksetConfig(vf.TasksetConfig):
    user: vf.UserConfig | None = DialogueUserConfig()
```

## Runtime

The harness owns live runtimes. Runtime config is resolved from taskset,
harness, and environment config into one provider/runtime pair for each rollout.
State stores only serializable runtime metadata and artifacts.

Available runtime providers:

- `vf.SubprocessRuntimeConfig`
- `vf.DockerRuntimeConfig`
- `vf.PrimeRuntimeConfig`
- `vf.ModalRuntimeConfig`
- `vf.DaytonaRuntimeConfig`

Live runtimes expose:

- `start`
- `stop`
- `expose`
- `run`
- `read`
- `write`

Custom runtimes implement `vf.RuntimeProvider` and `vf.Runtime`.

## Scoring

Tasksets define task lifecycle, metrics, and rewards. Harnesses may define
lifecycle and metrics for reusable execution telemetry:

```python
class MyTaskset(vf.Taskset[MyTasksetConfig]):
    @vf.setup
    async def setup(self, task: MyTask, state: vf.State) -> None:
        state.extras["started"] = True

    @vf.metric
    async def length(self, state: vf.State) -> float:
        return float(len(str(state.completion[-1].content)))

    @vf.reward
    async def exact(self, task: MyTask, state: vf.State) -> float:
        return float(state.completion[-1].content == task.answer)
```

Group rewards run through `env.score_group(tasks, states)`. Env-level advantage
functions mutate token-level advantages in place. The default v1 advantage is
`"rl"`; pass `advantage=None` only when the caller wants trainer-owned
advantages.

```python
env = vf.Env(taskset=MyTaskset(), advantage="grpo")
```

## Evaluation

Run v1 environments the same way as v0 packages:

```bash
prime eval run reverse-text-v1 -n 5 -r 1
```

Typed overrides address component config directly:

```bash
prime eval run my-env-v1 --taskset.system-prompt "Be terse." --harness.max-turns 4
```

The package name selects the default taskset/harness. `--taskset.id` and
`--harness.id` may point at other installed v1 component packages when an eval
needs to compose them.

## Design Rules

- Import `verifiers.v1 as vf` in v1 code.
- Prefer one package-level taskset and, only when necessary, one harness.
- Keep tools and users in `servers/`.
- Keep config serializable and typed.
- Keep live Python functions only as methods on loaded owner objects.
- Mutate framework state only through typed `State` fields.
- Put user-owned mutable rollout data in `state.extras`.
- Do not pass runtimes, clients, functions, or other live objects across
  task/state/tool/user/config boundaries.
