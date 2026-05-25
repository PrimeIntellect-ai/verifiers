# Research Environments v1 Migration

This guide maps research-environments packages onto the v1 Taskset/Harness
pattern with direct v1 golden paths.

Use these references in this repository while porting:

- `environments/reverse_text/reverse_text_v1.py`: simple taskset + reward.
- `environments/math_python/math_python_v1.py`: sandbox-backed callable tool.
- `environments/wiki_search/wiki_search_v1.py`: private dependencies + callable
  tools.
- `environments/bfcl_v3/bfcl_v3.py`: task-local dynamic tool schemas.
- `environments/alphabet_sort/alphabet_sort_v1.py`: user function.
- `environments/mcp_search_env/mcp_search_env.py`: MCP tools.
- `tau2-bench-v1` (`environments/tau2_bench_v1/tau2_bench_v1.py`): task-owned user simulator.
- `environments/hello_subagent_v1/hello_subagent_v1.py`: nested harness calls.
- `environments/hello_parallel_sandbox_v1/hello_parallel_sandbox_v1.py`: shared
  sandbox-backed tools across child harnesses.
- `environments/opencode_harbor/opencode_harbor.py`: sandbox CLI harness.

## Quick Pattern Map

Use this table first. Pick the row that matches the research-environments
package, copy the v1 reference shape, then fill in the package-specific dataset
and scoring logic.

| research-environments package | v1 reference to copy | pattern |
| --- | --- | --- |
| `aime2024`, `aime2025`, `aime2026`, `gpqa`, `math500`, `mmlu_pro`, `simpleqa`, `simpleqa_verified` | `environments/reverse_text/reverse_text_v1.py` | serializable rows, base `Harness`, taskset reward |
| `clbench`, `color_codeword`, `graphwalks`, `ifbench`, `ifeval`, `if_summarize_judge`, `patterned_needle_in_haystack`, `science_env`, `unscramble`, `verbatim_copy` | `environments/reverse_text/reverse_text_v1.py` | single-turn prompt taskset with shared extractor or judge dependencies |
| `math_env` with Python execution | `environments/math_python/math_python_v1.py` | sandbox-backed callable Python tool |
| `browsecomp`, `ddbc`, `deepdive`, `hle` with tools, `wikispeedia` | `environments/wiki_search/wiki_search_v1.py` | callable `Toolset` with private dependencies and hidden bindings |
| `bfcl_v3` | `environments/bfcl_v3/bfcl_v3.py` | task-local dynamic tool schemas |
| `alphabet_sort` | `environments/alphabet_sort/alphabet_sort_v1.py` | taskset user simulator |
| `tau2-bench-v1` | `environments/tau2_bench_v1/tau2_bench_v1.py` | task-owned user simulator with task/state-dependent sessions |
| MCP-backed search/tool evals | `environments/mcp_search_env/mcp_search_env.py` | stdio MCP toolset |
| `mcp_atlas` | `Sandbox Service Toolsets` below | task-local service sandbox plus callable schema tools |
| helper-agent or self-judge envs | `environments/hello_subagent_v1/hello_subagent_v1.py` | direct `child_harness.run(child_task)` from a tool/update/reward |
| shared sandbox helper-agent envs | `environments/hello_parallel_sandbox_v1/hello_parallel_sandbox_v1.py` | borrowed sandbox/model state across child harnesses |
| Harbor/OpenCode task directories | `environments/opencode_harbor/opencode_harbor.py` | `HarborTaskset` plus `OpenCode` harness |
| Pi Coding Agent task directories | `Sandbox CLI Harnesses` below | `HarborTaskset` or custom taskset plus `Pi` harness |
| `terminal_bench_2`, `general_agent`, `nl2repobench`, RLM task-directory packages | `Task-Directory Command Harnesses` below | sandbox command program with task-owned uploads and artifacts |
| `scicode`, `livecodebench`, `code_env` | `Code Verification And Post-Rollout Checks` below | update runs verification, reward reads serializable result |
| mixed benchmark suites | `Mixed Environment Suites` below | one v1 `Env` per taskset/harness pair, exposed through explicit loaders |
| third-party agent libraries such as DSPy | `environments/dspy_flights/dspy_flights.py` | Python program using `state.get_endpoint_config(...)` or `state.get_client(...)` |

## General Migration Shape

Every migrated package should expose:

```python
import verifiers as vf


class MyTasksetConfig(vf.TasksetConfig):
    tasks: str = "load_tasks"
    rewards: list[str] = ["exact_answer"]
    metrics: list[str] = ["accuracy"]
    split: str = "train"
    system_prompt: str = SYSTEM_PROMPT


def load_tasks(split: str = "train") -> vf.Tasks:
    return build_rows(split=split)


def load_taskset(config: MyTasksetConfig) -> vf.Taskset:
    return vf.Taskset(config=config)


def load_environment(config: vf.EnvConfig) -> vf.Env:
    return vf.Env(taskset=vf.load_taskset(config=config.taskset))
```

Rows should be plain serializable task data:

```python
{
    "prompt": [{"role": "user", "content": question}],
    "answer": answer,
    "info": {"source": source},
    "max_turns": 8,
}
```

Environment-specific dependencies belong in the environment package's own
`pyproject.toml`. This is expected: v1 environments are packages precisely so
BFCL, DSPy, browser/search, CLI-agent, and benchmark-specific dependencies can
live with the environment instead of the root `verifiers` package.

Put system instructions in `system_prompt`, not in `prompt`:

```python
class PromptTasksetConfig(vf.TasksetConfig):
    tasks: str = "load_tasks"
    system_prompt: str = "Answer concisely."


def load_tasks() -> vf.Tasks:
    return [{"prompt": [{"role": "user", "content": "Question?"}]}]


taskset = vf.Taskset(config=PromptTasksetConfig())
```

or per task:

```python
{
    "system_prompt": "Use the provided tools.",
    "prompt": [{"role": "user", "content": question}],
}
```

## Single-Turn QA, Math, and Instruction Following

Use this for:

- `aime2024`, `aime2025`, `aime2026`
- `clbench`
- `color_codeword`
- `gpqa`
- `graphwalks`
- `if_summarize_judge`
- `ifbench`, `ifeval`
- `math500`, `math_env`
- `mmlu_pro`
- `patterned_needle_in_haystack`
- `science_env`
- `simpleqa`, `simpleqa_verified`
- `unscramble`
- `verbatim_copy`

Migration:

1. Convert the old dataset builder into `load_tasks()`.
2. Convert each reward or metric into `@vf.reward` / `@vf.metric`.
3. Reference task loaders and handlers from `TasksetConfig`.
4. Return `vf.Env(taskset=vf.load_taskset(...))`.

Example:

```python
import verifiers as vf


@vf.reward(weight=1.0)
async def exact(task, state) -> float:
    messages = vf.get_messages(state.get("completion") or [], role="assistant")
    response = str(messages[-1].content or "") if messages else ""
    return float(str(task["answer"]).strip() in response)


class QATasksetConfig(vf.TasksetConfig):
    tasks: str = "load_tasks"
    rewards: list[str] = ["exact"]
    split: str = "train"


def load_tasks(split: str = "train") -> vf.Tasks:
    return [
        {
            "prompt": [{"role": "user", "content": row["question"]}],
            "answer": row["answer"],
            "info": {"id": row["id"]},
            "max_turns": 1,
        }
        for row in load_dataset(..., split=split)
    ]


def load_taskset(config: QATasksetConfig) -> vf.Taskset:
    return vf.Taskset(config=config)


def load_environment(config: vf.EnvConfig):
    return vf.Env(taskset=vf.load_taskset(config=config.taskset))
```

Gotchas:

- Reference answers stay on `task`; do not expect `state["answer"]` to be the
  gold answer.
- Shared extraction or judging dependencies belong on `TasksetConfig.objects` and
  enter reward signatures through `bindings`:

```python
class AnswerExtractor:
    def __call__(self, completion: list[dict[str, object]]) -> str:
        ...


@vf.reward
async def exact(task, state, extract_answer) -> float:
    return float(extract_answer(state.get("completion") or []) == task["answer"])


class ExtractTasksetConfig(vf.TasksetConfig):
    tasks: str = "load_tasks"
    rewards: list[str] = ["exact"]
    objects: dict[str, str] = {
        "extract_answer": "build_answer_extractor",
    }
    bindings: dict[str, str] = {
        "exact.extract_answer": "objects.extract_answer",
    }


def load_tasks() -> vf.Tasks:
    return [{"prompt": [{"role": "user", "content": "Question?"}], "answer": "A"}]


taskset = vf.Taskset(config=ExtractTasksetConfig())
```

- Judge metrics are regular reward/metric functions. Instantiate judge clients
  inside a lazy factory or pass a client config through taskset config.

## Callable Tool Environments

Use this for:

- `browsecomp`
- `ddbc`
- `deepdive`
- `hle` with tools enabled
- `wikispeedia`

Migration:

1. Move tool functions into a `vf.Toolset`.
2. Put shared clients, caches, and indexes in `Toolset(objects=...)`.
3. Bind hidden tool args with `bindings`.
4. Give global objects a `close()` or `aclose()` method when they must be
   closed at teardown.

Example:

```python
import verifiers as vf


async def search(query: str, exa) -> str:
    return await exa.search(query)


async def open_page(url: str, exa) -> str:
    return await exa.open(url)


def load_toolset(config=None):
    def load_exa():
        return ExaClient(...)

    return vf.Toolset(
        tools=[search, open_page],
        objects={"exa": load_exa},
        bindings={
            "search.exa": "objects.exa",
            "open_page.exa": "objects.exa",
        },
        config=config,
    )


class SearchTasksetConfig(vf.TasksetConfig):
    tasks: str = "load_tasks"
    rewards: list[str] = ["judge_reward"]


def load_tasks() -> vf.Tasks:
    return [
        {
            "prompt": [{"role": "user", "content": "Search the web."}],
            "answer": "example",
        }
    ]


taskset = vf.Taskset(config=SearchTasksetConfig())
taskset.add_toolset(load_toolset())
env = vf.Env(taskset=taskset)
```

Gotchas:

- Tool functions should only expose model-visible arguments in their signature.
  Hidden args come from `bindings`.
- Use `Toolset(objects={...})` for private dependencies owned by callable
  tools. Values should be named factory functions when construction is
  deferred; required factory parameters must be supplied through Toolset
  bindings.
- Reward and metric functions should read serializable task/state data or call
  a bound tool; they should not reach into toolset dependencies directly.
- For state-mutating tools such as `wikispeedia.click_link`, bind `state`
  implicitly by naming it in the function signature; the runtime passes it.
- A tool that marks completion can call `state.stop("reason")`.

Finish-tool pattern:

```python
async def finish(answer: str, state) -> str:
    state["answer"] = answer
    state.stop("submitted")
    return "submitted"


toolset = vf.Toolset(tools=[finish])
```

## Dynamic Tool Schemas And Services

Use this for `bfcl_v3`, `mcp_atlas`, and evals where each task row carries
schemas or service metadata. Keep schemas on `task`, point
`task["toolsets"]` at an importable task-local factory, and score from the
assistant tool calls or service call records in state.

References:

- `environments/bfcl_v3/bfcl_v3.py`
- service-style task-local toolsets in `mcp_atlas`

Gotchas:

- Dynamic tools still execute; use a recorder when the eval only needs emitted
  calls.
- Toolset factory refs in task data must be import strings so task rows stay
  serializable.
- Service sandboxes stay private to the toolset unless the task or harness
  explicitly shares a compatible primary sandbox.

## Sandbox-Backed Tools

Use this for Python execution tools and code-analysis helpers. Define a normal
callable, place it in `vf.Toolset(..., sandbox={...})`, add `sandbox` to the
tool signature, and choose `scope="rollout"`, `"group"`, or `"global"` based on
lifetime.

Reference: `environments/math_python/math_python_v1.py`.

Gotchas:

- Use `scope="group"` when scoring needs to inspect the sandbox after rollout.
- Save only serializable sandbox refs, command records, or artifacts on state.
- Updates should call resolved tools through `state.get_tools()` when they need
  live sandbox or service access.

## User Simulators

Use this for `tau2-bench-v1` and tasksets where the environment returns a user
message when the model does not call a tool. Put the simulator on
`TasksetConfig.user` or `HarnessConfig.user`; keep per-rollout simulator state
in `state`; put static clients behind `User(objects=...)`.

Reference: `environments/tau2_bench_v1/tau2_bench_v1.py`.

Gotchas:

- The base harness calls `user` only when the model returns no tool calls.
- Returning `[]` means no user response is available; the base harness stops.
- User functions receive `transcript` through the default binding.

## MCP Toolsets

Use this when tools already exist behind stdio MCP servers. Wrap each server as
`vf.MCPTool(command=..., args=[...])`, put the MCP tools in a taskset or
harness toolset, and use `program.channels="mcp"` for sandbox command harnesses
that should consume resolved toolsets through MCP.

Reference: `environments/mcp_search_env/mcp_search_env.py`.

Gotchas:

- MCP server auth and secrets should be handled by the server command or env.
- Callable tools and MCP tools can coexist in toolsets.
- `program.channels` names the program-facing channel, not a concrete tool.

## Nested Harness Calls

Use this for helper subagents, judges, or planners launched inside a tool call.
Construct the child `vf.Harness` as a normal object, bind it into a toolset
object, and call `await child_harness.run(child_task)`.

Reference: `environments/hello_subagent_v1/hello_subagent_v1.py`.

Gotchas:

- Child harnesses do not automatically inherit parent model controls.
- Persist summaries or child state explicitly when the parent needs them.
- Child runtime handles are stripped before state is finalized.

## Command Harnesses

Use this for Harbor, OpenCode, mini-swe-agent, RLM, terminal-bench-style task
directories, and other sandboxed CLI programs. Prefer packaged harnesses when
the format already matches:

```python
env = vf.Env(
    vf.EnvConfig(
        taskset=vf.HarborTasksetConfig(),
        harness=vf.OpenCodeConfig(),
    ),
    taskset=vf.HarborTaskset,
    harness=vf.OpenCode,
)
```

For custom command programs, put task-directory metadata on `task`, use
callable `program.files` / `program.dirs` import refs for task-dependent
uploads, set per-task sandbox overrides under `task["sandbox"]`, and collect
logs or reports through `program.artifacts`.

References:

- `environments/opencode_harbor/opencode_harbor.py`
- `environments/rlm_swe_v1/rlm_swe_v1.py`

Gotchas:

- `HarborTaskset` owns Harbor task loading, task sandbox overrides, `/task`
  uploads, and test scoring.
- CLI harnesses own installation, endpoint wiring, config generation, and log
  artifacts.
- Harness-owned MCP registration belongs in `program.channels.mcp`; it runs
  after ordinary setup and before the command.
- Use `scope="group"` when scoring needs sandbox state after rollout.

## Mixed Suites And Post-Rollout Checks

For mixed suites, build one v1 `Env` per independently configurable
taskset/harness pair until a v1-native suite wrapper exists. Do not wrap v1
`Env` objects in the v0 `EnvGroup`; use one `Taskset` with a `category` task
field when categories share the same harness and lifecycle.

For generated-code verification, run the agent with the base loop, a Python
program, or a sandbox command; materialize verification results in `@vf.update`;
score with `@vf.reward` / `@vf.metric`; and use `@vf.cleanup` for final
serializable mutation or resource cleanup.

References:

- `scicode`, `livecodebench`, and `code_env` style verification environments
- mixed v0/v1 packages during staged migrations

## Task and State Gotchas

- `Task` is immutable after `freeze()`.
- `Task` must be JSON-serializable.
- `State` is mutable during rollout but must be serializable before return.
- `task["prompt"]` cannot contain system messages.
- `state["prompt"]` may include the final rendered prompt after trajectory
  synchronization; read task input from `task`, not from `state`.
- `max_turns` can be set per task:

```python
{"max_turns": 32}
```

- Use `Taskset.init_group` for group-consistent task randomization.
- Use `@vf.metric(stage="group")`, `@vf.reward(stage="group")`, and
  `@vf.advantage` for group-level signals.
