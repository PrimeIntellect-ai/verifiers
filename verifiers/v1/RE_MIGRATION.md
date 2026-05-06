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
- `environments/mcp_search_env/mcp_search_v1.py`: MCP tools.
- `tau2-bench-v1` (`environments/tau2_bench_v1/tau2_bench_v1.py`): task-owned user simulator.
- `environments/hello_subagent_v1/hello_subagent_v1.py`: nested harness calls.
- `environments/hello_parallel_sandbox_v1/hello_parallel_sandbox_v1.py`: shared
  sandbox-backed tools across child harnesses.
- `environments/opencode_harbor/opencode_harbor_v1.py`: sandbox CLI harness.

## Quick Pattern Map

Use this table first. Pick the row that matches the research-environments
package, copy the v1 reference shape, then fill in the package-specific dataset
and scoring logic.

| research-environments package | v1 reference to copy | pattern |
| --- | --- | --- |
| `aime2024`, `aime2025`, `aime2026`, `gpqa`, `math500`, `mmlu_pro`, `simpleqa`, `simpleqa_verified` | `environments/reverse_text/reverse_text_v1.py` | serializable rows, base `Harness`, taskset reward |
| `clbench`, `color_codeword`, `graphwalks`, `ifbench`, `ifeval`, `if_summarize_judge`, `patterned_needle_in_haystack`, `science_env`, `unscramble`, `verbatim_copy` | `environments/reverse_text/reverse_text_v1.py` | single-turn prompt taskset with parser or judge closed over by reward |
| `math_env` with Python execution | `environments/math_python/math_python_v1.py` | sandbox-backed callable Python tool |
| `browsecomp`, `ddbc`, `deepdive`, `hle` with tools, `wikispeedia` | `environments/wiki_search/wiki_search_v1.py` | callable `Toolset` with private dependencies and hidden bindings |
| `bfcl_v3` | `environments/bfcl_v3/bfcl_v3.py` | task-local dynamic tool schemas |
| `alphabet_sort` | `environments/alphabet_sort/alphabet_sort_v1.py` | taskset user simulator |
| `tau2-bench-v1` | `environments/tau2_bench_v1/tau2_bench_v1.py` | task-owned user simulator with task/state-dependent sessions |
| MCP-backed search/tool evals | `environments/mcp_search_env/mcp_search_v1.py` | stdio MCP toolset |
| `mcp_atlas` | `Sandbox Service Toolsets` below | task-local service sandbox plus callable schema tools |
| helper-agent or self-judge envs | `environments/hello_subagent_v1/hello_subagent_v1.py` | direct `child_harness.run(child_task)` from a tool/update/reward |
| shared sandbox helper-agent envs | `environments/hello_parallel_sandbox_v1/hello_parallel_sandbox_v1.py` | borrowed sandbox/model state across child harnesses |
| Harbor/OpenCode task directories | `environments/opencode_harbor/opencode_harbor_v1.py` | `HarborTaskset` plus `OpenCode` harness |
| Pi Coding Agent task directories | `Sandbox CLI Harnesses` below | `HarborTaskset` or custom taskset plus `Pi` harness |
| `terminal_bench_2`, `general_agent`, `nl2repobench`, RLM task-directory packages | `Task-Directory Command Harnesses` below | sandbox command program with task-owned uploads and artifacts |
| `scicode`, `livecodebench`, `code_env` | `Code Verification And Post-Rollout Checks` below | update runs verification, reward reads serializable result |
| mixed benchmark suites | `Mixed Environment Suites` below | one v1 `Env` per taskset/harness pair, grouped with `EnvGroup` |
| third-party agent libraries such as DSPy | `environments/dspy_flights/dspy_flights.py` | Python program using `state.get_endpoint_config(...)` or `state.get_client(...)` |

## General Migration Shape

Every migrated package should expose:

```python
import verifiers.v1 as vf


def load_taskset(config=None) -> vf.Taskset:
    return vf.Taskset(
        source=load_rows,
        system_prompt=SYSTEM_PROMPT,
        rewards=[reward_fn],
        metrics=[metric_fn],
        toolsets=[load_toolset()],
        config=config,
    )


def load_harness(config=None) -> vf.Harness:
    return vf.Harness(config=config)


def load_v1_environment(config=None) -> vf.Env:
    return vf.Env(
        taskset=load_taskset(getattr(config, "taskset", None)),
        harness=load_harness(getattr(config, "harness", None)),
    )
```

If the base harness is enough, omit `load_harness`:

```python
def load_v1_environment(config=None) -> vf.Env:
    return vf.Env(taskset=load_taskset(getattr(config, "taskset", None)))
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
vf.Taskset(source=load_rows, system_prompt="Answer concisely.")
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

1. Convert the old dataset builder into `source` / `eval_source`.
2. Convert each reward or metric into `@vf.reward` / `@vf.metric`.
3. Return `vf.Env(taskset=taskset)`.

Example:

```python
import verifiers.v1 as vf


def source():
    for row in load_dataset(...):
        yield {
            "prompt": [{"role": "user", "content": row["question"]}],
            "answer": row["answer"],
            "info": {"id": row["id"]},
            "max_turns": 1,
        }


@vf.reward(weight=1.0)
async def exact(task, state) -> float:
    return float(str(task["answer"]).strip() in completion_text(state))


def load_taskset(config=None):
    return vf.Taskset(source=source, rewards=[exact], config=config)


def load_v1_environment(config=None):
    return vf.Env(taskset=load_taskset(getattr(config, "taskset", None)))
```

Gotchas:

- Reference answers stay on `task`; do not expect `state["answer"]` to be the
  gold answer.
- If a parser is needed, keep it as a normal Python object closed over by the
  reward function.
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
import verifiers.v1 as vf


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


def load_taskset(config=None):
    return vf.Taskset(
        source=source,
        toolsets=[load_toolset()],
        rewards=[judge_reward],
        config=config,
    )
```

Gotchas:

- Tool functions should only expose model-visible arguments in their signature.
  Hidden args come from `bindings`.
- Use `Toolset(objects={...})` for private dependencies owned by callable
  tools. Values should be named zero-arg factory functions when construction is
  deferred.
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

## Dynamic Tool Schema Environments

Use this for:

- `bfcl_v3`
- evals where every task row carries its own function schemas

Migration:

1. Keep schemas on `task`.
2. Add a task-local toolset spec under `task["toolsets"]`.
3. Have that toolset factory build callable objects with `tool_def`.
4. Score from the assistant tool calls in `state["completion"]`.

Reference: `environments/bfcl_v3/bfcl_v3.py`.

Example:

```python
from verifiers.types import Tool

import verifiers.v1 as vf


class SchemaTool:
    def __init__(self, tool_def):
        self.name = tool_def.name
        self.__name__ = tool_def.name
        self.tool_def = tool_def

    async def __call__(self, state, **arguments):
        state.setdefault("tool_calls", []).append({self.name: arguments})
        return "recorded"


def task_toolset(task):
    return vf.Toolset(
        tools=[SchemaTool(Tool(**schema)) for schema in task["tool_schemas"]]
    )


def source():
    yield {
        "prompt": [{"role": "user", "content": "Call the right function."}],
        "tool_schemas": [...],
        "toolsets": {"dynamic": {"fn": "my_env:task_toolset"}},
        "max_turns": 1,
    }
```

Gotchas:

- Dynamic tools still execute. Use a no-op recorder when the eval only needs to
  inspect emitted tool calls.
- Toolset factory refs in task data must be import strings so tasks remain
  serializable.
- For multi-turn benchmarks with task-specific execution semantics, use a
  custom harness program rather than subclassing Env.

## Sandbox-Backed Callable Tools

Use this for:

- Python execution tools in `math_env`
- code-analysis helpers that need isolated filesystem/process state

Migration:

1. Define the tool as a normal callable.
2. Place it in a `vf.Toolset(..., sandbox={...})`.
3. Add `sandbox` to the tool signature when it needs the sandbox handle.
4. Choose `scope="rollout"`, `scope="group"`, or `scope="global"` based on
   lifetime.

Reference: `environments/math_python/math_python_v1.py`.

Example:

```python
async def python(code: str, sandbox) -> str:
    result = await sandbox.execute(f"python - <<'PY'\n{code}\nPY")
    return result.stdout


def load_python_toolset(config=None):
    return vf.Toolset(
        tools=[python],
        write=True,
        scope="group",
        sandbox={
            "image": "python:3.11-slim",
            "packages": ["numpy", "sympy"],
            "timeout_minutes": 60,
        },
        config=config,
    )
```

Gotchas:

- Use `scope="group"` when scoring may need to inspect the same sandbox after
  rollout.
- Use `scope="rollout"` for throwaway execution where scoring only reads state.
- Live sandbox handles stay inside the runtime. Save only serializable sandbox
  refs, command records, or artifacts that downstream scoring/display should see.

## User Simulators

Use this for:

- `tau2-bench-v1`
- tasksets where the environment returns a user message when the model does not
  call a tool

Migration:

1. Make the user simulator a callable returning messages.
2. Pass it as `Taskset(user=...)` or `Harness(user=...)`.
3. Keep task-specific simulator state in `state`.
4. Put static simulator clients behind `User(objects=...)`; use a callable
   binding when the hidden argument depends on task or state.

Reference: `environments/tau2_bench_v1/tau2_bench_v1.py`.

Example:

```python
async def user(task, state, session) -> list[dict[str, object]]:
    if state.get("done"):
        return []
    message = await session.next_message(task, state)
    return [{"role": "user", "content": message}]


def load_session():
    return SessionFactory(...)


taskset = vf.Taskset(
    source=source,
    user={
        "fn": user,
        "scope": "rollout",
        "objects": {"session": load_session},
        "bindings": {"session": "objects.session"},
    },
    rewards=[reward],
)
```

For task/state-dependent sessions, bind a callable source directly:

```python
def session_for_rollout(task, state):
    return Session(task["scenario"], state["trajectory_id"])


taskset = vf.Taskset(
    source=source,
    user=vf.User(user, bindings={"session": session_for_rollout}),
)
```

Gotchas:

- The base harness calls `user` only when the model returns no tool calls.
- Returning `[]` means no user response is available; the base harness stops.
- User functions receive `transcript` through the default binding.

## MCP Toolsets

Use this for:

- environments where tools are already exposed by stdio MCP servers
- local servers that can be launched from a command plus args

Migration:

1. Wrap each server as `vf.MCPTool(command=..., args=[...])`.
2. Put MCP tools in a taskset or harness toolset.
3. Use `program={"command": [...], "sandbox": True, "tools": "mcp"}` for
   sandbox command harnesses that should consume resolved toolsets through MCP.

Reference: `environments/mcp_search_env/mcp_search_v1.py`.

Example:

```python
def load_toolset(config=None):
    return vf.Toolset(
        tools=[
            vf.MCPTool(
                command="python",
                args=["-m", "my_package.mcp_server", "--task-root", TASK_ROOT],
            )
        ],
        config=config,
    )
```

Gotchas:

- MCP server auth and secrets should be handled by the server command or env.
- Use task fields and bindings when the server needs task-specific arguments.
- Callable tools and MCP tools can coexist in toolsets. Python programs receive
  callable handles; sandbox command programs can request an MCP server through
  `program.tools`.
- `program.tools` names the program-facing interface, not a concrete tool. Use
  `"callable"` or `"mcp"`; tools such as `bash` are regular Toolset entries.

## Nested Harness Calls

Use this for:

- helper subagents inside a tool call
- judge or planner harnesses launched from a parent harness

Migration:

1. Construct the child `vf.Harness` as a normal object.
2. Bind it into a toolset object.
3. Call `await child_harness.run(child_task)`.

Reference: `environments/hello_subagent_v1/hello_subagent_v1.py`.

Example:

```python
async def ask_child(question: str, harness, state) -> str:
    child_task = vf.Task(
        {"prompt": [{"role": "user", "content": question}]}
    ).freeze()
    child_state = await harness.run(child_task)
    state.setdefault("child_answers", []).append(child_state["answer"])
    return completion_text(child_state)


def load_child_harness():
    return vf.Harness()


toolset = vf.Toolset(
    tools=[ask_child],
    objects={"child_harness": load_child_harness},
    bindings={"ask_child.harness": "objects.child_harness"},
)
```

Gotchas:

- Child harnesses do not automatically inherit parent model controls. Construct
  the child harness with the client/model it should use.
- Child rollout state is returned to the caller. Persist summaries or full child
  state explicitly when the parent needs it.
- Child runtime handles are stripped before state is finalized.

## Sandbox CLI Harnesses

Use this for:

- OpenCode-style task directories
- Harbor-shaped tasksets
- mini-swe-agent task directories
- RLM-style command harnesses
- CLI programs that call an intercepted OpenAI-compatible endpoint

Migration:

1. Use `vf.HarborTaskset` for Harbor-format task directories.
2. Use `vf.OpenCode()`, `vf.Pi()`, `vf.MiniSWEAgent()`, or `vf.RLM()` for the
   command harness.
3. Put task-owned uploads and sandbox overrides on the taskset.
4. Keep scoring as reward/metric functions on the taskset.

The packaged implementations live under `verifiers.v1.packages` while the v1
API stabilizes, and are re-exported from `verifiers.v1` for normal use.

Reference: `environments/opencode_harbor/opencode_harbor_v1.py`.

Example:

```python
env = vf.Env(
    taskset=vf.HarborTaskset(tasks="/path/to/harbor/tasks"),
    harness=vf.OpenCode(),
)
```

Gotchas:

- `HarborTaskset` owns task loading, per-task sandbox overrides, `/task`
  uploads, and test scoring.
- `OpenCode` owns OpenCode installation, config generation, MCP tool proxy
  wiring, and log artifacts.
- `Pi` owns Pi installation, intercepted model config generation, optional MCP
  adapter setup, and log artifacts.
- `MiniSWEAgent` owns mini-swe-agent installation, config layering, endpoint
  env, and log/trajectory artifacts.
- `RLM` owns RLM installation, optional `/task/rlm-skills` upload, endpoint
  wiring, and trajectory filtering.
- `task.program` is the merge point for task-owned program files/env/setup.
- Harness-owned CLI tool registration belongs in `program.tools.mcp`; it runs
  after ordinary setup and before the command.
- Use group-scoped sandbox lifetime when scoring needs to inspect the sandbox.

## Task-Directory Command Harnesses

Use this for:

- `terminal_bench_2`
- `general_agent`
- `nl2repobench`
- `clbench_rlm`, `graphwalks_rlm`, `longbenchpro_rlm`, `longcot_rlm`
- `math_env_rlm`, `mrcr_v2_rlm`, `needle_in_haystack_rlm`, `oolong_rlm`
- `rlm_graphwalks`, `rlm_longcot`, `rlm_mrcr_v2`, `rlm_oolong`,
  `rlm_secrets`, `tau3_bench_rlm`
- RLM/OpenCode packages that stage a per-task workspace

Migration:

1. Put task-directory metadata on `task`, including instruction text and local
   package paths.
2. Use a sandboxed command program for the solver.
3. Use callable `program.files` / `program.dirs` values when uploads depend on
   the task row.
4. Use `task["sandbox"]` for per-task sandbox overrides such as image, workdir,
   resources, or timeout.
5. Put final logs, JSON reports, and DB snapshots in `program.artifacts`.

Example:

```python
def task_package(task, state):
    return Path(task["task_dir"])


def instruction(task, state):
    return Path(task["task_dir"], "instruction.md").read_text()


harness = vf.Harness(
    sandbox={"image": "python:3.11-slim", "scope": "group"},
    program={
        "sandbox": True,
        "command": ["bash", "-lc", "solver run /task/instruction.md"],
        "tools": "mcp",
        "files": {"/task/instruction.md": instruction},
        "dirs": {"/workspace/task": task_package},
        "setup": ["pip install -e /workspace/task"],
        "artifacts": {
            "report": {
                "path": "/workspace/task/report.json",
                "format": "json",
                "optional": True,
            }
        },
    },
)
```

Gotchas:

- `program.files` values become file contents. `program.dirs` values become
  uploaded directory roots.
- `program.artifacts.*.optional` must be a boolean. Missing optional artifacts
  are recorded as `None`.
- Use `scope="group"` when scoring needs the sandbox after rollout; v1 keeps
  the sandbox alive until group scoring and cleanup complete.

## Mixed Environment Suites

Use this for:

- packages that currently route across several task categories or harness
  variants
- suites that should preserve separate tasksets, scoring, or harness configs

Migration:

1. Build one v1 `Env` per independently configurable taskset/harness pair.
2. Combine those envs with `vf.EnvGroup`.
3. Keep category-specific rewards, tools, and harness settings inside each
   child env.

Example:

```python
def load_environment(config=None):
    return vf.EnvGroup(
        [
            load_math_environment(getattr(config, "math", None)),
            load_graph_environment(getattr(config, "graph", None)),
        ]
    )
```

Gotchas:

- Use `EnvGroup` when categories need different harnesses or scoring contracts.
- Use one `Taskset` with a `category` task field when categories share the same
  harness and lifecycle.
- v1 env capability flags still flow through the child `Env` objects.

## Code Verification And Post-Rollout Checks

Use this for:

- `scicode`
- `livecodebench`
- `code_env`
- environments that verify generated code after the agent loop

Migration:

1. Run the agent with the base loop, a Python program, or a sandbox command.
2. Put verification that prepares state for scoring in `@vf.update`.
3. Put reward/metric computation in `@vf.reward` / `@vf.metric`.
4. Put post-scoring resource cleanup in `@vf.cleanup`.

Example:

```python
import json


async def bash(command: str, sandbox) -> str:
    result = await sandbox.execute(command, timeout=120)
    return json.dumps(
        {
            "returncode": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    )


@vf.update
async def run_tests(task, state):
    tools = state.get_tools()
    state["tests"] = json.loads(await tools["bash"](command="pytest -q"))


@vf.reward
async def pass_rate(task, state) -> float:
    return 1.0 if state["tests"]["returncode"] == 0 else 0.0


toolset = vf.Toolset(
    tools=[bash],
    updates=[run_tests],
    sandbox="program",
    write=True,
)
```

Gotchas:

- Updates run before rewards and metrics. Use them for parsing, verification,
  and serializable state materialization.
- Updates should call resolved tools through `state.get_tools()` when they need live
  sandbox or service access.
- Cleanup runs after scoring. Use it for user-visible final mutation or
  resource cleanup that is not handled by sandbox scope.
- For verification that needs sandbox state, keep the owning sandbox at
  `scope="group"` or borrow the primary program sandbox in the child state.

## Sandbox Service Toolsets

Use this for:

- `mcp_atlas`
- task-local services that expose tool schemas and mutate private state

Migration:

1. Load or derive tool schemas during taskset construction and store them on
   `task`.
2. Build task-local callable tools from those schemas.
3. Put the service container behind the same task-local `Toolset` as a sandbox.
4. Bind `sandbox` into each callable that needs to call the service.

Example:

```python
class ServiceTool:
    def __init__(self, tool_def):
        self.name = tool_def.name
        self.tool_def = tool_def

    async def __call__(self, state, sandbox, **arguments):
        result = await sandbox.execute(
            atlas_curl_command(
                "/call-tool",
                {"tool_name": self.name, "tool_args": arguments},
            )
        )
        state.setdefault("service_calls", []).append({self.name: arguments})
        return json.loads(result.stdout)


def service_toolset(task):
    tools = [ServiceTool(vf.Tool(**schema)) for schema in task["tool_schemas"]]
    return vf.Toolset(
        tools=tools,
        sandbox={
            "image": task["service_image"],
            "start_command": task["service_start_command"],
            "scope": "rollout",
        },
        write=True,
    )
```

Gotchas:

- Schemas on `task` keep tool definitions serializable and available before the
  first model request.
- The sandbox remains private to the toolset unless the task or harness
  explicitly passes a compatible primary sandbox.

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
