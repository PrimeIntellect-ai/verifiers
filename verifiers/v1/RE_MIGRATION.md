# Research Environments v1 Migration

This guide maps research-environments packages onto the v1 Taskset/Harness
pattern with direct v1 golden paths.

Use these references in this repository while porting:

- `environments/reverse_text/reverse_text_v1.py`: simple taskset + reward.
- `environments/math_python/math_python_v1.py`: sandbox-backed callable tool.
- `environments/wiki_search/wiki_search_v1.py`: lazy objects + callable tools.
- `environments/alphabet_sort/alphabet_sort_v1.py`: user function.
- `environments/mcp_search_env/mcp_search_v1.py`: MCP tools.
- `environments/tau2_bench/tau2_bench.py`: task-owned user simulator.
- `environments/hello_subagent_v1/hello_subagent_v1.py`: nested harness calls.
- `environments/opencode_harbor/opencode_harbor_v1.py`: sandbox CLI harness.

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
    return vf.Toolset(
        tools=[search, open_page],
        objects={"exa": lambda: ExaClient(...)},
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
- Use `Toolset(objects={...})` for heavy or stateful objects. Values may be lazy
  zero-arg callables.
- For state-mutating tools such as `wikispeedia.click_link`, bind `state`
  implicitly by naming it in the function signature; the runtime passes it.
- A tool that marks completion should contribute a stop condition through the
  same toolset.

Finish-tool pattern:

```python
async def finish(answer: str, state) -> str:
    state["answer"] = answer
    state["submitted"] = True
    return "submitted"


@vf.stop
async def submitted(task, state) -> bool:
    return bool(state.get("submitted"))


toolset = vf.Toolset(tools=[finish], stop=[submitted])
```

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
    result = await sandbox.run_python(code)
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
- Keep sandbox identifiers and handles out of final state; v1 strips runtime
  handles before returning.

## User Simulators

Use this for:

- `tau2_bench`
- tasksets where the environment returns a user message when the model does not
  call a tool

Migration:

1. Make the user simulator a callable returning messages.
2. Pass it as `Taskset(user=...)` or `Harness(user=...)`.
3. Keep task-specific simulator state in `state`.
4. Put simulator clients or sessions behind `User(objects=...)` or a callable
   factory.

Reference: `environments/tau2_bench/tau2_bench.py`.

Example:

```python
async def user(task, state, session) -> list[dict[str, object]]:
    if state.get("done"):
        return []
    message = await session.next_message(task, state)
    return [{"role": "user", "content": message}]


taskset = vf.Taskset(
    source=source,
    user={
        "fn": user,
        "scope": "rollout",
        "objects": {"session": lambda: SessionFactory(...)},
        "bindings": {"session": "objects.session"},
    },
    rewards=[reward],
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
3. Use `Harness(tool_protocol="mcp")` for command/program harnesses that should
   consume tools through MCP.

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
- Callable tools and MCP tools can coexist in toolsets; the harness chooses the
  protocol it consumes.

## Nested Harness Calls

Use this for:

- helper subagents inside a tool call
- judge or planner harnesses launched from a parent harness

Migration:

1. Construct the child `vf.Harness` as a normal object.
2. Bind it into a toolset object.
3. Call `await state.run_harness(child_harness, child_task)`.

Reference: `environments/hello_subagent_v1/hello_subagent_v1.py`.

Example:

```python
async def ask_child(question: str, harness, state) -> str:
    child_task = vf.Task(
        {"prompt": [{"role": "user", "content": question}]}
    ).freeze()
    child_state = await state.run_harness(harness, child_task)
    return completion_text(child_state)


toolset = vf.Toolset(
    tools=[ask_child],
    objects={"child_harness": lambda: vf.Harness()},
    bindings={"ask_child.harness": "objects.child_harness"},
)
```

Gotchas:

- Child rollouts inherit parent model controls unless the child harness supplies
  its own client/model.
- Child rollout state is recorded under `state["child_rollouts"]`.
- Child runtime handles are stripped before state is finalized.

## Sandbox CLI Harnesses

Use this for:

- OpenCode-style task directories
- Harbor-shaped tasksets
- CLI programs that call an intercepted OpenAI-compatible endpoint

Migration:

1. Build a taskset that yields task rows with prompt/instruction data and
   sandbox config.
2. Build a harness with `program={"command": [...], "sandbox": True, ...}`.
3. Use `program.files`, `program.dirs`, `program.setup`, and
   `program.artifacts` for uploads, setup, and output collection.
4. Keep scoring as reward/metric functions on the taskset.

Reference: `environments/opencode_harbor/opencode_harbor_v1.py`.

Example:

```python
harness = vf.Harness(
    program={
        "command": ["bash", "-lc", "opencode run < /task/instruction.md"],
        "sandbox": True,
        "files": {
            "/task/instruction.md": instruction_text,
            "/root/.config/opencode/opencode.json": opencode_config_json,
        },
        "artifacts": {
            "log": {"path": "/logs/agent/opencode.txt", "format": "text"},
        },
    },
    tool_protocol="mcp",
)
```

Gotchas:

- The command runs inside the program sandbox only when `program["sandbox"]` is
  true.
- The intercepted endpoint is injected into the sandbox program environment.
- Use `program.artifacts` for logs or files that need to become serializable
  state.
- Use group-scoped sandbox lifetime when scoring needs to inspect the sandbox.

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
