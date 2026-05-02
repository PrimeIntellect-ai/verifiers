# ruff: noqa
from __future__ import annotations

import asyncio
import uuid
from collections.abc import Mapping
from contextvars import ContextVar

from pydantic import BaseModel, ConfigDict

import verifiers as vf


CURRENT_RUNTIME = ContextVar("global_runtime")


class Config(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    source: object | None = None
    program: object | None = None
    sandbox: object | None = None
    toolsets: object | None = None
    metrics: object | None = None
    rewards: object | None = None
    cleanup: object | None = None


class Task(dict):
    def __init__(self, row=None):
        super().__init__(row or {})
        self._frozen = False

    def freeze(self):
        self._frozen = True
        return self

    def __setitem__(self, key, value):
        if self._frozen:
            raise TypeError("Task is immutable after freeze.")
        return super().__setitem__(key, value)

    def update(self, *args, **kwargs):
        if self._frozen:
            raise TypeError("Task is immutable after freeze.")
        return super().update(*args, **kwargs)

    def setdefault(self, key, default=None):
        if self._frozen:
            raise TypeError("Task is immutable after freeze.")
        return super().setdefault(key, default)


class State(dict): ...


def section(config, key):
    if config is None:
        return None
    return getattr(config, key, None)


def configure_value(value, config):
    # If value is None, config may materialize a serializable declaration
    # such as a command program, sandbox spec, or import ref.
    # If value is explicit, config may tune declared fields but must not
    # silently replace it with a different object.
    ...


def configure_named(items, config):
    # Items are addressed by function/object name.
    # Matching config entries tune metadata such as weight, skip, priority.
    # New config entries must be import refs or registry refs and are appended.
    # Duplicate final names hard fail.
    ...


def configure_toolsets(toolsets, config):
    # Toolsets are recursive containers, not name-addressed runtime objects.
    # Config may append toolset refs and tune visibility by flat tool names.
    # Final duplicate tool names hard fail after flattening.
    ...


class InterceptionServer:
    def __init__(self): ...

    async def register_rollout(self, state):
        return {"base_url": ..., "api_key": ...}

    async def next_request(self, state): ...

    async def deliver_response(self, request, response): ...


class SandboxManager:
    async def lease(self, task, state, config):
        # Returns a live sandbox handle with a serializable .ref.
        ...

    async def attach(self, ref): ...

    async def release(self, ref): ...

    async def teardown(self): ...


class ToolHandle:
    # Handle = live runtime object. Ref = serializable state pointer.
    def __init__(self, runtime, tool, bindings):
        self.runtime = runtime
        self.tool = tool
        self.bindings = bindings

    async def __call__(self, **args):
        bound_args = await self.runtime.resolve_bindings(self.bindings)
        return await self.tool(**args, **bound_args)

    def schema(self):
        # Public schema is function signature minus bound args.
        ...


class HarnessHandle:
    def __init__(self, runtime, harness, max_depth=None):
        self.runtime = runtime
        self.harness = harness
        self.max_depth = max_depth

    async def run(self, task, state=None):
        if state is None:
            state = await self.harness.init_state(task)
        state.setdefault("runtime", {})
        state["runtime"]["parent"] = self.runtime.state.get("id")
        state["runtime"]["depth"] = self.runtime.state["runtime"].get("depth", 0) + 1
        if self.max_depth is not None and state["runtime"]["depth"] > self.max_depth:
            raise RecursionError("maximum nested harness depth exceeded")
        return await self.harness.run(task, state)


class RuntimePool:
    def __init__(self, taskset=None, harness=None, config=None):
        self.taskset = taskset
        self.harness = harness
        self.config = Config.model_validate(config or {})
        self.interception = InterceptionServer()
        self.sandboxes = SandboxManager()
        self.globals = self.compile_globals(taskset, harness)
        self.toolsets = self.compile_toolsets(taskset, harness)
        self.sandbox_plan = self.compile_sandboxes(taskset, harness)
        self.rollout_signals, self.group_signals = self.compile_signals(
            taskset, harness
        )

    async def resolve(self, task, state, harness):
        return Runtime(self, task, state, harness)

    def compile_globals(self, taskset, harness):
        # Env-lifetime live objects come from taskset/harness/toolset configs.
        # A harness-only RuntimePool compiles the harness contributions alone.
        # Expensive loaders run here, not in task/state or inside Env.
        ...

    def compile_toolsets(self, taskset, harness):
        # Precompute recursive toolset graphs, callable signatures, bindings,
        # visibility rules, sandbox requirements, and transport adapters.
        ...

    def compile_sandboxes(self, taskset, harness):
        # Precompute harness and toolset sandbox requirements. Per-task state
        # can still select or refine the materialized sandbox plan.
        ...

    def compile_signals(self, taskset, harness):
        # Split metrics/rewards/cleanup by rollout vs group stage and priority.
        ...

    async def score_rollout(self, task, state, signals):
        # Signal public args are task/state. Bound args may be live runtime args.
        ...

    async def cleanup_rollout(self, task, state):
        # Rollout cleanup runs after rollout scoring or after an interrupted run.
        ...

    async def release_group(self, states):
        released = set()
        for state in states:
            sandbox = state.get("runtime", {}).get("sandbox")
            if sandbox and sandbox.get("scope") == "group":
                sandbox_id = sandbox["id"]
                if sandbox_id in released:
                    continue
                released.add(sandbox_id)
                await self.sandboxes.release(sandbox)

    async def score_group(self, tasks, states, signals):
        # Signal public args are tasks/states. Bound args may be live handles
        # rehydrated from state refs, e.g. sandboxes or judge harness handles.
        ...

    async def cleanup_group(self, tasks, states):
        # Group cleanup runs after group scoring and before group-scoped releases.
        ...

    async def teardown(self):
        await self.sandboxes.teardown()
        ...


class Runtime:
    def __init__(self, pool, task, state, harness):
        self.pool = pool
        self.task = task
        self.state = state
        self.harness = harness
        self.leases = {}

    async def __aenter__(self):
        self._token = CURRENT_RUNTIME.set(self)
        self.state.setdefault("runtime", {})
        self.state["runtime"][
            "endpoint"
        ] = await self.pool.interception.register_rollout(self.state)
        self.harness_handle = HarnessHandle(self, self.harness)
        await self.materialize_sandbox()
        await self.materialize_tools()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.release("rollout")
        CURRENT_RUNTIME.reset(self._token)

    async def lease(self, name, ref):
        self.leases[name] = ref
        return self.leases[name]

    async def release(self, scope):
        sandbox = self.state.get("runtime", {}).get("sandbox")
        if sandbox and sandbox.get("scope") == scope:
            await self.pool.sandboxes.release(sandbox)

    async def materialize_sandbox(self):
        sandbox_config = self.pool.sandbox_plan
        if sandbox_config is None:
            return
        sandbox = await self.pool.sandboxes.lease(self.task, self.state, sandbox_config)
        self.sandbox = sandbox
        self.state["runtime"]["sandbox"] = sandbox.ref
        await self.lease("sandbox", sandbox.ref)

    async def materialize_tools(self):
        available_toolsets = self.pool.toolsets
        # setup_state copies task-level runtime selections into state.
        selected_tools = self.state["runtime"].get("tools")
        # Public tool args are serializable transport args. Bound tool args may
        # be live runtime objects and never appear in public schemas or state.
        # flatten recursive toolsets, apply show/hide, infer/adapt transport,
        # allocate writable isolation, and satisfy bindings.
        internal_tools = ...
        exposed_tools = ...
        self.tools = {
            name: ToolHandle(self, tool, bindings=...)
            for name, tool in internal_tools.items()
        }
        self.state["tools"] = {
            name: self.tools[name].schema() for name in exposed_tools
        }
        ...

    async def collect_artifacts(self):
        # Program artifacts are collected into serializable state before scoring.
        # Artifact collectors may use live runtime handles, e.g. sandbox files.
        self.state.setdefault("artifacts", {})
        ...

    async def resolve_bindings(self, bindings):
        # tools.* resolves to live ToolHandle objects.
        # runtime.* resolves to live runtime handles such as sandbox/harness.
        # globals.* resolves to Env-lifetime objects declared by taskset/harness/toolsets.
        # task.* and state.* resolve to serializable task/state values.
        ...

    async def score_rollout(self, task, state):
        return await self.pool.score_rollout(
            task, state, signals=self.pool.rollout_signals
        )

    async def cleanup_rollout(self, task, state):
        return await self.pool.cleanup_rollout(task, state)


@vf.reward(weight=1.0)
async def exact_answer(task: Mapping[str, object], state: dict[str, object]) -> float:
    return float(state["answer"] == task["answer"])


@vf.metric
async def num_tool_calls(task: Mapping[str, object], state: dict[str, object]) -> float:
    return float(len(state.get("tool_calls", [])))


async def python_program(
    task: Mapping[str, object],
    state: dict[str, object],
) -> dict[str, object]: ...


async def bash(command, sandbox): ...


async def python(code, bash): ...


async def call_harness(prompt, harness):
    task = Task({"prompt": prompt})
    state = await harness.run(task)
    return state["answer"]


def search_wiki(query, db_id): ...


python_tools = vf.Toolset(
    tools=[bash, python],
    hide=["bash"],
    write=True,
    sandbox={
        "image": "python:3.11-slim",
        "packages": ["numpy", "sympy", "scipy"],
        # TODO: infer lifecycle from declared dataflow and handler bindings.
        "scope": "group",
    },
    bindings={
        "python.bash": "tools.bash",
        "bash.sandbox": "runtime.sandbox",
    },
)


wiki_tools = vf.Toolset(
    tools=[search_wiki],
    bindings={"search_wiki.db_id": "task.info.user.db_id"},
)


subharness_tools = vf.Toolset(
    tools=[call_harness],
    bindings={"call_harness.harness": "runtime.harness"},
)


async def standalone_harness_example():
    task = Task({"question": "what is 2+2?"}).freeze()
    harness = vf.Harness()
    return await harness.run(task)


def load_taskset(config=None) -> vf.Taskset:
    return vf.Taskset(
        source=lambda: ...,
        rewards=[exact_answer],
        config=config,
    )


def load_harness(config=None) -> vf.Harness:
    return vf.Harness(
        program={"entrypoint": "pkg.module:run"},
        metrics=[num_tool_calls],
        config=config,
    )


def load_harness(config=None) -> vf.Harness:
    return vf.Harness(
        program={"entrypoint": "pkg.module:run"},
        metrics=[num_tool_calls],
        config=config,
    )


def load_harness(config=None) -> vf.Harness:
    return vf.Harness(
        program={"entrypoint": "pkg.module:run"},
        sandbox={...},
        toolsets=[wiki_tools],
        metrics=[num_tool_calls],
        config=config,
    )


def load_harness(config=None) -> vf.Harness:
    return vf.Harness(
        program={
            "command": ["opencode", "run"],
            "args": ["--task", {"state": "paths.task"}],
        },
        sandbox={...},
        metrics=[num_tool_calls],
        config=config,
    )


async def library_program(task, state): ...


def load_harness(config=None) -> vf.Harness:
    return vf.Harness(
        program={"entrypoint": "pkg.library:run"},
        metrics=[num_tool_calls],
        config=config,
    )


def load_harness(config=None) -> vf.Harness:
    return vf.Harness(
        program=None,
        toolsets=[wiki_tools],
        metrics=[num_tool_calls],
        config=config,
    )


def load_environment(config=None) -> vf.Environment:
    return vf.Env(
        taskset=load_taskset(config.taskset),
        harness=load_harness(config.harness),
    )


class Taskset:
    def __init__(
        self,
        source=None,
        taskset_id=None,
        metrics=None,
        rewards=None,
        toolsets=None,
        cleanup=None,
        config=None,
    ):
        self.config = Config.model_validate(config or {})
        self.source = configure_value(source, section(self.config, "source"))
        self.taskset_id = taskset_id or type(self).__name__
        self.metrics = configure_named(metrics or [], section(self.config, "metrics"))
        self.rewards = configure_named(rewards or [], section(self.config, "rewards"))
        self.toolsets = configure_toolsets(
            toolsets or [], section(self.config, "toolsets")
        )
        self.cleanup = configure_named(cleanup or [], section(self.config, "cleanup"))

    def task(self, row) -> Task:
        # Stamp routing identity before the task is frozen.
        # Task ids may come from a stable row key, generated UUID, or index.
        task = Task(row)
        task["taskset_id"] = self.taskset_id
        task["task_id"] = str(task.get("task_id") or task.get("id") or uuid.uuid4().hex)
        return task.freeze()

    async def init_group(self, task, num_rollouts) -> tuple[list[Task], list[State]]:
        tasks = [task for _ in range(num_rollouts)]
        states = [State({"runtime": dict(task.get("runtime", {}))}) for task in tasks]
        return tasks, states

    async def score_group(self, tasks, states): ...


class Harness:
    def __init__(
        self,
        program=None,
        toolsets=None,
        sandbox=None,
        metrics=None,
        rewards=None,
        cleanup=None,
        config=None,
    ):
        self.config = Config.model_validate(config or {})
        self.program = configure_value(program, section(self.config, "program"))
        self.toolsets = configure_toolsets(
            toolsets or [], section(self.config, "toolsets")
        )
        self.sandbox = configure_value(sandbox, section(self.config, "sandbox"))
        self.metrics = configure_named(metrics or [], section(self.config, "metrics"))
        self.rewards = configure_named(rewards or [], section(self.config, "rewards"))
        self.cleanup = configure_named(cleanup or [], section(self.config, "cleanup"))
        self.taskset = None
        self.runtime = Runtime(taskset=None, harness=self)
        self._run_program = self.compile_program(self.program)

    def attach_taskset(self, taskset):
        self.taskset = taskset
        self.runtime = Runtime(taskset=taskset, harness=self)

    async def run(self, task, state=None):
        if state is None:
            state = await self.init_state(task)
        state = await self.setup_state(task, state)
        runtime = self.runtime
        async with runtime:
            try:
                state = await self.run_program(task, state)
                await runtime.collect_artifacts()
                state = await runtime.score_rollout(task, state)
            finally:
                await runtime.cleanup_rollout(task, state)
        return state

    async def init_state(self, task):
        return State(
            {
                # Task is read-only. Runtime selections are copied into state
                # before the harness runs so rollout-local changes stay in state.
                "runtime": dict(task.get("runtime", {})),
                "trajectory": [],
            }
        )

    async def setup_state(self, task, state):
        state.setdefault("runtime", {})
        state["runtime"] = {**task.get("runtime", {}), **state["runtime"]}
        state.setdefault("trajectory", [])
        return state

    async def run_program(self, task, state):
        return await self._run_program(
            task, state, tools=self.runtime.tool_calls(task, state)
        )

    def compile_program(self, program):
        if program is None:
            return self.base_program
        if "entrypoint" in program:
            return self.entrypoint_program(program)
        if "command" in program:
            return self.command_program(program)
        raise TypeError(program)

    async def base_program(self, task, state):
        runtime = CURRENT_RUNTIME.get()
        request = await runtime.pool.interception.next_request(state)
        response = await self.submit_model_request(request, task, state)
        await runtime.pool.interception.deliver_response(request, response)
        return state

    def entrypoint_program(self, program):
        async def run(task, state): ...

        return run

    def command_program(self, program):
        async def run(task, state): ...

        return run

    async def submit_model_request(self, request, task, state): ...


class Env(vf.Environment):
    def __init__(self, taskset, harness, config=None):
        self.taskset = taskset
        self.harness = harness
        self.config = config
        self.harness.attach_taskset(taskset)

    async def run_group(self, task, num_rollouts, controls=None):
        tasks, states = await self.taskset.init_group(task, num_rollouts)
        states = self.apply_controls(states, controls)
        states = await self.dispatch_rollouts(tasks, states, controls)
        states = await self.score_group(tasks, states)
        return states

    def apply_controls(self, states, controls=None):
        # Trainer-provided per-group runtime selections become serializable
        # state before rollout dispatch. Live clients are materialized by runtime.
        if controls is None:
            return states
        for state in states:
            state.setdefault("runtime", {})
            state["runtime"].update(controls)
        return states

    async def dispatch_rollouts(self, tasks, states, controls=None):
        return await asyncio.gather(
            *[self.rollout(task, state) for task, state in zip(tasks, states)]
        )

    async def rollout(self, task, state):
        state = await self.harness.run(task, state)
        return state

    async def score_group(self, tasks, states):
        states = await self.harness.score_group(tasks, states)
        await self.harness.cleanup_group(tasks, states)
        return states
