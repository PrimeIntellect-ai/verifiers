from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any, ClassVar, cast

from verifiers.errors import Error, OverlongPromptError
from verifiers.utils.error_utils import error_info
from verifiers.utils.message_utils import normalize_messages
from verifiers.clients import Client
from verifiers.types import ClientConfig, Messages, SamplingArgs, ToolMessage

from .config import (
    HarnessConfig,
    import_config_ref,
    merge_config_items,
    merge_config_value,
    resolve_config_object,
    string_mapping,
)
from .utils.endpoint_utils import (
    Endpoint,
    assistant_completion_from_messages,
    run_intercepted_program,
)
from .utils.json_utils import json_args
from .utils.mcp_proxy_utils import (
    proxy_program,
    proxy_sandbox,
    validate_tool_protocol,
)
from .utils.program_utils import run_local_command
from .runtime import Runtime
from .utils.sandbox_utils import run_sandbox_command
from .utils.sandbox_program_utils import run_sandbox_python_program
from .utils.trajectory_utils import sync_trajectory
from .state import State
from .task import Task
from .toolset import normalize_toolsets
from .user import normalize_user
from .utils.tool_utils import load_tools_from_state


PROGRAM_KIND_KEYS = {"base", "entrypoint", "command"}
PROGRAM_OPTION_KEYS = {"sandbox", "files", "dirs", "setup", "env", "artifacts"}
PROGRAM_KEYS = PROGRAM_KIND_KEYS | PROGRAM_OPTION_KEYS | {"args"}
SANDBOX_ONLY_PROGRAM_KEYS = {"files", "dirs", "setup", "artifacts"}


class Harness:
    config_type: ClassVar[type[HarnessConfig]] = HarnessConfig

    def __init__(
        self,
        program: Callable[..., object] | Mapping[str, object] | None = None,
        toolsets: list[object] | None = None,
        user: object | None = None,
        sandbox: Mapping[str, object] | None = None,
        tool_protocol: str | None = None,
        client: Client | ClientConfig | None = None,
        model: str | None = None,
        sampling_args: SamplingArgs | None = None,
        stop: list[Callable[..., object]] | None = None,
        metrics: list[Callable[..., object]] | None = None,
        rewards: list[Callable[..., object]] | None = None,
        advantages: list[Callable[..., object]] | None = None,
        cleanup: list[Callable[..., object]] | None = None,
        max_turns: int | None = None,
        keep_trajectory_step: Callable[..., object] | None = None,
        config: HarnessConfig | Mapping[str, object] | None = None,
    ):
        self.config = type(self).config_type.model_validate(config or {})
        if max_turns is not None:
            self.config.max_turns = max_turns
        program_value = resolve_config_object(
            merge_config_value(program, self.config.program)
        )
        self.program = cast(
            Callable[..., object] | Mapping[str, object] | None, program_value
        )
        self.toolsets = normalize_toolsets(
            merge_config_items(toolsets or (), self.config.toolsets)
        )
        self.user = normalize_user(merge_config_value(user, self.config.user))
        self.sandbox = merge_config_value(sandbox, self.config.sandbox)
        self.tool_protocol = validate_tool_protocol(
            merge_config_value(tool_protocol, self.config.tool_protocol)
        )
        self.client = cast(
            Client | ClientConfig | None,
            resolve_config_object(merge_config_value(client, self.config.client)),
        )
        self.model = cast(str | None, merge_config_value(model, self.config.model))
        self.sampling_args = cast(
            SamplingArgs,
            merge_config_value(sampling_args, self.config.sampling_args),
        )
        self.stop = cast(
            list[Callable[..., object]],
            merge_config_items(stop or (), self.config.stop),
        )
        self.metrics = cast(
            list[Callable[..., object]],
            merge_config_items(metrics or (), self.config.metrics),
        )
        self.rewards = cast(
            list[Callable[..., object]],
            merge_config_items(rewards or (), self.config.rewards),
        )
        self.advantages = cast(
            list[Callable[..., object]],
            merge_config_items(advantages or (), self.config.advantages),
        )
        self.cleanup = cast(
            list[Callable[..., object]],
            merge_config_items(cleanup or (), self.config.cleanup),
        )
        keep_step_value = resolve_config_object(
            merge_config_value(keep_trajectory_step, self.config.keep_trajectory_step)
        )
        if keep_step_value is not None and not callable(keep_step_value):
            raise TypeError("keep_trajectory_step must be callable.")
        self.keep_trajectory_step = cast(Callable[..., object] | None, keep_step_value)
        self.taskset: object | None = None
        self.runtime = self.resolve_runtime()
        self.endpoint = Endpoint(use_tunnel=self.program_uses_sandbox())
        self._program = self.compile_program(self.program)

    @classmethod
    def config_schema(cls) -> str:
        return cls.config_type.schema_text()

    def add_metric(self, fn: Callable[..., object]) -> None:
        self.metrics.append(fn)
        self.runtime = self.resolve_runtime()

    def add_reward(self, fn: Callable[..., object]) -> None:
        self.rewards.append(fn)
        self.runtime = self.resolve_runtime()

    def add_advantage(self, fn: Callable[..., object]) -> None:
        self.advantages.append(fn)
        self.runtime = self.resolve_runtime()

    def add_toolset(self, toolset: object) -> None:
        self.toolsets.extend(normalize_toolsets([toolset]))
        self.runtime = self.resolve_runtime()

    def add_stop(self, fn: Callable[..., object]) -> None:
        self.stop.append(fn)
        self.runtime = self.resolve_runtime()

    def add_cleanup(self, fn: Callable[..., object]) -> None:
        self.cleanup.append(fn)
        self.runtime = self.resolve_runtime()

    def attach_taskset(self, taskset: object) -> None:
        self.taskset = taskset
        self.runtime = self.resolve_runtime()

    def resolve_runtime(self) -> Runtime:
        return Runtime(taskset=self.taskset, harness=self)

    async def run(
        self, task: Task | Mapping[str, Any], state: State | None = None
    ) -> State:
        task = task if isinstance(task, Task) else Task(task).freeze()
        state = await self.init_state(task) if state is None else state
        timing_recorded = False
        completed = False
        try:
            try:
                state = await self.setup_state(task, state)
                if not await self.runtime.is_completed(task, state):
                    state = await self.run_program(task, state)
                    await self.runtime.is_completed(task, state)
                state["stop_condition"] = (
                    state.get("stop_condition") or "program_completed"
                )
                await self.runtime.collect_artifacts(task, state)
            except Error as e:
                self.record_error(state, e)
            sync_trajectory(state)
            self.record_generation_timing(state)
            timing_recorded = True
            if state.get("runtime", {}).get("score_rollout", True):
                await self.runtime.score_rollout(task, state)
            state["is_completed"] = True
            completed = True
        finally:
            if not timing_recorded:
                self.record_generation_timing(state)
            await self.runtime.cleanup_rollout(task, state)
            if not self.has_group_boundary(state):
                await self.runtime.cleanup_group([task], [state])
            if completed:
                state.assert_serializable()
        return state

    def record_error(self, state: State, error: Error) -> None:
        if isinstance(error, OverlongPromptError):
            state["prompt_too_long"] = True
            state["is_truncated"] = True
            state["stop_condition"] = "prompt_too_long"
            return
        state["error"] = error_info(error)
        state["stop_condition"] = "has_error"

    async def score_group(self, tasks: list[Task], states: list[State]) -> list[State]:
        return await self.runtime.score_group(tasks, states)

    async def cleanup_group(self, tasks: list[Task], states: list[State]) -> None:
        await self.runtime.cleanup_group(tasks, states)

    def has_group_boundary(self, state: State) -> bool:
        runtime = state.get("runtime", {})
        return isinstance(runtime, Mapping) and "group_key" in runtime

    async def teardown(self) -> None:
        await self.runtime.teardown()
        await self.endpoint.teardown()

    async def init_state(self, task: Task) -> State:
        return State.for_task(task)

    async def setup_state(self, task: Task, state: State) -> State:
        state.setdefault("runtime", {})
        state["runtime"] = {**task.get("runtime", {}), **state["runtime"]}
        if self.client is not None and "client_key" not in state["runtime"]:
            self.runtime.bind_model_client(state, self.client)
        if self.model is not None:
            state["runtime"].setdefault("model", self.model)
        if self.sampling_args:
            sampling_args = dict(self.sampling_args)
            sampling_args.update(
                cast(Mapping[str, object], state["runtime"].get("sampling_args") or {})
            )
            state["runtime"]["sampling_args"] = sampling_args
        await self.runtime.ensure_mcp_tools(state)
        self.runtime.prepare_state(task, state)
        await self.runtime.ensure_global_sandboxes()
        self.runtime.bind_global_sandboxes(state)
        state.setdefault("trajectory", [])
        state.setdefault("artifacts", {})
        state.setdefault("metrics", {})
        state.setdefault("reward", 0.0)
        state.setdefault(
            "timing",
            {
                "generation_ms": 0.0,
                "scoring_ms": 0.0,
                "total_ms": 0.0,
                "start_time": time.time(),
            },
        )
        return state

    def record_generation_timing(self, state: State) -> None:
        timing = state["timing"]
        elapsed_ms = (time.time() - timing["start_time"]) * 1000
        timing["generation_ms"] = elapsed_ms
        timing["total_ms"] = elapsed_ms

    async def run_program(self, task: Task, state: State) -> State:
        result = await run_intercepted_program(
            self._program, self.endpoint, self.runtime, task, state
        )
        if result is None:
            return state
        if isinstance(result, State):
            return result
        if isinstance(result, Mapping):
            state.update(result)
            return state
        raise TypeError("Harness program must return None, State, or a mapping.")

    def compile_program(
        self, program: Callable[..., object] | Mapping[str, object] | None
    ) -> Callable[..., object]:
        if program is None:
            return self.base_program
        if callable(program):
            return program
        if not isinstance(program, Mapping):
            raise TypeError("program must be None, callable, or a mapping.")
        kind = self.program_kind(program)
        if kind == "base":
            self.validate_program_options(program, kind)
            sandbox_config = self.program_sandbox_config(program)
            if sandbox_config is not None:
                return self.sandbox_base_program(program, sandbox_config)
            return self.base_program
        if kind == "entrypoint":
            self.validate_program_options(program, kind)
            entrypoint = program["entrypoint"]
            if not isinstance(entrypoint, str):
                raise TypeError("program.entrypoint must be a string ref.")
            sandbox_config = self.program_sandbox_config(program)
            if sandbox_config is not None:
                return self.sandbox_entrypoint_program(
                    program, sandbox_config, entrypoint
                )
            fn = import_config_ref(entrypoint)
            if not callable(fn):
                raise TypeError("program.entrypoint did not resolve to a callable.")
            return fn
        if kind == "command":
            self.validate_program_options(program, kind)
            return self.command_program(cast(Mapping[str, object], program))
        raise AssertionError(f"Unhandled program kind: {kind}")

    def program_kind(self, program: Mapping[str, object]) -> str:
        base = program.get("base", False)
        if not isinstance(base, bool):
            raise TypeError("program.base must be a boolean.")
        kinds = []
        if base:
            kinds.append("base")
        if "entrypoint" in program:
            kinds.append("entrypoint")
        if "command" in program:
            kinds.append("command")
        if not kinds and any(key in program for key in PROGRAM_OPTION_KEYS):
            if "sandbox" not in program or program.get("sandbox") is False:
                raise ValueError(
                    "option-only program mappings require sandbox placement."
                )
            kinds.append("base")
        if len(kinds) != 1:
            raise ValueError(
                "program mapping must specify exactly one of base=true, entrypoint, or command."
            )
        return kinds[0]

    def validate_program_options(
        self, program: Mapping[str, object], kind: str
    ) -> None:
        unknown = sorted(set(program) - PROGRAM_KEYS)
        if unknown:
            raise ValueError(f"Unknown program keys: {unknown}.")
        sandbox_config = self.program_sandbox_config(program)
        if sandbox_config is None:
            sandbox_only = sorted(set(program) & SANDBOX_ONLY_PROGRAM_KEYS)
            if sandbox_only:
                raise ValueError(
                    f"Program keys {sandbox_only} require sandbox placement."
                )
        if kind == "base" and sandbox_config is None:
            inert = sorted(set(program) & (PROGRAM_OPTION_KEYS - {"sandbox"}))
            if inert:
                raise ValueError(
                    f"Base program keys {inert} require sandbox placement."
                )

    async def base_program(self, task: Task, state: State) -> State:
        prompt = normalize_messages(state.get("prompt", []), field_name="state.prompt")
        prompt_messages = [message.model_dump(exclude_none=True) for message in prompt]
        messages = list(prompt)

        def sync_completion() -> list[dict[str, object]]:
            rendered_messages = [
                message.model_dump(exclude_none=True) for message in messages
            ]
            state["completion"] = assistant_completion_from_messages(
                prompt_messages, rendered_messages
            )
            return rendered_messages

        turn = 0
        while self.config.max_turns <= 0 or turn < self.config.max_turns:
            if await self.runtime.is_completed(task, state):
                return state
            response = await self.runtime.submit_model_request(
                messages,
                task,
                state,
                tool_defs=self.runtime.tool_defs(state),
            )
            turn += 1
            messages.append(response.message)
            rendered_messages = sync_completion()
            tool_calls = list(response.message.tool_calls or [])
            if not tool_calls:
                user_messages = await self.runtime.user_messages(
                    task, state, transcript=rendered_messages
                )
                if user_messages:
                    messages.extend(
                        normalize_messages(
                            cast(Messages, user_messages),
                            field_name="user_messages",
                        )
                    )
                    sync_completion()
                    continue
                state["stop_condition"] = state.get("stop_condition") or "no_tools"
                return state
            callable_tools = load_tools_from_state(state, runtime=self.runtime)
            for tool_call in tool_calls:
                name = tool_call.name
                result = await callable_tools[name](**json_args(tool_call.arguments))
                messages.append(
                    ToolMessage(tool_call_id=tool_call.id, content=str(result))
                )
                sync_completion()
                if await self.runtime.is_completed(task, state):
                    return state
            if self.config.max_turns > 0 and turn >= self.config.max_turns:
                state["stop_condition"] = "max_turns_reached"
                return state
        return state

    def command_program(self, program: Mapping[str, object]) -> Callable[..., object]:
        async def run(task: Task, state: State) -> State:
            runtime = self.runtime
            sandbox_config = self.program_sandbox_config(program)
            if sandbox_config is not None:
                return await run_sandbox_command(
                    self.prepare_sandbox_program(program),
                    self.prepare_sandbox_config(
                        self.task_merged_sandbox(sandbox_config, task)
                    ),
                    task,
                    state,
                    runtime,
                )
            return await run_local_command(program, task, state, runtime)

        return run

    def sandbox_base_program(
        self, program: Mapping[str, object], sandbox_config: Mapping[str, object]
    ) -> Callable[..., object]:
        async def run(task: Task, state: State) -> State:
            return await run_sandbox_python_program(
                program=self.prepare_sandbox_program(program),
                sandbox_config=self.prepare_sandbox_config(
                    self.task_merged_sandbox(sandbox_config, task)
                ),
                task=task,
                state=state,
                runtime=self.runtime,
                mode="base",
                entrypoint=None,
                max_turns=self.config.max_turns,
            )

        return run

    def sandbox_entrypoint_program(
        self,
        program: Mapping[str, object],
        sandbox_config: Mapping[str, object],
        entrypoint: str,
    ) -> Callable[..., object]:
        async def run(task: Task, state: State) -> State:
            return await run_sandbox_python_program(
                program=self.prepare_sandbox_program(program),
                sandbox_config=self.prepare_sandbox_config(
                    self.task_merged_sandbox(sandbox_config, task)
                ),
                task=task,
                state=state,
                runtime=self.runtime,
                mode="entrypoint",
                entrypoint=entrypoint,
                max_turns=self.config.max_turns,
            )

        return run

    def program_uses_sandbox(self) -> bool:
        if not isinstance(self.program, Mapping):
            return False
        return (
            self.program_sandbox_config(cast(Mapping[str, object], self.program))
            is not None
        )

    def program_sandbox_config(
        self, program: Mapping[str, object]
    ) -> Mapping[str, object] | None:
        sandbox = program.get("sandbox")
        if sandbox is None or sandbox is False:
            return None
        if sandbox is True:
            if self.sandbox is None:
                raise ValueError("program.sandbox=true requires Harness.sandbox.")
            if not isinstance(self.sandbox, Mapping):
                raise TypeError("Harness.sandbox must be a mapping.")
            sandbox_config = cast(Mapping[str, object], self.sandbox)
            self.validate_program_sandbox_scope(sandbox_config)
            return sandbox_config
        if not isinstance(sandbox, Mapping):
            raise TypeError("program.sandbox must be true, false, or a mapping.")
        sandbox_config = {}
        if self.sandbox is not None:
            if not isinstance(self.sandbox, Mapping):
                raise TypeError("Harness.sandbox must be a mapping.")
            sandbox_config.update(
                string_mapping(cast(Mapping[object, object], self.sandbox))
            )
        sandbox_config.update(string_mapping(cast(Mapping[object, object], sandbox)))
        self.validate_program_sandbox_scope(sandbox_config)
        return sandbox_config

    def prepare_sandbox_program(
        self, program: Mapping[str, object]
    ) -> Mapping[str, object]:
        if self.tool_protocol == "mcp":
            return proxy_program(program)
        return program

    def prepare_sandbox_config(
        self, sandbox_config: Mapping[str, object]
    ) -> Mapping[str, object]:
        if self.tool_protocol == "mcp":
            return proxy_sandbox(sandbox_config)
        return sandbox_config

    def validate_program_sandbox_scope(
        self, sandbox_config: Mapping[str, object]
    ) -> None:
        scope = str(sandbox_config.get("scope") or "rollout")
        if scope not in {"rollout", "group", "global"}:
            raise ValueError("program sandbox scope must be rollout, group, or global.")

    def task_merged_sandbox(
        self, sandbox_config: Mapping[str, object], task: Task
    ) -> Mapping[str, object]:
        config = dict(sandbox_config)
        task_sandbox = task.get("sandbox")
        if isinstance(task_sandbox, Mapping):
            config.update(string_mapping(cast(Mapping[object, object], task_sandbox)))
        self.validate_program_sandbox_scope(config)
        return config
