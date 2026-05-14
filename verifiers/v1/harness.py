from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar, cast

from verifiers.decorators import metric, update
from verifiers.errors import Error, OverlongPromptError
from verifiers.types import (
    MessageContent,
    Messages,
    SamplingArgs,
    ToolMessage,
)
from verifiers.utils.async_utils import maybe_call_with_named_args
from verifiers.utils.error_utils import error_info
from verifiers.utils.message_utils import normalize_messages
from verifiers.utils.response_utils import parse_response_message
from verifiers.utils.tool_utils import is_valid_tool_content_parts

from .config import (
    HarnessConfig,
    SandboxConfig,
    import_config_ref,
    merge_config_handler_map,
    merge_config_value,
    resolve_config_object,
    sandbox_config_mapping,
)
from .utils.binding_utils import BindingMap, normalize_binding_map
from .utils.endpoint_utils import (
    Endpoint,
    assistant_completion_from_messages,
    run_intercepted_program,
)
from .utils.json_utils import json_args
from .utils.mcp_proxy_utils import (
    proxy_program,
    proxy_sandbox,
)
from .utils.program_utils import endpoint_api_key, program_channels, run_local_command
from .utils.program_utils import (
    merge_task_program,
    merge_task_sandbox,
    program_kind,
    validate_program_options,
    validate_program_sandbox_scope,
)
from .runtime import Runtime
from .utils.sandbox_utils import run_sandbox_command
from .utils.sandbox_program_utils import (
    python_program_sandbox,
    run_sandbox_python_program,
)
from .utils.prompt_utils import (
    normalize_prompt,
    normalize_system_prompt,
    resolve_system_prompt,
)
from .utils.timing_utils import ensure_timing, record_generation_timing
from .utils.tool_utils import tool_error_content
from .utils.trajectory_utils import has_borrowed_trajectory, sync_trajectory
from .state import State
from .task import Task
from .toolset import ToolsetCollection, merge_toolsets, normalize_toolset_collection
from .user import normalize_user
from .types import ConfigData, ConfigMap, Handler, ModelClient, ProgramMap, PromptInput

if TYPE_CHECKING:
    from .taskset import Taskset


class Harness:
    config_type: ClassVar[type[HarnessConfig]] = HarnessConfig

    def __init__(
        self,
        # Singleton fields.
        program: Handler | ProgramMap | None = None,
        system_prompt: PromptInput | None = None,
        user: Handler | str | ConfigMap | None = None,
        bindings: BindingMap | None = None,
        sandbox: ConfigMap | SandboxConfig | None = None,
        client: ModelClient | None = None,
        model: str | None = None,
        sampling_args: SamplingArgs | None = None,
        max_turns: int | None = None,
        # Collection fields.
        toolsets: ToolsetCollection | None = None,
        stops: list[Handler] | None = None,
        setups: list[Handler] | None = None,
        updates: list[Handler] | None = None,
        metrics: list[Handler] | None = None,
        rewards: list[Handler] | None = None,
        advantages: list[Handler] | None = None,
        cleanups: list[Handler] | None = None,
        # Config.
        config: HarnessConfig | None = None,
    ):
        self.config = type(self).config_type.from_config(config)
        if max_turns is not None:
            self.config.max_turns = max_turns
        program_value = resolve_config_object(
            merge_config_value(program, self.config.program)
        )
        self.program = cast(Handler | ProgramMap | None, program_value)
        system_prompt_value = cast(
            PromptInput | None,
            merge_config_value(system_prompt, self.config.system_prompt),
        )
        self.system_prompt = normalize_system_prompt(
            system_prompt_value, field_name="harness.system_prompt"
        )
        self.system_prompt_merge = self.config.system_prompt_merge
        self.user = normalize_user(merge_config_value(user, self.config.user))
        self.bindings = {
            **self.config.bindings,
            **normalize_binding_map(bindings, "Harness bindings", allow_objects=False),
        }
        self.sandbox = sandbox_config_mapping(
            merge_config_value(sandbox, self.config.sandbox)
        )
        self.client = cast(
            ModelClient | None,
            resolve_config_object(merge_config_value(client, self.config.client)),
        )
        self.model = cast(str | None, merge_config_value(model, self.config.model))
        self.sampling_args = cast(
            SamplingArgs,
            merge_config_value(sampling_args, self.config.sampling_args),
        )
        self.toolsets, self.named_toolsets = merge_toolsets(
            toolsets or (), self.config.toolsets
        )
        handlers = merge_config_handler_map(
            {
                "stop": stops or (),
                "setup": setups or (),
                "update": updates or (),
                "metric": [num_turns, *(metrics or [])],
                "reward": rewards or (),
                "advantage": advantages or (),
                "cleanup": cleanups or (),
            },
            self.config,
        )
        self.stops = handlers["stop"]
        self.setups = handlers["setup"]
        self.updates = handlers["update"]
        self.metrics = handlers["metric"]
        self.rewards = handlers["reward"]
        self.advantages = handlers["advantage"]
        self.cleanups = handlers["cleanup"]
        keep_step_value = resolve_config_object(self.config.keep_trajectory_step)
        if keep_step_value is not None and not callable(keep_step_value):
            raise TypeError("keep_trajectory_step must be callable.")
        self.keep_trajectory_step = cast(Handler | None, keep_step_value)
        self.taskset: "Taskset | None" = None
        self.runtime = self.resolve_runtime()
        self.endpoint = Endpoint(use_tunnel=self.program_uses_sandbox())
        self._program = self.compile_program(self.program)

    @classmethod
    def config_schema(cls) -> str:
        return cls.config_type.schema_text()

    def _add_handler(self, handlers: list[Handler], fn: Handler) -> None:
        handlers.append(fn)
        self.runtime = self.resolve_runtime()

    def add_metric(self, fn: Handler) -> None:
        self._add_handler(self.metrics, fn)

    def add_reward(self, fn: Handler) -> None:
        self._add_handler(self.rewards, fn)

    def add_advantage(self, fn: Handler) -> None:
        self._add_handler(self.advantages, fn)

    def add_toolset(self, toolset: object) -> None:
        toolsets, named_toolsets = normalize_toolset_collection(toolset)
        duplicate = set(self.named_toolsets) & set(named_toolsets)
        if duplicate:
            raise ValueError(f"Toolsets are defined twice: {sorted(duplicate)}.")
        self.toolsets.extend(toolsets)
        self.named_toolsets.update(named_toolsets)
        self.runtime = self.resolve_runtime()

    def add_stop(self, fn: Handler) -> None:
        self._add_handler(self.stops, fn)

    def add_setup(self, fn: Handler) -> None:
        self._add_handler(self.setups, fn)

    def add_update(self, fn: Handler) -> None:
        self._add_handler(self.updates, fn)

    def add_cleanup(self, fn: Handler) -> None:
        self._add_handler(self.cleanups, fn)

    def attach_taskset(self, taskset: "Taskset") -> None:
        self.taskset = taskset
        taskset.attach_harness(self)
        self.runtime = self.resolve_runtime()

    def resolve_runtime(self) -> Runtime:
        return Runtime(taskset=self.taskset, harness=self)

    async def run(self, task: Task | ConfigMap, state: State | None = None) -> State:
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
                state._set_stop_condition("program_completed")
                await self.runtime.collect_artifacts(task, state)
            except Error as e:
                self.record_error(state, e)
            await self.runtime.update_rollout(task, state)
            record_generation_timing(state)
            timing_recorded = True
            if state.get("runtime", {}).get("score_rollout", True):
                await self.runtime.score_rollout(task, state)
            state._set_completed(True)
            completed = True
        finally:
            if not timing_recorded:
                record_generation_timing(state)
            await self.runtime.cleanup_rollout(task, state)
            if not self.has_group_boundary(state):
                await self.runtime.cleanup_group([task], [state])
                state.strip_runtime_handles()
            if completed:
                state.assert_serializable()
        return state

    def record_error(self, state: State, error: Error) -> None:
        if isinstance(error, OverlongPromptError):
            state["prompt_too_long"] = True
            state._set_truncated(True)
            state._set_stop_condition("prompt_too_long", overwrite=True)
            return
        state._set_error(error_info(error))
        state._set_stop_condition("has_error", overwrite=True)

    async def score_group(self, tasks: list[Task], states: list[State]) -> list[State]:
        return await self.runtime.score_group(tasks, states)

    async def cleanup_group(self, tasks: list[Task], states: list[State]) -> None:
        await self.runtime.cleanup_group(tasks, states)
        for state in states:
            state.strip_runtime_handles()

    def has_group_boundary(self, state: State) -> bool:
        runtime = state.get("runtime", {})
        return isinstance(runtime, Mapping) and "group_key" in runtime

    async def teardown(self) -> None:
        await self.runtime.teardown()
        await self.endpoint.teardown()

    async def init_state(self, task: Task) -> State:
        return State.for_task(task)

    @update(priority=-100)
    async def render_completion(self, task: Task, state: State) -> None:
        _ = task
        if has_borrowed_trajectory(state):
            return
        sync_trajectory(state)

    async def setup_state(self, task: Task, state: State) -> State:
        explicit_runtime = dict(cast(ConfigMap, state.get("runtime") or {}))
        task_controls = {
            key: task[key] for key in ("max_turns", "tools") if key in task
        }
        state["runtime"] = {**task_controls, **explicit_runtime}
        if "tools" in task and not isinstance(task["tools"], Mapping):
            raise TypeError("task.tools must be a mapping with show or hide.")
        model_handle = self.runtime.resolved_handle(state, "model")
        if (
            model_handle is None
            and self.client is not None
            and "client_key" not in state["runtime"]
        ):
            self.runtime.bind_model_client(state, self.client)
        elif model_handle is not None:
            for key in ("model", "client_type", "sampling_args"):
                if key in model_handle:
                    state["runtime"].setdefault(key, model_handle[key])
        if self.model is not None:
            state["runtime"].setdefault("model", self.model)
        if self.sampling_args:
            sampling_args = dict(self.sampling_args)
            sampling_args.update(
                cast(ConfigMap, state["runtime"].get("sampling_args") or {})
            )
            state["runtime"]["sampling_args"] = sampling_args
        self.resolve_system_prompt(task, state)
        await self.runtime.ensure_rollout_toolsets(task, state)
        self.runtime.validate_bindings(state)
        await self.runtime.ensure_mcp_tools(state)
        self.runtime.resolve_trajectory(state)
        self.runtime.prepare_state(task, state)
        await self.runtime.ensure_global_sandboxes(state)
        self.runtime.bind_global_sandboxes(state)
        state.setdefault("artifacts", {})
        state.setdefault("metrics", {})
        state.setdefault("reward", 0.0)
        ensure_timing(state)
        return state

    def resolve_system_prompt(self, task: Task, state: State) -> None:
        taskset_system_prompt = getattr(self.taskset, "system_prompt", [])
        state["system_prompt"] = resolve_system_prompt(
            task=task,
            taskset_system_prompt=taskset_system_prompt,
            harness_system_prompt=self.system_prompt,
            merge=self.system_prompt_merge,
        )

    async def run_program(self, task: Task, state: State) -> State:
        endpoint = self.resolved_endpoint(state)
        result = await run_intercepted_program(
            self._program, endpoint, self.runtime, task, state
        )
        if result is None:
            return state
        if isinstance(result, State):
            return result
        if isinstance(result, Mapping):
            state.update(result)
            return state
        raise TypeError("Harness program must return None, State, or a mapping.")

    def resolved_endpoint(self, state: State) -> Endpoint:
        handle = self.runtime.resolved_handle(state, "endpoint")
        if handle is None:
            return self.endpoint
        runtime = self.runtime.handle_runtime(handle, "endpoint")
        harness = getattr(runtime, "harness", None)
        endpoint = getattr(harness, "endpoint", None)
        if not isinstance(endpoint, Endpoint):
            raise RuntimeError("Resolved endpoint handle has no live endpoint.")
        return endpoint

    def compile_program(self, program: Handler | ProgramMap | None) -> Handler:
        if program is None:
            return self.base_program
        if callable(program):
            return self.local_callable_program(program)
        if not isinstance(program, Mapping):
            raise TypeError("program must be None, callable, or a mapping.")
        kind = program_kind(program)
        if kind == "base":
            sandbox_config = self.program_sandbox_config(program)
            validate_program_options(program, kind, sandbox_config)
            if sandbox_config is not None:
                return self.sandbox_base_program(program, sandbox_config)
            return self.base_program
        if kind == "fn":
            sandbox_config = self.program_sandbox_config(program)
            validate_program_options(program, kind, sandbox_config)
            fn_ref = program["fn"]
            if not isinstance(fn_ref, str):
                raise TypeError("program.fn must be a string ref.")
            if sandbox_config is not None:
                return self.sandbox_fn_program(program, sandbox_config, fn_ref)
            fn = import_config_ref(fn_ref)
            if not callable(fn):
                raise TypeError("program.fn did not resolve to a callable.")
            return self.local_callable_program(cast(Handler, fn))
        if kind == "command":
            sandbox_config = self.program_sandbox_config(program)
            validate_program_options(program, kind, sandbox_config)
            return self.command_program(cast(ConfigMap, program))
        raise AssertionError(f"Unhandled program kind: {kind}")

    def local_callable_program(self, fn: Handler) -> Handler:
        async def run(task: Task, state: State) -> object:
            await self.runtime.setup_rollout(task, state)
            return await maybe_call_with_named_args(fn, task=task, state=state)

        return run

    async def base_program(self, task: Task, state: State) -> State:
        await self.runtime.setup_rollout(task, state)
        prompt = normalize_messages(
            cast(
                Messages,
                normalize_prompt(state.get("prompt", []), field_name="state.prompt"),
            ),
            field_name="state.prompt",
        )
        system_prompt = normalize_messages(
            state.get("system_prompt", []), field_name="state.system_prompt"
        )
        messages = [*system_prompt, *prompt]
        prompt_messages = [
            message.model_dump(exclude_none=True) for message in messages
        ]

        def sync_completion() -> list[ConfigData]:
            rendered_messages = [
                message.model_dump(exclude_none=True) for message in messages
            ]
            state["completion"] = assistant_completion_from_messages(
                prompt_messages, rendered_messages
            )
            return rendered_messages

        turn = 0
        max_turns = state.get_max_turns(self.config.max_turns)
        while max_turns <= 0 or turn < max_turns:
            if await self.runtime.is_completed(task, state):
                return state
            response = await self.runtime.submit_model_request(
                messages,
                task,
                state,
                tool_defs=self.runtime.tool_defs(state),
            )
            turn += 1
            messages.extend(await parse_response_message(response))
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
                state._set_stop_condition("no_tools")
                return state
            callable_tools = state.get_tools()
            for tool_call in tool_calls:
                content: MessageContent
                try:
                    name = tool_call.name
                    result = await maybe_call_with_named_args(
                        callable_tools[name], **json_args(tool_call.arguments)
                    )
                    content = (
                        cast(MessageContent, result)
                        if is_valid_tool_content_parts(result)
                        else str(result)
                    )
                except Exception as e:
                    content = tool_error_content(e)
                messages.append(ToolMessage(tool_call_id=tool_call.id, content=content))
                sync_completion()
                if await self.runtime.is_completed(task, state):
                    return state
            if max_turns > 0 and turn >= max_turns:
                state._set_stop_condition("max_turns_reached", overwrite=True)
                return state
        return state

    def command_program(self, program: ConfigMap) -> Handler:
        async def run(task: Task, state: State) -> State:
            runtime = self.runtime
            merged_program = merge_task_program(program, task, kind="command")
            sandbox_config = self.program_sandbox_config(program)
            if sandbox_config is not None:
                return await run_sandbox_command(
                    self.prepare_sandbox_program(merged_program, state),
                    self.prepare_sandbox_config(
                        merge_task_sandbox(sandbox_config, task), program
                    ),
                    task,
                    state,
                    runtime,
                )
            await runtime.setup_rollout(task, state)
            return await run_local_command(merged_program, task, state, runtime)

        return run

    def sandbox_base_program(
        self, program: ConfigMap, sandbox_config: ConfigMap
    ) -> Handler:
        async def run(task: Task, state: State) -> State:
            merged_program = merge_task_program(program, task, kind="base")
            return await run_sandbox_python_program(
                program=self.prepare_sandbox_program(merged_program, state),
                sandbox_config=self.prepare_sandbox_config(
                    merge_task_sandbox(sandbox_config, task), merged_program
                ),
                task=task,
                state=state,
                runtime=self.runtime,
                mode="base",
                fn_ref=None,
                max_turns=state.get_max_turns(self.config.max_turns),
            )

        return run

    def sandbox_fn_program(
        self,
        program: ConfigMap,
        sandbox_config: ConfigMap,
        fn_ref: str,
    ) -> Handler:
        async def run(task: Task, state: State) -> State:
            merged_program = merge_task_program(program, task, kind="fn")
            return await run_sandbox_python_program(
                program=self.prepare_sandbox_program(merged_program, state),
                sandbox_config=self.prepare_sandbox_config(
                    merge_task_sandbox(sandbox_config, task), merged_program
                ),
                task=task,
                state=state,
                runtime=self.runtime,
                mode="fn",
                fn_ref=fn_ref,
                max_turns=state.get_max_turns(self.config.max_turns),
            )

        return run

    def program_uses_sandbox(self) -> bool:
        if not isinstance(self.program, Mapping):
            return False
        return self.program_sandbox_config(cast(ConfigMap, self.program)) is not None

    def program_sandbox_config(self, program: ConfigMap) -> ConfigMap | None:
        sandbox = program.get("sandbox")
        if sandbox is None or sandbox is False:
            return None
        if sandbox is True:
            if self.sandbox is None:
                raise ValueError("program.sandbox=true requires Harness.sandbox.")
            if not isinstance(self.sandbox, Mapping):
                raise TypeError("Harness.sandbox must be a mapping.")
            sandbox_config = cast(ConfigMap, self.sandbox)
            validate_program_sandbox_scope(sandbox_config)
            return sandbox_config
        if not isinstance(sandbox, Mapping | SandboxConfig):
            raise TypeError("program.sandbox must be true, false, or a mapping.")
        sandbox_config = {}
        if self.sandbox is not None:
            sandbox_config.update(cast(ConfigMap, self.sandbox))
        sandbox_config.update(sandbox_config_mapping(sandbox) or {})
        validate_program_sandbox_scope(sandbox_config)
        return sandbox_config

    def prepare_sandbox_program(self, program: ConfigMap, state: State) -> ConfigMap:
        if "mcp" in program_channels(program):
            endpoint_root_url = state.get("endpoint_root_url")
            if not isinstance(endpoint_root_url, str):
                raise RuntimeError("MCP program tools require an active endpoint.")
            return proxy_program(
                program,
                tool_base_url=f"{endpoint_root_url.rstrip('/')}/vf/tools",
                tool_api_key=endpoint_api_key(self.runtime),
            )
        return program

    def prepare_sandbox_config(
        self, sandbox_config: ConfigMap, program: ConfigMap
    ) -> ConfigMap:
        config = dict(sandbox_config)
        if "mcp" in program_channels(program):
            config = proxy_sandbox(config)
        if program_kind(program) in {"base", "fn"}:
            config = python_program_sandbox(config)
        return config


@metric
async def num_turns(task: Task, state: State) -> float:
    _ = task
    trajectory = state.get("trajectory") or []
    if not isinstance(trajectory, list):
        raise TypeError("state.trajectory must be a list.")
    return float(len(trajectory))
