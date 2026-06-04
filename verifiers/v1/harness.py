import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Generic, TypeAlias, TypeVar, cast, final

from pydantic import AliasChoices, Field

import verifiers as vf
from verifiers.clients.client import Client
from verifiers.errors import Error, OverlongPromptError
from verifiers.types import (
    ClientConfig,
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
    ConfigSource,
    LifecycleConfig,
    import_config_ref,
)
from .artifact import ArtifactsConfig
from .program import (
    ProgramConfig,
)
from .model import ModelConfig, model_config_from_task
from .sandbox import SandboxConfig
from .user import UserConfig
from .utils.binding_utils import (
    BindingSources,
    BindingsConfig,
    ObjectsConfig,
)
from .utils.endpoint_utils import (
    Endpoint,
    assistant_completion_from_messages,
    run_intercepted_program,
)
from .utils.config_utils import (
    coerce_config,
    config_ref_context,
    config_type_from_class,
    qualified_config_ref,
    registered_config_type,
    register_config_type,
)
from .utils.runtime_owner_utils import RuntimeOwnerMixin
from .utils.json_utils import json_args
from .utils.mcp_proxy_utils import (
    proxy_program,
    proxy_sandbox,
)
from .utils.program_utils import (
    merge_task_program,
    merge_task_sandbox,
    program_channels,
    program_kind,
    run_local_command,
    validate_program_options,
    validate_program_sandbox_scope,
)
from .runtime import Runtime
from .utils.sandbox_utils import run_sandbox_command
from .utils.sandbox_program_utils import (
    python_program_sandbox,
    run_sandbox_python_program,
)
from .utils.logging_utils import log_rollout_finish, log_rollout_start
from .utils.prompt_utils import (
    SystemPrompt,
    SystemPromptStrategy,
    normalize_prompt,
    normalize_system_prompt,
    resolve_system_prompt,
)
from .utils.tool_utils import tool_error_content
from .utils.trajectory_utils import has_borrowed_trajectory, sync_trajectory
from .state import State
from .task import Task
from .types import (
    Handler,
    ModelClient,
    ConfigData,
    JsonData,
    Objects,
)

if TYPE_CHECKING:
    from .taskset import Taskset

ProgramResult: TypeAlias = State | ConfigData | None
ProgramRunner: TypeAlias = Callable[[Task, State], Awaitable[ProgramResult]]


class HarnessConfig(LifecycleConfig):
    # Core fields configure harness-owned runtime behavior.
    harness_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("harness_id", "id"),
    )
    program: ProgramConfig = ProgramConfig()
    model: ModelConfig = ModelConfig()
    system_prompt: SystemPrompt = None
    system_prompt_strategy: SystemPromptStrategy = "HT"
    sandbox: SandboxConfig | None = None
    user: UserConfig | None = None
    bindings: BindingsConfig = BindingsConfig()
    objects: ObjectsConfig = ObjectsConfig()
    artifacts: ArtifactsConfig = ArtifactsConfig()
    max_turns: int = -1

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: object) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        field = cls.model_fields.get("harness_id")
        if field is not None:
            field.validation_alias = AliasChoices("harness_id", "id")
            cls.model_rebuild(force=True)


ConfigT = TypeVar("ConfigT", bound=HarnessConfig)


class Harness(RuntimeOwnerMixin[ConfigT], Generic[ConfigT]):
    config: ConfigT
    program_config: ProgramConfig
    program: ProgramRunner

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        config_type = config_type_from_class(
            cls,
            inherited=False,
            owner_base=Harness,
            config_base=HarnessConfig,
        )
        if config_type is not None:
            register_config_type(cls, config_type)

    @final
    def __init__(
        self,
        config: ConfigSource = None,
        *,
        model: str | ModelConfig | None = None,
        client: ModelClient | str | None = None,
        sampling_args: SamplingArgs | None = None,
    ):
        model_kwargs_present = (
            model is not None or client is not None or sampling_args is not None
        )
        if config is not None and model_kwargs_present:
            raise TypeError("Pass either config or model/client/sampling_args kwargs.")
        runtime_model_client: Client | None = None
        config_type = registered_config_type(type(self), HarnessConfig)
        if config is None and model_kwargs_present:
            if isinstance(model, ModelConfig):
                if client is not None or sampling_args is not None:
                    raise TypeError(
                        "ModelConfig cannot be combined with client or sampling_args."
                    )
                config = config_type(model=model)
            else:
                model_client: ClientConfig | str | None = None
                if client is not None:
                    if isinstance(client, Client):
                        runtime_model_client = client
                    else:
                        model_client = client
                config = config_type(
                    model=ModelConfig(
                        name=model,
                        client=model_client,
                        sampling_args=sampling_args or {},
                    )
                )
        self.config = cast(ConfigT, coerce_config(config_type, config))
        with config_ref_context(self.config):
            self.initialize_runtime_refresh()
            resolved_harness_id = self.config.harness_id
            if resolved_harness_id is not None and not isinstance(
                resolved_harness_id, str
            ):
                raise TypeError("harness_id must be a string.")
            self.harness_id = resolved_harness_id or type(self).__name__
            self.program_config = self.config.program.resolve()
            system_prompt_value = self.load_system_prompt(self.config)
            self.system_prompt = normalize_system_prompt(
                system_prompt_value, field_name="harness.system_prompt"
            )
            self.system_prompt_strategy = self.config.system_prompt_strategy
            self.initialize_runtime_user(self.config.user)
            self.bindings: BindingSources = self.config.bindings.entries(
                "harness.bindings"
            )
            self.objects: Objects = self.load_objects(self.config.objects)
            self.artifacts = self.load_artifacts(self.config.artifacts)
            self.sandbox = self.load_sandbox(self.config.sandbox)
            self.model = self.load_model(self.config.model)
            self.model_client: ModelClient | None = cast(
                ModelClient | None,
                self.model.client_object(),
            )
            if runtime_model_client is not None:
                self.model_client = runtime_model_client
            self.initialize_runtime_toolsets(self.config, self.config.toolsets)
            self.initialize_runtime_handlers()
            self.taskset: "Taskset | None" = None
            self.runtime = Runtime(taskset=self.taskset, harness=self)
            self.runtime_refresh = self.rebuild_runtime
            self.endpoint = self.load_endpoint()
            self.program = self.compile_program(self.program_config)

    def load_system_prompt(self, config: ConfigT) -> SystemPrompt:
        return config.system_prompt

    def load_sandbox(self, config: SandboxConfig | None) -> SandboxConfig | None:
        sandbox = self.program_config.sandbox
        if sandbox is None or sandbox is False:
            return config
        base = config.data() if config is not None else {}
        if sandbox is True:
            return SandboxConfig.model_validate(base)
        return SandboxConfig.model_validate({**base, **sandbox.data()})

    def load_model(self, config: ModelConfig) -> ModelConfig:
        return config

    def load_endpoint(self) -> Endpoint:
        return Endpoint(
            use_tunnel=self.program_sandbox_config(self.program_config) is not None
        )

    def rebuild_runtime(self) -> None:
        self.runtime = Runtime(taskset=self.taskset, harness=self)

    async def run(self, task: Task, state: State | None = None) -> State:
        state = await self.init_state(task) if state is None else state
        log_rollout_start(state)
        timing_recorded = False
        completed = False
        try:
            try:
                timeout = cast(
                    float | None,
                    state.runtime_state().get("rollout_timeout_seconds"),
                )
                async with asyncio.timeout(timeout):
                    state = await self.setup_state(task, state)
                    if not await self.runtime.is_completed(task, state):
                        state = await self.run_program(task, state)
                        await self.runtime.is_completed(task, state)
                    state._set_stop_condition("program_completed")
                    await self.runtime.collect_artifacts(task, state)
                    await self.runtime.update_rollout(task, state)
                    state.record_generation_timing()
                    timing_recorded = True
                    if state.runtime_state().get("score_rollout", True):
                        await self.runtime.score_rollout(task, state)
                    state._set_completed(True)
                    completed = True
            except TimeoutError:
                state["timed_out"] = True
                state.stop("timeout_reached")
            except Error as e:
                self.record_error(state, e)
        finally:
            if not timing_recorded:
                state.record_generation_timing()
            await self.runtime.cleanup_rollout(task, state)
            if "group_key" not in state.runtime_state():
                await self.runtime.cleanup_group([task], [state])
                if completed:
                    state.finalize()
                else:
                    state.strip_runtime_handles()
            elif completed:
                state.assert_serializable()
            log_rollout_finish(state)
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

    async def teardown(self) -> None:
        await self.runtime.teardown()
        await self.endpoint.teardown()

    async def init_state(self, task: Task) -> State:
        return State.for_task(task)

    @vf.update(priority=-100)
    async def render_completion(self, state: State) -> None:
        if has_borrowed_trajectory(state):
            return
        sync_trajectory(state)

    @vf.metric
    async def num_turns(self, state: State) -> float:
        trajectory = state.get("trajectory") or []
        if not isinstance(trajectory, list):
            raise TypeError("state.trajectory must be a list.")
        return float(len(trajectory))

    async def setup_state(self, task: Task, state: State) -> State:
        await self.setup_runtime_state(task, state)
        await self.setup_model_state(task, state)
        await self.resolve_system_prompt(task, state)
        await self.setup_tool_state(task, state)
        await self.setup_sandbox_state(state)
        await self.setup_default_state_fields(state)
        return state

    async def setup_runtime_state(self, task: Task, state: State) -> None:
        runtime_state = state.runtime_state()
        if "max_turns" in task:
            runtime_state.setdefault("max_turns", task["max_turns"])
        task.tools_config()
        task.toolsets_config()

    async def setup_model_state(self, task: Task, state: State) -> None:
        runtime_state = state.runtime_state()
        model_handle = self.runtime.model_handle(state)
        if model_handle is not None:
            if model_handle.model is not None:
                runtime_state.setdefault("model", model_handle.model)
            if model_handle.client_type is not None:
                runtime_state.setdefault("client_type", model_handle.client_type)
            if model_handle.sampling_args is not None:
                runtime_state.setdefault("sampling_args", model_handle.sampling_args)
        task_model = model_config_from_task(task)
        model_name = task_model.name or self.model.name
        if model_name is not None:
            runtime_state.setdefault("model", model_name)
        sampling_args = dict(self.model.sampling_args)
        sampling_args.update(task_model.sampling_args)
        state_sampling_args = runtime_state.get("sampling_args")
        if state_sampling_args is not None:
            if not isinstance(state_sampling_args, dict):
                raise TypeError("state.runtime.sampling_args must be a mapping.")
            sampling_args.update(state_sampling_args)
        if sampling_args:
            runtime_state["sampling_args"] = sampling_args
        task_model_client = (
            cast(ModelClient | None, task_model.client_object())
            if task_model.client is not None
            else None
        )
        model_client = (
            task_model_client if task_model_client is not None else self.model_client
        )
        if (
            model_handle is None
            and "client_key" not in runtime_state
            and model_client is not None
        ):
            self.runtime.bind_model_client(state, model_client)

    async def setup_tool_state(self, task: Task, state: State) -> None:
        self.runtime.prepare_state(task, state)
        self.runtime.validate_bindings(state, allow_unresolved_tool_bindings=True)
        await self.runtime.ensure_mcp_tools(state)
        self.runtime.refresh_tools(state, validate=False)

    async def setup_sandbox_state(self, state: State) -> None:
        await self.runtime.ensure_global_sandboxes(state)
        self.runtime.bind_global_sandboxes(state)

    async def setup_default_state_fields(self, state: State) -> None:
        state.setdefault("artifacts", {})
        state.setdefault("metrics", {})
        state.setdefault("reward", 0.0)
        state.ensure_timing()

    async def resolve_system_prompt(self, task: Task, state: State) -> None:
        taskset_system_prompt = (
            self.taskset.system_prompt if self.taskset is not None else []
        )
        state["system_prompt"] = resolve_system_prompt(
            task=task,
            taskset_system_prompt=taskset_system_prompt,
            harness_system_prompt=self.system_prompt,
            strategy=self.system_prompt_strategy,
        )

    async def run_program(self, task: Task, state: State) -> State:
        endpoint = self.resolved_endpoint(state)
        result = await run_intercepted_program(
            self.program, endpoint, self.runtime, task, state
        )
        if result is None:
            return state
        if isinstance(result, State):
            return result
        if isinstance(result, dict):
            state.update(result)
            return state
        raise TypeError("Harness program must return None, State, or a mapping.")

    def resolved_endpoint(self, state: State) -> Endpoint:
        handle = self.runtime.endpoint_handle(state)
        if handle is None:
            return self.endpoint
        runtime = self.runtime.resolved_runtime(handle)
        harness = runtime.harness
        if harness is None:
            raise RuntimeError("Resolved endpoint handle has no live harness.")
        endpoint = harness.endpoint
        if not isinstance(endpoint, Endpoint):
            raise RuntimeError("Resolved endpoint handle has no live endpoint.")
        return endpoint

    def compile_program(self, program: ProgramConfig) -> ProgramRunner:
        program_data = program.data()
        if not isinstance(program_data, dict):
            raise TypeError("program must materialize to a mapping.")
        kind = program_kind(program_data)
        if kind == "base":
            sandbox_config = self.program_sandbox_config(program)
            validate_program_options(program_data, kind, sandbox_config)
            if sandbox_config is not None:
                return self.sandbox_base_program(program_data, sandbox_config)
            return self.base_program
        if kind == "fn":
            sandbox_config = self.program_sandbox_config(program)
            validate_program_options(program_data, kind, sandbox_config)
            fn_ref = program_data["fn"]
            if not isinstance(fn_ref, str):
                raise TypeError("program.fn must be a string ref.")
            if sandbox_config is not None:
                return self.sandbox_fn_program(
                    program_data, sandbox_config, qualified_config_ref(fn_ref)
                )
            fn = import_config_ref(fn_ref)
            if not callable(fn):
                raise TypeError("program.fn did not resolve to a callable.")
            return self.local_callable_program(cast(Handler, fn))
        if kind == "command":
            sandbox_config = self.program_sandbox_config(program)
            validate_program_options(program_data, kind, sandbox_config)
            return self.command_program(program_data, sandbox_config)
        raise AssertionError(f"Unhandled program kind: {kind}")

    def local_callable_program(self, fn: Handler) -> ProgramRunner:
        async def run(task: Task, state: State) -> ProgramResult:
            await self.runtime.setup_rollout(task, state)
            timeout = cast(
                float | None, state.runtime_state().get("task_timeout_seconds")
            )
            try:
                async with asyncio.timeout(timeout):
                    result = await maybe_call_with_named_args(
                        fn, task=task, state=state, runtime=self.runtime, harness=self
                    )
            except TimeoutError:
                state["timed_out"] = True
                state["task_timed_out"] = True
                state.stop("timeout_reached")
                return state
            if result is None or isinstance(result, State | dict):
                return cast(ProgramResult, result)
            raise TypeError("program.fn must return None, State, or a mapping.")

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

        def sync_completion() -> list[JsonData]:
            rendered_messages = [
                message.model_dump(exclude_none=True) for message in messages
            ]
            state["completion"] = assistant_completion_from_messages(
                prompt_messages, rendered_messages
            )
            return rendered_messages

        turn = 0
        max_turns = state.get_max_turns(self.config.max_turns)
        timeout = cast(float | None, state.runtime_state().get("task_timeout_seconds"))
        try:
            async with asyncio.timeout(timeout):
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

                    async def call_tool(tool_call) -> ToolMessage:
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
                        return ToolMessage(tool_call_id=tool_call.id, content=content)

                    messages.extend(
                        await asyncio.gather(
                            *(call_tool(tool_call) for tool_call in tool_calls)
                        )
                    )
                    sync_completion()
                    if await self.runtime.is_completed(task, state):
                        return state
                    if max_turns > 0 and turn >= max_turns:
                        state._set_stop_condition("max_turns_reached", overwrite=True)
                        return state
        except TimeoutError:
            state["timed_out"] = True
            state["task_timed_out"] = True
            state.stop("timeout_reached")
        return state

    def command_program(
        self, program: ConfigData, sandbox_config: SandboxConfig | None
    ) -> ProgramRunner:
        async def run(task: Task, state: State) -> State:
            runtime = self.runtime
            merged_program = merge_task_program(program, task, kind="command")
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
            timeout = cast(
                float | None, state.runtime_state().get("task_timeout_seconds")
            )
            try:
                async with asyncio.timeout(timeout):
                    await run_local_command(merged_program, task, state, runtime)
            except TimeoutError:
                state["timed_out"] = True
                state["task_timed_out"] = True
                state.stop("timeout_reached")
            return state

        return run

    def sandbox_base_program(
        self, program: ConfigData, sandbox_config: SandboxConfig
    ) -> ProgramRunner:
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
        program: ConfigData,
        sandbox_config: SandboxConfig,
        fn_ref: str,
    ) -> ProgramRunner:
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

    def program_sandbox_config(self, program: ProgramConfig) -> SandboxConfig | None:
        sandbox = program.resolve().sandbox
        if sandbox is False:
            return None
        if self.sandbox is None:
            return None
        if sandbox is None and self.config.sandbox is None:
            return None
        validate_program_sandbox_scope(self.sandbox)
        return self.sandbox

    def prepare_sandbox_program(self, program: ConfigData, state: State) -> ConfigData:
        if "mcp" in program_channels(program):
            endpoint_root_url = state.get("endpoint_root_url")
            if not isinstance(endpoint_root_url, str):
                raise RuntimeError("MCP program tools require an active endpoint.")
            api_key_var = state.get("endpoint_api_key_var")
            if not isinstance(api_key_var, str):
                api_key_var = "OPENAI_API_KEY"
            return proxy_program(
                program,
                tool_base_url=f"{endpoint_root_url.rstrip('/')}/vf/tools",
                tool_auth_var=api_key_var,
            )
        return program

    def prepare_sandbox_config(
        self, sandbox_config: SandboxConfig, program: ConfigData
    ) -> SandboxConfig:
        config = sandbox_config.data()
        if "mcp" in program_channels(program):
            config = proxy_sandbox(config)
        if program_kind(program) in {"base", "fn"}:
            config = python_program_sandbox(config)
        return SandboxConfig.model_validate(config)
