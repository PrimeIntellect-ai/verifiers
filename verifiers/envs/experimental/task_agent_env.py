from __future__ import annotations

import uuid

from prime_sandboxes import CreateSandboxRequest

from verifiers.clients import Client
from verifiers.decorators import cleanup, stop, teardown
from verifiers.envs.environment import Environment
from verifiers.envs.experimental.harnesses.base import Harness
from verifiers.envs.multiturn_env import MultiTurnEnv
from verifiers.envs.experimental.sandbox_mixin import SandboxMixin, SandboxMonitorRubric
from verifiers.envs.experimental.tasksets.base import Task, TaskSet
from verifiers.rubrics.rubric import Rubric
from verifiers.types import Messages, Response, SamplingArgs, State, Tool
from verifiers.utils.message_utils import normalize_messages


class TaskAgentEnv(SandboxMixin, MultiTurnEnv):
    """Composable multi-turn environment built from a taskset and an agent harness."""

    def __init__(
        self,
        harness: Harness | None,
        taskset: TaskSet,
        max_turns: int = -1,
        environment_vars: dict[str, str] | None = None,
        max_retries: int = 5,
        base_delay: float = 0.5,
        backoff_factor: float = 2.0,
        max_backoff_seconds: float = 30.0,
        jitter: float = 1e-3,
        sandbox_client_max_workers: int = 10,
        sandbox_client_max_connections: int = 100,
        sandbox_client_max_keepalive_connections: int = 50,
        sandbox_wait_for_creation_max_attempts: int = 120,
        **kwargs,
    ):
        if "dataset" in kwargs or "eval_dataset" in kwargs:
            raise ValueError("TaskAgentEnv derives datasets from the provided taskset.")

        rubric = kwargs.pop("rubric", None) or Rubric()
        kwargs.setdefault("message_type", "chat")
        super().__init__(
            dataset=taskset.get_dataset(),
            rubric=rubric,
            max_turns=max_turns,
            **kwargs,
        )

        self.init_sandbox_client(
            max_retries=max_retries,
            base_delay=base_delay,
            backoff_factor=backoff_factor,
            max_backoff_seconds=max_backoff_seconds,
            jitter=jitter,
            sandbox_client_max_workers=sandbox_client_max_workers,
            sandbox_client_max_connections=sandbox_client_max_connections,
            sandbox_client_max_keepalive_connections=sandbox_client_max_keepalive_connections,
            sandbox_wait_for_creation_max_attempts=sandbox_wait_for_creation_max_attempts,
        )
        self.taskset = taskset
        resolved_harness = harness or taskset.build_harness()
        if resolved_harness is None:
            raise ValueError(
                "TaskAgentEnv requires a harness or taskset.build_harness()."
            )
        self.harness = resolved_harness
        self.base_environment_vars = dict(environment_vars or {})

        self.add_rubric(SandboxMonitorRubric())
        for rubric in (
            self.harness.build_monitor_rubric(),
            self.taskset.build_rubric(),
            self.taskset.build_monitor_rubric(),
        ):
            if rubric is not None:
                self.add_rubric(rubric)

    def get_task(self, state: State) -> Task:
        task = state.get("task_instance")
        if isinstance(task, Task):
            return task
        task = self.taskset.get_task(state)
        state["task_instance"] = task
        return task

    async def build_env_vars(self, state: State) -> dict[str, str]:
        return dict(self.base_environment_vars)

    async def default_get_prompt_messages(self, state: State) -> Messages:
        return await MultiTurnEnv.get_prompt_messages(self, state)

    async def request_model_response(
        self,
        state: State,
        prompt: Messages | str,
        client: Client | None = None,
        model: str | None = None,
        tool_defs: list[Tool] | None = None,
        sampling_args: SamplingArgs | None = None,
    ) -> Response:
        return await Environment.get_model_response(
            self,
            state=state,
            prompt=prompt,
            client=client,
            model=model,
            tool_defs=tool_defs,
            sampling_args=sampling_args,
        )

    async def record_model_response(
        self,
        state: State,
        prompt_messages: Messages,
        response: Response,
    ) -> None:
        await MultiTurnEnv.add_model_response(
            self,
            state=state,
            prompt_messages=prompt_messages,
            response=response,
        )

    def normalize_response(self, response: Response) -> Response:
        return self.harness.normalize_response(self, response)

    async def setup_state(self, state: State) -> State:
        state = await super().setup_state(state)
        state.setdefault("rollout_id", f"rollout_{uuid.uuid4().hex[:8]}")

        task = self.get_task(state)
        prompt = await task.prompt(state)
        if prompt is not None:
            state["prompt"] = normalize_messages(prompt, field_name="task.prompt")

        await self.harness.setup(self, state)

        sandbox_spec = await task.get_sandbox_spec(state)
        if sandbox_spec is not None:
            environment_vars = {
                **await self.build_env_vars(state),
                **await task.build_env_vars(state),
                **await self.harness.build_env_vars(self, state),
                **sandbox_spec.environment_vars,
            }
            sandbox_request = CreateSandboxRequest(
                name=state["rollout_id"],
                docker_image=sandbox_spec.docker_image,
                start_command=sandbox_spec.start_command,
                cpu_cores=sandbox_spec.cpu_cores,
                memory_gb=sandbox_spec.memory_gb,
                disk_size_gb=sandbox_spec.disk_size_gb,
                gpu_count=sandbox_spec.gpu_count,
                timeout_minutes=sandbox_spec.timeout_minutes,
                environment_vars=environment_vars,
                team_id=sandbox_spec.team_id,
                advanced_configs=sandbox_spec.advanced_configs,
                labels=list(sandbox_spec.labels),
            )
            await self.create_sandbox(state, sandbox_request)

        await task.setup(self, state)
        await self.harness.start(self, state)
        return state

    async def get_prompt_messages(self, state: State) -> Messages:
        return await self.harness.get_prompt_messages(self, state)

    async def get_model_response(
        self,
        state: State,
        prompt: Messages | str,
        client: Client | None = None,
        model: str | None = None,
        tool_defs: list[Tool] | None = None,
        sampling_args: SamplingArgs | None = None,
    ) -> Response:
        return await self.harness.get_model_response(
            self,
            state=state,
            prompt=prompt,
            client=client,
            model=model,
            tool_defs=tool_defs,
            sampling_args=sampling_args,
        )

    async def add_model_response(
        self,
        state: State,
        prompt_messages: Messages,
        response: Response,
    ) -> None:
        await self.harness.add_model_response(
            self,
            state=state,
            prompt_messages=prompt_messages,
            response=response,
        )

    @stop
    async def agent_completed(self, state: State) -> bool:
        return await self.harness.agent_completed(self, state)

    @stop
    async def timeout_reached(self, state: State) -> bool:
        return await self.harness.timeout_reached(self, state)

    @cleanup
    async def cleanup_harness(self, state: State) -> None:
        await self.harness.cleanup(self, state)

    @cleanup
    async def destroy_sandbox(self, state: State) -> None:
        await self.get_task(state).post_rollout(self, state)
        sandbox_id = state.get("sandbox_id")
        if sandbox_id:
            await self.delete_sandbox(sandbox_id)

    @teardown
    async def teardown_harness(self) -> None:
        await self.harness.teardown(self)

    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Messages | str:
        return await self.get_task(state).env_response(messages, state, **kwargs)
