from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import cast, final

import verifiers as vf
from verifiers.clients import Client, resolve_client
from verifiers.types import ClientConfig, RolloutInput, SamplingArgs
from verifiers.serve import EnvClient
from verifiers.utils.async_utils import maybe_retry
from verifiers.utils.save_utils import state_to_output

from .harness import Harness
from .state import State
from .taskset import Taskset


class Env(vf.Environment):
    def __init__(
        self,
        taskset: Taskset,
        harness: Harness,
        config: object | None = None,
    ):
        self.taskset = taskset
        self.harness = harness
        self.config = config
        self.harness.attach_taskset(taskset)
        super().__init__(dataset=self.taskset.get_dataset, rubric=vf.Rubric())

    @vf.teardown
    async def teardown_harness(self) -> None:
        await self.harness.teardown()

    async def rollout(
        self,
        input: RolloutInput,
        client: Client,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        task = self.taskset.to_task(input)
        state = State.for_task(task)
        self.apply_controls(
            [state],
            {
                "client": client,
                "model": model,
                "sampling_args": sampling_args or {},
            },
        )
        return await self.harness.run(task, state)

    @final
    async def run_rollout(
        self,
        input: RolloutInput,
        client: Client | ClientConfig,
        model: str,
        sampling_args: SamplingArgs,
        max_retries: int = 0,
        state_columns: list[str] | None = None,
        env_client: EnvClient | None = None,
    ) -> vf.RolloutOutput:
        if env_client is not None:
            return await super().run_rollout(
                input,
                client,
                model,
                sampling_args,
                max_retries,
                state_columns,
                env_client,
            )

        async def run_rollout_attempt() -> State:
            return await self.rollout(
                input, resolve_client(client), model, sampling_args
            )

        state = await maybe_retry(run_rollout_attempt, max_retries=max_retries)()
        return state_to_output(state, state_columns or [])

    @final
    async def run_group(
        self,
        group_inputs: list[RolloutInput],
        client: Client | ClientConfig,
        model: str,
        sampling_args: SamplingArgs,
        max_retries: int = 0,
        state_columns: list[str] | None = None,
        env_client: EnvClient | None = None,
        **kwargs: object,
    ) -> list[vf.RolloutOutput]:
        if env_client is not None:
            return await super().run_group(
                group_inputs,
                client,
                model,
                sampling_args,
                max_retries,
                state_columns,
                env_client,
            )

        async def run_group_attempt() -> list[State]:
            local_client = resolve_client(client)
            tasks = [self.taskset.to_task(input) for input in group_inputs]
            states = [State.for_task(task) for task in tasks]
            self.apply_controls(
                states,
                {
                    "client": local_client,
                    "model": model,
                    "sampling_args": sampling_args,
                },
            )
            states = await asyncio.gather(
                *[self.harness.run(task, state) for task, state in zip(tasks, states)]
            )
            try:
                await self.harness.score_group(tasks, states)
            finally:
                await self.harness.cleanup_group(tasks, states)
            return states

        states = await maybe_retry(run_group_attempt, max_retries=max_retries)()
        return [state_to_output(state, state_columns or []) for state in states]

    def apply_controls(
        self, states: list[State], controls: Mapping[str, object] | None = None
    ) -> list[State]:
        if controls is None:
            return states
        serializable_controls = {
            key: value for key, value in controls.items() if key != "client"
        }
        for state in states:
            state.setdefault("runtime", {})
            client = controls.get("client")
            self.harness.runtime.bind_model_client(
                state, cast(Client | None, client) if client is not None else None
            )
            state["runtime"].update(serializable_controls)
        return states
