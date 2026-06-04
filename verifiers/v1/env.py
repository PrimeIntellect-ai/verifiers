import asyncio
import uuid
from typing import TYPE_CHECKING, cast

import verifiers as vf
from verifiers.clients import Client
from verifiers.types import ClientConfig
from verifiers.types import RolloutInput, SamplingArgs
from verifiers.utils.error_utils import error_info, note_secondary_error

from .config import Config
from .harness import Harness, HarnessConfig
from .state import State
from .taskset import Taskset, TasksetConfig
from .types import JsonData, RuntimeData
from .utils.taskset_utils import task_from_dataset_record

if TYPE_CHECKING:
    from datasets import Dataset


class EnvConfig(Config):
    taskset: TasksetConfig = TasksetConfig()
    harness: HarnessConfig = HarnessConfig()

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: object) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        extra_fields = set(cls.model_fields) - set(EnvConfig.model_fields)
        if extra_fields:
            raise TypeError(
                f"{cls.__name__} defines unsupported root env config fields: "
                f"{', '.join(sorted(extra_fields))}. Put env-specific settings on "
                "a TasksetConfig or HarnessConfig instead."
            )
        for field_name, expected_type in (
            ("taskset", TasksetConfig),
            ("harness", HarnessConfig),
        ):
            annotation = cls.model_fields[field_name].annotation
            if not (
                isinstance(annotation, type) and issubclass(annotation, expected_type)
            ):
                raise TypeError(
                    f"{cls.__name__}.{field_name} must be typed as a "
                    f"{expected_type.__name__} subclass."
                )


class Env(vf.Environment):
    def __init__(
        self,
        *,
        taskset: Taskset | None = None,
        harness: Harness | None = None,
    ):
        if taskset is None:
            raise TypeError("Env requires a taskset.")
        if not isinstance(taskset, Taskset):
            raise TypeError("Env taskset must be a Taskset.")
        if harness is not None and not isinstance(harness, Harness):
            raise TypeError("Env harness must be a Harness.")
        self.taskset = taskset
        self.harness = harness or Harness(config=HarnessConfig())
        self.config = EnvConfig(
            taskset=cast(TasksetConfig, self.taskset.config),
            harness=cast(HarnessConfig, self.harness.config),
        )
        self.harness.taskset = self.taskset
        self.taskset.runtime_refresh = self.harness.rebuild_runtime
        self.harness.rebuild_runtime()
        super().__init__(
            dataset=self.taskset.get_dataset,
            eval_dataset=self.taskset.get_eval_dataset,
            rubric=vf.Rubric(),
        )
        self._empty_dataset_checked = False
        self._empty_eval_dataset_checked = False

    def build_dataset(self) -> "Dataset | None":
        if self.dataset is not None:
            return self.dataset
        if self._empty_dataset_checked:
            return None
        dataset = self.taskset.get_dataset()
        if not len(dataset):
            self._empty_dataset_checked = True
            return None
        self.dataset = self._format_dataset_source(dataset)
        return self.dataset

    def build_eval_dataset(self) -> "Dataset | None":
        if self.eval_dataset is not None:
            return self.eval_dataset
        if self._empty_eval_dataset_checked:
            return None
        eval_dataset = self.taskset.get_eval_dataset()
        if not len(eval_dataset):
            self._empty_eval_dataset_checked = True
            return None
        self.eval_dataset = self._format_dataset_source(eval_dataset)
        return self.eval_dataset

    @vf.teardown
    async def teardown_harness(self) -> None:
        await self.harness.teardown()

    @property
    def requires_group_rollouts(self) -> bool:
        uses_custom_init_group = type(self.taskset).init_group is not Taskset.init_group
        return self.harness.runtime.has_group_stage or uses_custom_init_group

    @property
    def provides_advantages(self) -> bool:
        return self.harness.runtime.has_group_advantages

    async def rollout(
        self,
        input: RolloutInput,
        client: Client | ClientConfig,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        task = task_from_dataset_record(cast(JsonData, input), self.taskset.taskset_id)
        state = State.for_task(task)
        self.apply_controls(
            [state],
            {
                "client": client,
                "model": model,
                "sampling_args": sampling_args or {},
                "score_rollout": self.score_rollouts,
            },
        )
        return await self.harness.run(task, state)

    async def _run_rollout_state(
        self,
        input: RolloutInput,
        client: Client,
        model: str,
        sampling_args: SamplingArgs,
    ) -> State:
        return await self.rollout(input, client, model, sampling_args)

    async def _run_group_states(
        self,
        group_inputs: list[RolloutInput],
        client: Client,
        model: str,
        sampling_args: SamplingArgs,
    ) -> list[vf.State]:
        base_task = task_from_dataset_record(
            cast(JsonData, group_inputs[0]), self.taskset.taskset_id
        )
        tasks, states = await self.taskset.init_group(base_task, len(group_inputs))
        if len(tasks) != len(group_inputs) or len(states) != len(group_inputs):
            raise ValueError(
                "Taskset.init_group must return one task/state per rollout."
            )
        group_key = uuid.uuid4().hex
        for state in states:
            state.runtime_state()["group_key"] = group_key
        self.apply_controls(
            states,
            {
                "client": client,
                "model": model,
                "sampling_args": sampling_args,
                "score_rollout": self.score_rollouts,
            },
        )
        run_tasks = [
            asyncio.create_task(self.harness.run(task, state))
            for task, state in zip(tasks, states)
        ]
        try:
            states = await asyncio.gather(*run_tasks)
        except BaseException as primary_error:
            pending_state_pairs = [
                (task, state)
                for task, state in zip(run_tasks, states)
                if not task.done()
            ]
            pending = [task for task, _ in pending_state_pairs]
            for task in pending:
                task.cancel()
            secondary_notes: list[str] = []
            if pending:
                drained_errors = await asyncio.gather(*pending, return_exceptions=True)
                for drained_error in drained_errors:
                    if not isinstance(drained_error, BaseException):
                        continue
                    for note in getattr(drained_error, "__notes__", []):
                        secondary_notes.append(f"cancelled group sibling: {note}")
                for _, state in pending_state_pairs:
                    for cleanup_error in state.get("cleanup_errors", []):
                        if not isinstance(cleanup_error, dict):
                            continue
                        stage = cleanup_error.get("stage") or "cleanup"
                        message = cleanup_error.get("message") or cleanup_error.get(
                            "error"
                        )
                        secondary_notes.append(
                            f"cancelled group sibling {stage} failed: {message}"
                        )
                for secondary_note in dict.fromkeys(secondary_notes):
                    add_note = getattr(primary_error, "add_note", None)
                    if callable(add_note):
                        add_note(secondary_note)
                    else:
                        notes = getattr(primary_error, "__notes__", [])
                        if not isinstance(notes, list):
                            notes = [str(notes)]
                        notes.append(secondary_note)
                        setattr(primary_error, "__notes__", notes)
                    self.logger.error(
                        "Cancelled v1 group sibling reported secondary error: %s",
                        secondary_note,
                    )
            try:
                await self.harness.cleanup_group(tasks, states)
            except Exception as cleanup_error:
                for state in states:
                    state.setdefault("cleanup_errors", []).append(
                        error_info(cleanup_error, stage="cleanup")
                    )
                note_secondary_error(primary_error, cleanup_error, stage="cleanup")
                self.logger.exception("Cleanup failed after v1 group rollout error")
            for state in states:
                state.strip_runtime_handles()
            raise
        primary_error = None
        try:
            if self.score_rollouts:
                await self.harness.score_group(tasks, states)
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                await self.harness.cleanup_group(tasks, states)
            except Exception as cleanup_error:
                if primary_error is None:
                    raise
                for state in states:
                    state.setdefault("cleanup_errors", []).append(
                        error_info(cleanup_error, stage="cleanup")
                    )
                note_secondary_error(primary_error, cleanup_error, stage="cleanup")
                self.logger.exception("Cleanup failed after v1 group scoring error")
        for state in states:
            state.strip_runtime_handles()
            state.assert_serializable()
        return cast(list[vf.State], states)

    def apply_controls(
        self, states: list[State], controls: RuntimeData | None = None
    ) -> list[State]:
        if controls is None:
            return states
        serializable_controls = {
            key: value for key, value in controls.items() if key != "client"
        }
        for state in states:
            runtime_state = state.runtime_state()
            client = controls.get("client")
            self.harness.runtime.bind_model_client(
                state,
                cast(Client | ClientConfig | None, client)
                if client is not None
                else None,
            )
            runtime_state.update(serializable_controls)
        return states
