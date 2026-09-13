"""Run the portable subset of Inspect tasks through Verifiers v1.

Inspect supplies the finite dataset and an output-only built-in scorer. Verifiers
owns the harness, runtime, model calls, trace, and scalar reward. Executable Inspect
lifecycle, sandbox, tool, and custom-scorer behavior is rejected rather than silently
approximated; those tasks need a benchmark-specific native v1 port.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, JsonValue, model_validator

import verifiers.v1 as vf
from verifiers.v1.tasksets.inspect.compat import (
    load_inspect_task,
    registry_spec,
    require_inspect,
    task_provenance,
)
from verifiers.v1.tasksets.inspect.messages import convert_input
from verifiers.v1.tasksets.inspect.scoring import (
    InspectScorerSpec,
    compile_scorer,
    score_completion,
)


class InspectConfig(vf.TasksetConfig):
    source: str = ""
    """Exactly one installed ``package/task`` or local ``file.py@task``."""
    source_args: dict[str, JsonValue] = Field(default_factory=dict)
    """Arguments passed to the Inspect task factory."""
    sample_ids: list[str | int] | None = None
    """Optional exact sample-id selection; normal truncation still uses ``-n``."""
    solver_policy: Literal["replace"] = "replace"
    """The Inspect solver is explicitly replaced by the selected v1 harness."""

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        if not self.source.strip():
            raise ValueError(
                "InspectConfig.source is required (an installed package/task or "
                "local file.py@task)"
            )
        return self


class InspectData(vf.TaskData):
    adapter_version: Literal[1] = 1
    source: str
    source_args: dict[str, JsonValue] = Field(default_factory=dict)
    inspect_ai_version: str
    inspect_task_name: str
    inspect_task_version: str | int
    inspect_sample_id: str | int
    dataset_name: str | None = None
    targets: list[str] = Field(default_factory=list)
    choices: list[str] = Field(default_factory=list)
    sample_metadata: dict[str, JsonValue] = Field(default_factory=dict)
    scorer: InspectScorerSpec
    source_solver: str
    solver_policy: Literal["replace"] = "replace"
    ignored_task_settings: list[str] = Field(default_factory=list)
    source_distribution: str | None = None
    source_distribution_version: str | None = None
    source_revision: str | None = None
    source_origin: str | None = None


class InspectTask(vf.Task[InspectData]):
    @property
    def key(self) -> str:
        # Exclude only the taskset-local row number. In particular, include prompt,
        # targets, scorer, and Inspect/package provenance so an edited local task or
        # upgraded scorer implementation cannot retain the same task identity.
        identity = json.dumps(
            self.data.model_dump(mode="json", exclude={"idx"}),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return f"inspect:{hashlib.sha256(identity.encode()).hexdigest()}"

    @vf.reward
    async def inspect_score(self, trace: vf.Trace) -> dict[str, float]:
        completion = (
            trace.root_reply
            if trace.root_reply is not None
            else (
                trace.assistant_messages[-1].content or ""
                if trace.assistant_messages
                else ""
            )
        )
        value, details = await score_completion(
            self.data.scorer,
            completion,
            sample_id=self.data.inspect_sample_id,
            prompt=self.data.prompt_text,
            targets=self.data.targets,
            choices=self.data.choices,
            metadata=self.data.sample_metadata,
        )
        trace.info.setdefault("inspect", {})["score"] = details
        return {f"inspect/{self.data.scorer.short_name}": value}


def _ignored_settings(task: Any) -> list[str]:
    ignored: list[str] = []
    for name in (
        "model",
        "model_roles",
        "metrics",
        "checkpoint",
        "on_checkpoint",
        "on_resume",
        "approval",
        "epochs",
        "epochs_reducer",
        "fail_on_error",
        "continue_on_fail",
        "score_on_error",
        "message_limit",
        "token_limit",
        "turn_limit",
        "time_limit",
        "working_limit",
        "cost_limit",
        "early_stopping",
    ):
        if getattr(task, name, None) not in (None, False, [], {}):
            ignored.append(name)
    config = getattr(task, "config", None)
    if config is not None and config.model_dump(exclude_none=True):
        ignored.append("generate_config")
    return ignored


def _validate_task(task: Any) -> tuple[InspectScorerSpec, str]:
    if getattr(task, "sample_source", None) is not None:
        raise ValueError("dynamic Inspect SampleSource tasks need a native v1 port")
    if getattr(task, "setup", None) is not None:
        raise ValueError(
            "Inspect Task.setup executes even when the main solver is replaced and "
            "needs a native v1 port"
        )
    if getattr(task, "cleanup", None) is not None:
        raise ValueError("Inspect Task.cleanup needs a native v1 finalize port")
    if getattr(task, "sandbox", None) is not None:
        raise ValueError("Inspect task sandboxes need a native v1 runtime port")
    try:
        solver_name, solver_args = registry_spec(task.solver)
    except ValueError as e:
        raise ValueError(
            "custom Inspect solver code needs a native v1 harness/task port"
        ) from e
    if solver_name != "inspect_ai/generate" or solver_args:
        raise ValueError(
            f"Inspect solver {solver_name!r} with arguments {solver_args!r} cannot be "
            "discarded safely; the generic adapter accepts only bare generate()"
        )
    scorers = getattr(task, "scorer", None)
    if not scorers:
        raise ValueError("Inspect task has no scorer to turn into a v1 reward")
    if len(scorers) != 1:
        raise ValueError(
            f"Inspect task has {len(scorers)} scorers; choose a primary reward in a "
            "benchmark-specific v1 port"
        )
    return compile_scorer(scorers[0]), solver_name


def _sample_id(sample: Any, source_index: int) -> str | int:
    # Inspect assigns missing ids from one, in source order.
    return sample.id if sample.id is not None else source_index + 1


def _trace_source(source: str) -> str:
    """Keep a useful reference in traces without retaining an absolute host path."""
    module, separator, task_name = source.rpartition("@")
    if separator and module.endswith(".py"):
        return f"{Path(module).name}@{task_name}"
    return source


def _validate_sample(sample: Any, sample_id: str | int) -> None:
    if sample.sandbox is not None:
        raise ValueError(
            f"Inspect sample {sample_id!r} declares a sandbox and needs a native "
            "v1 runtime port"
        )
    if sample.files:
        raise ValueError(
            f"Inspect sample {sample_id!r} declares files and needs a native v1 "
            "runtime-input port"
        )
    if sample.setup:
        raise ValueError(
            f"Inspect sample {sample_id!r} declares a setup script and needs a "
            "native v1 setup port"
        )
    if sample.checkpoint is not None:
        raise ValueError(
            f"Inspect sample {sample_id!r} declares checkpoint behavior and needs a "
            "native v1 port"
        )


class InspectTaskset(vf.Taskset[InspectTask, InspectConfig]):
    def load(self) -> list[InspectTask]:
        inspect_version = require_inspect()
        source_task = load_inspect_task(
            self.config.source, dict(self.config.source_args)
        )
        scorer, source_solver = _validate_task(source_task)
        provenance = task_provenance(source_task)
        task_name = getattr(source_task, "name", self.config.source)
        task_version = getattr(source_task, "version", 0)
        ignored_settings = _ignored_settings(source_task)
        trace_source = _trace_source(self.config.source)
        selected = (
            {str(sample_id) for sample_id in self.config.sample_ids}
            if self.config.sample_ids is not None
            else None
        )

        tasks: list[InspectTask] = []
        seen_ids: set[str] = set()
        for source_index, sample in enumerate(source_task.dataset):
            sample_id = _sample_id(sample, source_index)
            if selected is not None and str(sample_id) not in selected:
                continue
            normalized_id = str(sample_id)
            if normalized_id in seen_ids:
                raise ValueError(f"duplicate Inspect sample id {sample_id!r}")
            seen_ids.add(normalized_id)
            _validate_sample(sample, sample_id)
            targets = (
                list(sample.target)
                if isinstance(sample.target, list)
                else [sample.target]
            )
            tasks.append(
                InspectTask(
                    InspectData(
                        idx=len(tasks),
                        name=f"{task_name}#{sample_id}",
                        description=(
                            f"Inspect task {task_name}, sample {sample_id}; source "
                            f"solver {source_solver} replaced by "
                            "the selected Verifiers harness"
                        ),
                        prompt=convert_input(sample.input),
                        source=trace_source,
                        source_args=self.config.source_args,
                        inspect_ai_version=inspect_version,
                        inspect_task_name=task_name,
                        inspect_task_version=task_version,
                        inspect_sample_id=sample_id,
                        dataset_name=getattr(source_task.dataset, "name", None),
                        targets=targets,
                        choices=list(sample.choices or []),
                        sample_metadata=dict(sample.metadata or {}),
                        scorer=scorer,
                        source_solver=source_solver,
                        solver_policy=self.config.solver_policy,
                        ignored_task_settings=ignored_settings,
                        source_distribution=provenance["distribution"],
                        source_distribution_version=provenance["distribution_version"],
                        source_revision=provenance["revision"],
                        source_origin=provenance["origin"],
                    ),
                    self.config.task,
                )
            )

        if selected is not None:
            missing = sorted(selected - seen_ids)
            if missing:
                raise ValueError(
                    f"Inspect source {self.config.source!r} has no samples with ids "
                    f"{missing}"
                )
        if not tasks:
            raise ValueError(
                f"Inspect source {self.config.source!r} yielded no samples"
            )
        return tasks
