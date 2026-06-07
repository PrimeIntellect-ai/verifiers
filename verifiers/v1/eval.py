from __future__ import annotations

import asyncio
from collections import defaultdict
from pathlib import Path
from typing import cast

from verifiers.types import (
    EvalConfig,
    GenerateOutputs,
    LogCallback,
    ProgressCallback,
    RolloutInput,
    RolloutOutput,
    StartCallback,
)
from verifiers.utils.eval_utils import effective_max_concurrent, filter_inputs
from verifiers.utils.path_utils import is_valid_eval_results_path
from verifiers.utils.save_utils import (
    GenerateOutputsBuilder,
    load_outputs,
    push_results_to_hf_hub,
    save_metadata,
    save_outputs,
    truncate_malformed_trailing_line,
    validate_resume_metadata,
)

from .env import Env
from .state import State
from .task import Task
from .types import ModelConfig
from .utils.json_utils import json_data


def eval_inputs(
    env: Env,
    num_examples: int,
    rollouts_per_example: int,
) -> list[RolloutInput]:
    rows = [
        cast(RolloutInput, json_data(row))
        for row in env.get_eval_dataset(n=num_examples)
    ]
    if rollouts_per_example <= 1:
        return rows
    return [
        cast(RolloutInput, dict(row))
        for _ in range(rollouts_per_example)
        for row in rows
    ]


def progress_inputs(
    env: Env,
    inputs: list[RolloutInput],
    independent_scoring: bool,
) -> list[RolloutInput] | list[list[RolloutInput]]:
    if independent_scoring or not env.requires_group_rollouts:
        return inputs
    groups: dict[object, list[RolloutInput]] = defaultdict(list)
    for row in inputs:
        groups[row["example_id"]].append(row)
    return list(groups.values())


async def run_rollouts(
    env: Env,
    inputs: list[RolloutInput],
    model: ModelConfig,
    max_concurrent: int,
    max_retries: int,
    state_columns: list[str],
) -> list[RolloutOutput]:
    semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None

    async def run_one(row: RolloutInput) -> RolloutOutput:
        if semaphore is None:
            state = await env.run_rollout(row, model=model, max_retries=max_retries)
        else:
            async with semaphore:
                state = await env.run_rollout(row, model=model, max_retries=max_retries)
        task = env.taskset.to_task(json_data(row))
        return RolloutOutput(state.to_output(task, state_columns))

    return list(await asyncio.gather(*(run_one(row) for row in inputs)))


async def run_rollout_groups(
    env: Env,
    inputs: list[RolloutInput],
    model: ModelConfig,
    max_concurrent: int,
    max_retries: int,
    state_columns: list[str],
) -> list[RolloutOutput]:
    semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None
    groups: dict[object, list[RolloutInput]] = defaultdict(list)
    for row in inputs:
        groups[row["example_id"]].append(row)

    async def run_rows(rows: list[RolloutInput]) -> list[RolloutOutput]:
        base_task = env.taskset.to_task(json_data(rows[0]))
        tasks, states = await env.taskset.init_group(base_task, len(rows))
        group_id = rows[0]["example_id"]
        for state in states:
            state.group_id = str(group_id)

        async def run_member(task: Task, state: State) -> State:
            return await env.run_rollout(
                task,
                model=model,
                state=state,
                max_retries=max_retries,
            )

        if semaphore is None:
            states = list(
                await asyncio.gather(
                    *(
                        run_member(task, state)
                        for task, state in zip(tasks, states, strict=True)
                    )
                )
            )
        else:
            async with semaphore:
                states = list(
                    await asyncio.gather(
                        *(
                            run_member(task, state)
                            for task, state in zip(tasks, states, strict=True)
                        )
                    ),
                )
        states = await env.score_group(tasks, states)
        return [
            RolloutOutput(state.to_output(task, state_columns))
            for task, state in zip(tasks, states, strict=True)
        ]

    grouped_outputs = await asyncio.gather(
        *(run_rows(rows) for rows in groups.values())
    )
    return [output for group_outputs in grouped_outputs for output in group_outputs]


async def run_evaluation(
    env: Env,
    config: EvalConfig,
    results_path: Path,
    on_start: StartCallback | None,
    on_progress: ProgressCallback | list[ProgressCallback] | None,
    on_log: LogCallback | None,
) -> GenerateOutputs:
    if config.extra_env_kwargs:
        raise ValueError("extra_env_kwargs are only supported by legacy environments.")

    raw_inputs = eval_inputs(
        env,
        config.num_examples,
        config.rollouts_per_example,
    )
    example_ids = {row["example_id"] for row in raw_inputs}
    num_examples = len(example_ids)
    rollouts_per_example = (
        len(raw_inputs) // num_examples
        if num_examples > 0
        else config.rollouts_per_example
    )
    model_config = ModelConfig(
        client=config.client_config,
        model=config.model,
        sampling_args=dict(config.sampling_args),
    )
    builder = GenerateOutputsBuilder(
        env_id=env.env_id,
        env_args=env.env_args,
        model=config.model,
        client=config.client_config,
        num_examples=num_examples,
        rollouts_per_example=rollouts_per_example,
        state_columns=config.state_columns,
        sampling_args=dict(config.sampling_args),
        results_path=results_path,
        pass_threshold=env.pass_threshold,
    )
    callbacks: list[ProgressCallback] = []
    if isinstance(on_progress, list):
        callbacks = cast(list[ProgressCallback], on_progress)
    elif on_progress is not None:
        callbacks = [on_progress]

    try:
        if is_valid_eval_results_path(results_path):
            validate_resume_metadata(
                results_path=results_path,
                env_id=env.env_id,
                model=config.model,
                num_examples=num_examples,
                rollouts_per_example=rollouts_per_example,
            )
            if on_log is not None:
                on_log(f"Resuming evaluation from {results_path}")
            outputs = load_outputs(results_path)
            truncate_malformed_trailing_line(results_path / "results.jsonl")
            builder.add_outputs(outputs)
            filtered_inputs = filter_inputs(raw_inputs, outputs, rollouts_per_example)
        else:
            filtered_inputs = raw_inputs

        if on_start is not None:
            on_start(
                raw_inputs,
                progress_inputs(env, filtered_inputs, config.independent_scoring),
            )
        if not filtered_inputs:
            return builder.build(sort_by_example_id=True)
        if config.save_results and on_log is not None:
            on_log(f"Saving results to {builder.results_path}")

        if config.independent_scoring or not env.requires_group_rollouts:
            state_columns = config.state_columns or []
            new_outputs = await run_rollouts(
                env,
                filtered_inputs,
                model_config,
                config.max_concurrent,
                config.max_retries,
                state_columns,
            )
        else:
            state_columns = config.state_columns or []
            new_outputs = await run_rollout_groups(
                env,
                filtered_inputs,
                model_config,
                effective_max_concurrent(config),
                config.max_retries,
                state_columns,
            )
        builder.add_outputs(new_outputs)
        metadata = builder.build_metadata()
        for callback in callbacks:
            callback(builder.outputs, new_outputs, metadata)

        results = builder.build(sort_by_example_id=True)
        if config.save_results:
            await asyncio.to_thread(
                save_outputs,
                results["outputs"],
                builder.results_path,
            )
            await asyncio.to_thread(
                save_metadata,
                results["metadata"],
                builder.results_path,
            )
            if config.save_to_hf_hub:
                push_results_to_hf_hub(results, config.hf_hub_dataset_name)
            if on_log is not None:
                on_log(f"Saved final results to {results['metadata']['path_to_save']}")
        return results
    finally:
        await env.close()
