"""Type conversion utilities."""

import time
from datetime import datetime
from pathlib import Path
from typing import Any

from verifiers.types import (
    GenerateMetadata,
    GenerateOutputs,
    RolloutResult,
    RolloutResultTrajectoryStep,
    SamplingArgs,
    State,
)


def build_generate_outputs(
    rollouts: list[RolloutResult],
    env_id: str,
    env_args: dict,
    model: str,
    base_url: str,
    sampling_args: SamplingArgs,
    start_time: float,
    path_to_save: Path,
    state_columns: list[str],
) -> GenerateOutputs:
    """Build GenerateOutputs from a list of RolloutResults."""
    rewards = [r.get("reward", 0.0) for r in rollouts]
    example_ids = [r["example_id"] for r in rollouts]

    # Aggregate metrics
    metrics: dict[str, list[float]] = {}
    for r in rollouts:
        for name, value in r["metrics"].items():
            if name not in metrics:
                metrics[name] = []
            metrics[name].append(value)

    num_unique = len(set(example_ids)) if example_ids else 0
    rollouts_per = len(rollouts) // num_unique if num_unique > 0 else 1

    return GenerateOutputs(
        rollouts=rollouts,
        metadata=GenerateMetadata(
            env_id=env_id,
            env_args=env_args,
            model=model,
            base_url=base_url,
            num_examples=num_unique,
            rollouts_per_example=rollouts_per,
            sampling_args=sampling_args,
            date=datetime.now().isoformat(),
            time_ms=(time.time() - start_time) * 1000.0,
            avg_reward=sum(rewards) / len(rewards) if rewards else 0.0,
            avg_metrics={k: sum(v) / len(v) if v else 0.0 for k, v in metrics.items()},
            state_columns=state_columns,
            path_to_save=path_to_save,
        ),
    )


def _serialize_value(value: Any) -> Any:
    """Attempt to serialize a value for IPC. Raises if not serializable."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if hasattr(value, "model_dump"):
        return value.model_dump()
    raise TypeError(f"Cannot serialize value of type {type(value).__name__}")


def state_to_result(
    state: State,
    state_columns: list[str] | None = None,
) -> RolloutResult:
    """Convert internal State to serializable RolloutResult.

    This is the single conversion point from runtime state to output format.
    Call this at the boundary where internal processing ends and output begins.

    Args:
        state: The internal State object to convert.
        state_columns: Optional list of extra state fields to include.
            Values must be serializable (primitives, dicts, lists, or have model_dump).
            Raises TypeError if a non-serializable value is encountered.
    """
    trajectory: list[RolloutResultTrajectoryStep] = []
    for step in state.get("trajectory", []):
        traj_step: RolloutResultTrajectoryStep = {}
        if step.get("prompt") is not None:
            traj_step["prompt"] = step["prompt"]
        if step.get("completion") is not None:
            traj_step["completion"] = step["completion"]
        if step.get("tokens") is not None:
            traj_step["tokens"] = step["tokens"]
        if step.get("advantage") is not None:
            traj_step["advantage"] = step["advantage"]
        if step.get("extras") is not None:
            traj_step["extras"] = _serialize_value(step["extras"])
        trajectory.append(traj_step)

    # Required fields
    result: RolloutResult = {
        "prompt": state["prompt"],
        "completion": state.get("completion") or [],
        "example_id": state.get("example_id", 0),
        "task": state.get("task", "default"),
        "metrics": state.get("metrics") or {},
    }

    # Optional fields
    if state.get("answer") is not None:
        result["answer"] = state["answer"]
    if state.get("info") is not None:
        result["info"] = state["info"]
    if state.get("reward") is not None:
        result["reward"] = state["reward"]
    if state.get("is_truncated") is not None:
        result["is_truncated"] = state["is_truncated"]
    if state.get("stop_condition") is not None:
        result["stop_condition"] = state["stop_condition"]
    if trajectory:
        result["trajectory"] = trajectory
    if state.get("timing"):
        result["timing"] = state["timing"]
    if state.get("error"):
        err = state["error"]
        result["error"] = f"{type(err).__name__}: {err}"

    # Extra state columns - must be serializable
    if state_columns:
        for col in state_columns:
            if col in state:
                try:
                    result[col] = _serialize_value(state[col])  # type: ignore[literal-required]
                except TypeError as e:
                    raise TypeError(
                        f"Cannot serialize state column '{col}': {e}"
                    ) from e

    return result
