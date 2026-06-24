"""Shared Harbor task parsing and reward helpers."""

import io
import json
import tarfile
import tomllib
from collections.abc import Awaitable, Callable, Iterable
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

RewardStrategy = Literal["mean", "final"]
MinReward = float | dict[str, float]


class HarborModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HarborImageEnvironment(HarborModel):
    """A Harbor task environment backed by a pullable container image."""

    kind: Literal["image"] = "image"
    image: str


class HarborDockerfileEnvironment(HarborModel):
    """A Harbor task environment backed by an environment/Dockerfile build context."""

    kind: Literal["dockerfile"] = "dockerfile"
    context: Path = Field(exclude=True)
    build_timeout: float | None = None


class HarborDefaultEnvironment(HarborModel):
    """A Harbor task with no declared environment; use the runtime default image."""

    kind: Literal["default"] = "default"


HarborEnvironment = Annotated[
    HarborImageEnvironment | HarborDockerfileEnvironment | HarborDefaultEnvironment,
    Field(discriminator="kind"),
]


class HarborStep(HarborModel):
    """One ordered `[[steps]]` entry from a Harbor multi-step task."""

    name: str
    prompt: str
    task_dir: Path = Field(exclude=True)
    harness_timeout: float | None = None
    scoring_timeout: float | None = None
    min_reward: MinReward | None = None
    verifier_env: dict[str, str] = Field(default_factory=dict)


class HarborStepResult(HarborModel):
    name: str
    rewards: dict[str, float] = Field(default_factory=dict)


def load_harbor_config(task_dir: Path) -> dict:
    return tomllib.loads((task_dir / "task.toml").read_text())


def scaled_timeout(value: float | int | None, multiplier: float = 1.0) -> float | None:
    return float(value) * multiplier if value is not None else None


def sum_timeouts(values: Iterable[float | None]) -> float | None:
    resolved = list(values)
    if not resolved or any(value is None for value in resolved):
        return None
    return sum(resolved)


def load_harbor_environment(
    task_dir: Path, config: dict, timeout_multiplier: float = 1.0
) -> HarborEnvironment:
    env_config = config.get("environment", {})
    declared = env_config.get("docker_image")
    if declared:
        return HarborImageEnvironment(image=declared)
    context = task_dir / "environment"
    if (context / "Dockerfile").is_file():
        return HarborDockerfileEnvironment(
            context=context,
            build_timeout=scaled_timeout(
                env_config.get("build_timeout_sec"), timeout_multiplier
            ),
        )
    return HarborDefaultEnvironment()


def load_harbor_steps(
    task_dir: Path, config: dict, timeout_multiplier: float = 1.0
) -> list[HarborStep]:
    root_agent_timeout = config.get("agent", {}).get("timeout_sec")
    root_scoring_timeout = config.get("verifier", {}).get("timeout_sec")
    steps = []
    for raw_step in config.get("steps") or []:
        if not isinstance(raw_step, dict):
            raise ValueError(f"{task_dir.name}: Harbor step must be a table")
        name = raw_step.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{task_dir.name}: Harbor step missing non-empty name")
        step_dir = task_dir / "steps" / name
        instruction_path = step_dir / "instruction.md"
        if not instruction_path.is_file():
            raise FileNotFoundError(f"{name}: missing instruction.md")
        agent_timeout = raw_step.get("agent", {}).get("timeout_sec", root_agent_timeout)
        scoring_timeout = raw_step.get("verifier", {}).get(
            "timeout_sec", root_scoring_timeout
        )
        steps.append(
            HarborStep(
                name=name,
                prompt=instruction_path.read_text().strip(),
                task_dir=step_dir,
                harness_timeout=scaled_timeout(agent_timeout, timeout_multiplier),
                scoring_timeout=scaled_timeout(scoring_timeout, timeout_multiplier),
                min_reward=raw_step.get("min_reward"),
                verifier_env=raw_step.get("verifier", {}).get("env", {}),
            )
        )
    return steps


def harbor_task_prompt(task_dir: Path, steps: list[HarborStep]) -> str:
    instruction = task_dir / "instruction.md"
    if instruction.is_file():
        return instruction.read_text().strip()
    step_names = ", ".join(step.name for step in steps)
    suffix = f" Steps: {step_names}." if step_names else ""
    return f"Complete this Harbor multi-step task.{suffix}"


def valid_harbor_task_dir(task_dir: Path) -> bool:
    try:
        config = load_harbor_config(task_dir)
        steps = load_harbor_steps(task_dir, config)
    except (OSError, tomllib.TOMLDecodeError, ValueError, FileNotFoundError):
        return False
    return (task_dir / "instruction.md").is_file() or bool(steps)


def harbor_step_info(step: HarborStep) -> dict:
    info = step.model_dump(mode="json")
    info["task_dir"] = str(step.task_dir)
    return info


def harbor_step_result_info(result: HarborStepResult) -> dict:
    return result.model_dump(mode="json")


def harbor_step_summary(
    results: list[HarborStepResult], strategy: RewardStrategy
) -> tuple[list[dict], dict[str, float]]:
    return [
        harbor_step_result_info(result) for result in results
    ], aggregate_step_rewards(results, strategy)


async def run_harbor_steps(
    steps: Iterable[HarborStep],
    run_step: Callable[[HarborStep], Awaitable[dict[str, float]]],
) -> list[HarborStepResult]:
    results = []
    for step in steps:
        result = HarborStepResult(name=step.name, rewards=await run_step(step))
        results.append(result)
        if should_stop_after_step(result, step.min_reward):
            break
    return results


def make_harbor_tar(directories: Iterable[tuple[Path, str]]) -> bytes:
    """Tar directory contents under target prefixes in order.

    Later entries with the same archive path intentionally override earlier ones
    on extraction, matching Harbor's root-tests then step-tests merge semantics.
    """
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz", dereference=True) as tar:
        for directory, prefix in directories:
            if not directory.exists():
                continue
            for item in sorted(directory.iterdir()):
                arcname = f"{prefix.rstrip('/')}/{item.name}" if prefix else item.name
                tar.add(item, arcname=arcname)
    return buffer.getvalue()


def parse_harbor_rewards(value: str) -> dict[str, float]:
    stripped = value.strip()
    if not stripped:
        return {}
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        try:
            return {"reward": float(stripped)}
        except ValueError:
            return {}
    if isinstance(data, int | float) and not isinstance(data, bool):
        return {"reward": float(data)}
    if not isinstance(data, dict):
        return {}
    return {
        key: float(raw)
        for key, raw in data.items()
        if isinstance(raw, int | float) and not isinstance(raw, bool)
    }


def min_reward_failed(rewards: dict[str, float], min_reward: MinReward | None) -> bool:
    if min_reward is None:
        return False
    thresholds = (
        {"reward": min_reward} if isinstance(min_reward, int | float) else min_reward
    )
    return any(
        rewards.get(key, float("-inf")) < value for key, value in thresholds.items()
    )


def should_stop_after_step(
    result: HarborStepResult, min_reward: MinReward | None
) -> bool:
    return not result.rewards or min_reward_failed(result.rewards, min_reward)


def aggregate_step_rewards(
    results: list[HarborStepResult], strategy: RewardStrategy
) -> dict[str, float]:
    if not results:
        return {}
    if strategy == "final":
        return dict(results[-1].rewards)
    keys = {key for result in results for key in result.rewards}
    if not keys:
        return {}
    count = len(results)
    return {
        key: sum(result.rewards.get(key, 0.0) for result in results) / count
        for key in keys
    }
