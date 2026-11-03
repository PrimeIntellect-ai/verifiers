"""Run RewardKit in a task runtime and import its grading results into the trace."""

import json
import uuid

from pydantic import ConfigDict, Field, TypeAdapter
from pydantic_config import BaseConfig

from verifiers.v1.configs.harness import ID
from verifiers.v1.configs.judge import JudgePluginConfig
from verifiers.v1.errors import SandboxError, TaskError
from verifiers.v1.judge import Judge, JudgeResponse
from verifiers.v1.runtimes import Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace
from verifiers.v1.types import Usage
from verifiers.v1.utils.env import resolve_env

MAX_REWARD_BYTES = 1024 * 1024
MAX_VERIFIER_OUTPUT_CHARS = 16 * 1024
REWARD_JSON_ADAPTER = TypeAdapter(
    dict[str, float], config=ConfigDict(strict=True, allow_inf_nan=False)
)


class RewardKitConfig(BaseConfig):
    """RewardKit execution overrides; unset fields preserve the authored TOMLs."""

    judge: str | None = None
    """RewardKit selector: an LLM model or an agent backend, passed as --judge."""
    model: str | None = None
    """Model override for agent backends (--model); LLM models use `judge`."""
    reasoning_effort: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    """Provider endpoints and credentials, resolved only when starting RewardKit."""


class RewardKitJudgeConfig(RewardKitConfig, JudgePluginConfig):
    id: ID = "rewardkit"
    path: str = "/tests"
    """Tests directory already present in the grading runtime."""
    reward_key: str = Field("reward", min_length=1)
    """RewardKit output used as this judge's reward; other outputs become metrics."""


class RewardKitJudge(Judge[float, RewardKitJudgeConfig]):
    async def score(
        self,
        task: TaskData,
        trace: Trace,
        runtime: Runtime,
        env: dict[str, str] | None = None,
    ) -> float:
        # Each invocation owns its output files, including concurrent judges and
        # retries. A previous reward can never stand in for a failed invocation.
        directory = f"/logs/verifier/rewardkit/{uuid.uuid4().hex}"
        await runtime.write(f"{directory}/instruction.md", task.prompt_text.encode())
        await runtime.write(f"{directory}/response.txt", trace.last_reply.encode())
        command = [
            "sh",
            "-c",
            (
                'mv "$1/instruction.md" /logs/verifier/instruction.md && '
                'mv "$1/response.txt" /logs/verifier/response.txt && '
                'shift && exec rewardkit "$@" --workspace "$PWD"'
            ),
            "vf-rewardkit",
            directory,
            self.config.path,
            "--output",
            f"{directory}/reward.json",
        ]
        for option in ("judge", "model", "reasoning_effort"):
            if (value := getattr(self.config, option)) is not None:
                command.extend([f"--{option.replace('_', '-')}", value])
        result = await runtime.run(
            command,
            resolve_env({**(task.scoring.env or {}), **(env or {}), **self.config.env}),
        )
        trace.info.setdefault("scoring_diagnostics", {})[self.reward_name] = {
            "exit_code": result.exit_code,
            "stdout": result.stdout[-MAX_VERIFIER_OUTPUT_CHARS:],
            "stderr": result.stderr[-MAX_VERIFIER_OUTPUT_CHARS:],
            "stdout_truncated": len(result.stdout) > MAX_VERIFIER_OUTPUT_CHARS,
            "stderr_truncated": len(result.stderr) > MAX_VERIFIER_OUTPUT_CHARS,
        }
        if result.exit_code:
            raise TaskError(
                f"RewardKit judge failed (exit {result.exit_code}): "
                f"{(result.stderr or result.stdout).strip()[-500:]}"
            )
        try:
            details = json.loads(
                await runtime.read(
                    f"{directory}/reward-details.json", max_bytes=MAX_REWARD_BYTES
                )
            )
            if not isinstance(details, dict):
                raise TypeError("expected an object of reward details")
        except (SandboxError, OSError, ValueError, TypeError) as exc:
            raise TaskError("RewardKit produced no valid reward-details.json") from exc

        errors = []
        pending = list(details.items())
        while pending:
            name, detail = pending.pop()
            if "components" in detail:
                pending.extend(
                    (f"{name}/{component['name']}", component["detail"])
                    for component in detail["components"]
                )
                continue
            prefix = f"{self.reward_name}/{name}"
            if "judge" in detail:
                usage = detail.get("usage")
                # RewardKit reports aggregate usage per component, with cache
                # reads included in input_tokens. Do not add per-model totals again.
                trace.record_judge_call(
                    name=prefix,
                    request={
                        "model": detail["judge"].get("model"),
                        "judge": detail["judge"],
                        "backend": "rewardkit",
                        "scope": "aggregate",
                    },
                    response=JudgeResponse(
                        text=detail.get("judge_output", ""),
                        parsed=detail,
                        usage=Usage(
                            prompt_tokens=usage.get("input_tokens", 0)
                            - usage.get("cache_read_input_tokens", 0),
                            completion_tokens=usage.get("output_tokens", 0),
                            cached_input_tokens=usage.get("cache_read_input_tokens"),
                            reasoning_tokens=usage.get("reasoning_output_tokens"),
                            cost=usage.get("cost_usd"),
                        )
                        if usage
                        else None,
                    ),
                )
            for criterion in detail["criteria"]:
                if criterion.get("error") is not None:
                    errors.append(criterion["error"])
                else:
                    trace.record_metric(
                        f"{prefix}/{criterion['name']}", criterion["value"]
                    )
            if error := (detail.get("guard") or {}).get("error"):
                errors.append(error)
        # Preserve verdicts and billed usage even when a timed-out criterion makes
        # the aggregate unusable as a training reward.
        if errors:
            raise TaskError(f"RewardKit judge failed: {'; '.join(errors)}")
        try:
            scores = REWARD_JSON_ADAPTER.validate_json(
                await runtime.read(
                    f"{directory}/reward.json", max_bytes=MAX_REWARD_BYTES
                )
            )
        except (SandboxError, OSError, ValueError) as exc:
            raise TaskError("RewardKit produced no valid reward.json") from exc
        key = self.config.reward_key
        if key not in scores:
            raise TaskError(
                f"RewardKit reward key {key!r} is missing; available keys: {sorted(scores)}"
            )
        trace.record_metrics(
            {
                f"{self.reward_name}/{name}": value
                for name, value in scores.items()
                if name != key
            }
        )
        return scores[key]


__all__ = ["RewardKitConfig", "RewardKitJudge", "RewardKitJudgeConfig"]
