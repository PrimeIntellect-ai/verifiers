"""Version-scoped RewardKit integration; scoring semantics stay in RewardKit.

Private imports are confined here because 0.2.4 does not expose a backend-neutral
judge-attempt runner or a serializable discovery plan. No task code runs here.
"""

import asyncio
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace

try:
    if version("harbor-rewardkit") != "0.2.4":
        raise ImportError("Native Harbor judging requires harbor-rewardkit==0.2.4")
except PackageNotFoundError as exc:
    raise ImportError(
        "Install harbor-rewardkit==0.2.4 in the evaluator project for native Harbor judging"
    ) from exc

from rewardkit.agents import merge_usage
from rewardkit.judges import (
    _arun_agent_call,
    _extract_guard,
    _join_labeled,
    _with_guard,
    build_prompt,
)
from rewardkit.models import AgentJudge, Binary, JudgeResult, JudgeTomlConfig
from rewardkit.reward import Reward
from rewardkit.runner import _build_judge_reward


class NativeBinary(Binary):
    """Reject malformed answers before RewardKit applies criterion polarity."""

    def normalize(self, raw: float | bool | str) -> float:
        valid = (
            raw.strip().lower() in {"yes", "no", "true", "false", "1", "0"}
            if isinstance(raw, str)
            else isinstance(raw, (bool, int, float)) and raw in (0, 1)
        )
        if not valid:
            raise ValueError(f"Invalid binary judge answer: {raw!r}")
        return super().normalize(raw)


def load_rubric(raw: dict, path: str, workspace: str, template: str | None) -> Reward:
    """Use upstream configuration, criterion construction and cross-field checks."""
    config = JudgeTomlConfig.model_validate(raw)
    if template is not None and "{criteria}" not in template:
        raise ValueError("judge prompt_template must contain {criteria}")
    # The runtime owns file reads; prevent upstream from opening an evaluator path.
    config.judge = config.judge.model_copy(update={"prompt_template": None})
    reward = _build_judge_reward(Path(path), config, Path(path).parent, Path(workspace))
    reward.system_prompt = template
    return reward


class NativeReward(Reward):
    """Run native callbacks through upstream retry, normalization and sample reduction."""

    @classmethod
    def bind(cls, reward: Reward, call, *, files=()) -> "NativeReward":
        bound = cls(
            criteria=reward.criteria,
            weights=reward.weights,
            judge=reward.judge,
            workspace=reward.workspace,
            name=reward.name,
            reward_weight=reward.reward_weight,
            system_prompt=reward.system_prompt,
            aggregation=reward.aggregation,
            threshold=reward.threshold,
        )
        bound.call = call
        bound.files = files
        return bound

    async def arun(self, sem=None):
        # Keep provider/runtime exceptions recognizable to native scoring retries;
        # Reward.arun wraps them in RuntimeError, while its inner runner does not.
        try:
            return await self._arun_inner(sem)
        except ExceptionGroup as exc:
            cause = exc
            while isinstance(cause, ExceptionGroup):
                cause = cause.exceptions[0]
            raise cause from exc

    async def _judge_once(self, judge) -> JudgeResult:
        criteria, weights = self.criteria, self.weights
        agentic = isinstance(judge, AgentJudge)
        default_files = getattr(judge, "files", self.files)
        files = tuple(
            dict.fromkeys(f for c in criteria for f in (c.files or default_files))
        )
        guarded = judge.guard != "off"
        if guarded:
            criteria, weights = _with_guard(criteria, weights, files)
        batches = (
            [
                ([c], [weights[i]] if weights else None, c.files or default_files)
                for i, c in enumerate(criteria)
            ]
            if judge.mode == "individual"
            else [(criteria, weights, files)]
        )

        async def grade(batch, batch_weights, files):
            batch = [
                c.model_copy(update={"output_format": NativeBinary()})
                if isinstance(c.output_format, Binary)
                else c
                for c in batch
            ]
            prompt = build_prompt(
                batch,
                self.system_prompt,
                kind="agent" if agentic else "llm",
                guard=guarded,
            )

            async def run(retry_prompt, schema, label):
                return await self.call(retry_prompt, schema, files)

            # This upstream loop only needs a backend's run() method. LLM calls
            # raise provider errors directly and return output or a timeout, so
            # they share its parse/retry path without invoking an agent CLI.
            return await _arun_agent_call(
                judge.model_copy(update={"guard": "off"}),
                batch,
                batch_weights,
                prompt,
                SimpleNamespace(run=run),
            )

        if agentic:
            results = [await grade(*batch) for batch in batches]
        else:
            async with asyncio.TaskGroup() as group:
                calls = [group.create_task(grade(*batch)) for batch in batches]
            results = [call.result() for call in calls]
        result = JudgeResult(
            [s for r in results for s in r.scores],
            _join_labeled(range(len(results)), (r.output for r in results)),
            [w for r in results for w in r.warnings],
            merge_usage([r.usage for r in results]),
            [log for r in results for log in r.judge_logs],
        )
        return _extract_guard(result) if guarded else result
