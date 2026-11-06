"""Version-scoped RewardKit integration; scoring semantics stay in RewardKit.

Private imports are confined here because 0.2.4 does not expose a backend-neutral
judge-attempt runner or a serializable discovery plan. Only the verifier worker
calls discover_plan, which imports task Python; the evaluator restores JSON data.
"""

import asyncio
import tomllib
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
from rewardkit.models import (
    AgentJudge,
    Binary,
    Criterion,
    JevJudge,
    JudgeResult,
    JudgeTomlConfig,
    Likert,
    LLMJudge,
    Numeric,
    RewardAggregationConfig,
    Rubric,
    Score,
)
from rewardkit.reward import Reward
from rewardkit.runner import (
    _build_judge_reward,
    _discover_layout,
    _evaluate_layout,
    _flatten_layout,
    _RewardGroup,
    _RewardLayout,
    _run_all,
)

evaluate_layout = _evaluate_layout


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
    def bind(cls, reward: Reward, call, *, files=(), selections=None) -> "NativeReward":
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
        bound.selections = selections
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
        selections = self.selections or {
            c.name: c.files or default_files for c in criteria
        }
        files = tuple(dict.fromkeys(f for c in criteria for f in selections[c.name]))
        guarded = judge.guard != "off"
        if guarded:
            criteria, weights = _with_guard(criteria, weights, files)
            selections = {**selections, criteria[-1].name: files}
        batches = (
            [
                ([c], [weights[i]] if weights else None, selections[c.name])
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
                return await self.call(
                    retry_prompt, schema, files, batch, batch_weights
                )

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


async def discover_plan(tests: str, workspace: str) -> dict:
    """Worker-only discovery/execution; export data, never Python code, to the host."""
    from rewardkit import runner

    # Authored evidence is data even when it is Python source. Discovery still
    # imports actual criterion modules, inside this disposable verifier worker.
    evidence = set()
    for path in Path(tests).rglob("*.toml"):
        raw = tomllib.loads(path.read_text())
        if "judge" not in raw and "criterion" not in raw:
            continue
        config = JudgeTomlConfig.model_validate(raw)
        evidence.update(
            (Path(workspace) / filename).resolve()
            for filename in (
                *config.judge.files,
                *([config.judge.reference] if config.judge.reference else []),
                *(
                    filename
                    for criterion in config.criterion
                    for filename in criterion.files
                ),
            )
        )
    importer = runner._import_py_file
    runner._import_py_file = lambda path: (
        None
        if any(path.resolve().is_relative_to(root) for root in evidence)
        else importer(path)
    )
    try:
        layout = _discover_layout(tests, workspace)
    finally:
        runner._import_py_file = importer
    rewards = _flatten_layout(layout)
    # Every native agent attempt receives a fresh runtime, including retries,
    # so upstream's shared-workspace rejection does not apply here.
    configs = {
        r.name: JudgeTomlConfig.model_validate(
            tomllib.loads((Path(tests) / f"{r.name}.toml").read_text())
        ).model_dump(mode="json", by_alias=True, exclude_unset=True)
        for r in rewards
        if r.judge is not None
    }
    await _run_all(
        [r for r in rewards if r.judge is None], max_concurrent_programmatic=8
    )
    exported = {}

    def encode(node):
        if not isinstance(node, Reward):
            return {
                "inputs": {k: encode(v) for k, v in node.inputs.items()},
                "spec": node.spec.model_dump(mode="json") if node.spec else None,
            }
        exported[node.name] = {
            "config": configs.get(node.name),
            "settings": {
                "name": node.name,
                "workspace": str(node.workspace) if node.workspace else None,
                "reward_weight": node.reward_weight,
                "system_prompt": node.system_prompt,
                "aggregation": node.aggregation,
                "threshold": node.threshold,
            },
            "judge": {
                "type": type(node.judge).__name__,
                "value": node.judge.model_dump(mode="json"),
            }
            if node.judge
            else None,
            "criteria": [
                {
                    **c.model_dump(mode="json", exclude={"output_format"}),
                    "output_format": {
                        "type": type(c.output_format).__name__,
                        "value": c.output_format.model_dump(mode="json"),
                    },
                }
                for c in node.criteria
            ]
            if node.judge
            else [],
            "weights": node.weights if node.judge else None,
            "scores": [s.model_dump() for s in node.scores],
            "warnings": node.warnings,
        }
        return {"reward": node.name}

    return {
        "dimensions": {k: encode(v) for k, v in layout.dimensions.items()},
        "spec_inputs": {k: encode(v) for k, v in layout.spec_inputs.items()}
        if layout.spec_inputs is not None
        else None,
        "specs": [s.model_dump() for s in layout.specs],
        "rewards": exported,
    }


def restore_plan(data: dict):
    """Rebuild upstream's graph from a JSON-only, whitelisted model description."""
    formats = {c.__name__: c for c in (Binary, Likert, Numeric, Rubric)}
    judges = {c.__name__: c for c in (LLMJudge, AgentJudge, JevJudge)}
    rewards = {}
    for name, spec in data["rewards"].items():
        judge = spec["judge"]
        criteria = [
            Criterion(
                **(
                    c
                    | {
                        "output_format": formats[c["output_format"]["type"]](
                            **c["output_format"]["value"]
                        )
                    }
                )
            )
            for c in spec["criteria"]
        ]
        reward = Reward(
            criteria=criteria,
            weights=spec["weights"],
            judge=judges[judge["type"]](**judge["value"]) if judge else None,
            **spec["settings"],
        )
        reward.scores = [Score.model_validate(s) for s in spec["scores"]]
        reward.warnings = spec["warnings"]
        rewards[name] = reward

    def decode(node):
        if "reward" in node:
            return rewards[node["reward"]]
        return _RewardGroup(
            inputs={k: decode(v) for k, v in node["inputs"].items()},
            spec=RewardAggregationConfig.model_validate(node["spec"])
            if node["spec"] is not None
            else None,
        )

    return _RewardLayout(
        dimensions={k: decode(v) for k, v in data["dimensions"].items()},
        specs=[RewardAggregationConfig.model_validate(s) for s in data["specs"]],
        spec_inputs={k: decode(v) for k, v in data["spec_inputs"].items()}
        if data["spec_inputs"] is not None
        else None,
    ), rewards
