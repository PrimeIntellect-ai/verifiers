# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "harbor-rewardkit[documents] @ git+https://github.com/harbor-framework/harbor.git@548b8d30d2e406ab8fe246da5caaea7bee0aca18#subdirectory=packages/rewardkit",
# ]
# ///
"""RewardKit semantics with native execution; also the verifier discovery worker.

Install the pinned dependency above in the evaluator and verifier image. Discovery,
task Python and document conversion run only in the verifier; the host receives JSON.
Private upstream interfaces are confined to this revision-scoped adapter.
"""

import asyncio
import json
import sys
import tomllib
from importlib import import_module
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from types import SimpleNamespace

REWARDKIT_REVISION = "548b8d30d2e406ab8fe246da5caaea7bee0aca18"
REWARDKIT_REQUIREMENT = (
    "harbor-rewardkit[documents] @ git+https://github.com/harbor-framework/harbor.git@"
    f"{REWARDKIT_REVISION}#subdirectory=packages/rewardkit"
)
try:
    source = json.loads(
        distribution("harbor-rewardkit").read_text("direct_url.json") or "{}"
    )
except PackageNotFoundError:
    source = {}
if source.get("vcs_info", {}).get("commit_id") != REWARDKIT_REVISION:
    raise ImportError(
        f"Native Harbor grading requires: uv add '{REWARDKIT_REQUIREMENT}'"
    )

from rewardkit.agents import merge_usage
from rewardkit.judges import (
    _arun_agent_call,
    _build_judge_blocks,
    _extract_guard,
    _fence,
    _join_labeled,
    _with_guard,
    build_prompt,
)
from rewardkit.models import (
    AgentJudge,
    JevJudge,
    JudgeResult,
    JudgeTomlConfig,
    LLMJudge,
    RewardAggregationConfig,
    Score,
)
from rewardkit.reward import Reward
from rewardkit.runner import (
    _build_criteria_from_toml,
    _discover_layout,
    _evaluate_layout,
    _flatten_layout,
    _RewardGroup,
    _RewardLayout,
    _run_all,
    _subdirs,
)

evaluate_layout = _evaluate_layout
fence_submission = _fence


def load_native_trajectory(path: str | Path) -> dict | None:
    """Supply sampled native nodes to RewardKit's existing trajectory criteria."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("version") != 1 or "nodes" not in data:
        raise ValueError("Trajectory criteria require a native Verifiers v1 trace")
    return {
        "steps": [
            {
                "source": "agent",
                "tool_calls": [
                    {
                        "function_name": ".".join(
                            p for p in (c.get("namespace"), c["name"]) if p
                        )
                    }
                    for c in node["message"].get("tool_calls") or []
                ],
            }
            for node in data["nodes"]
            if node["sampled"] and node["message"]["role"] == "assistant"
        ]
    }


class NativeReward(Reward):
    """Replace model execution while upstream owns parsing and score reduction."""

    async def arun(self, sem=None):
        # Preserve provider/runtime exception types for VF's scoring retry policy.
        try:
            return await self._arun_inner(sem)
        except ExceptionGroup as exc:
            cause = exc
            while isinstance(cause, ExceptionGroup):
                cause = cause.exceptions[0]
            raise cause from exc

    async def _judge_once(self, judge) -> JudgeResult:
        criteria, weights = self.criteria, self.weights
        files = tuple(
            dict.fromkeys(f for c in criteria for f in (c.files or judge.files))
        )
        guarded = judge.guard != "off"
        if guarded:
            criteria, weights = _with_guard(criteria, weights, files)
        batches = (
            [
                ([c], [weights[i]] if weights else None, c.files or judge.files)
                for i, c in enumerate(criteria)
            ]
            if judge.mode == "individual"
            else [(criteria, weights, judge.files)]
        )

        async def grade(batch, batch_weights, files):
            prompt = build_prompt(batch, self.system_prompt, kind="llm", guard=guarded)

            async def run(prompt, schema, label):
                return await self.call(prompt, schema, self.contents[tuple(files)])

            # Reuse upstream's schema validation and retry loop without its CLI backend.
            return await _arun_agent_call(
                judge.model_copy(update={"guard": "off"}),
                batch,
                batch_weights,
                prompt,
                SimpleNamespace(run=run),
            )

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
    """Discover, run Python criteria and render evidence inside the verifier."""
    for name in (
        "trajectory_tool_used",
        "trajectory_tool_not_used",
        "trajectory_turn_count",
    ):
        import_module(
            f"rewardkit.criteria.{name}"
        ).load_trajectory = load_native_trajectory
    from rewardkit import runner

    # Python supplied as judge evidence must not also execute as a criterion module.
    evidence = set()
    directories = [Path(tests)]
    tomls = []
    while directories:
        directory = directories.pop()
        directories.extend(_subdirs(directory))
        tomls.extend(directory.glob("*.toml"))
    for path in tomls:
        raw = tomllib.loads(path.read_text())
        if "judge" not in raw and "criterion" not in raw:
            continue
        config = JudgeTomlConfig.model_validate(raw)
        evidence.update(
            (Path(workspace) / filename).resolve()
            for filename in (
                *config.judge.files,
                *([config.judge.reference] if config.judge.reference else []),
                *(filename for c in config.criterion for filename in c.files),
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
        if node.name in exported:
            return {"reward": node.name}
        config = (
            JudgeTomlConfig.model_validate(
                tomllib.loads((Path(tests) / f"{node.name}.toml").read_text())
            )
            if node.judge is not None
            else None
        )
        contents = []
        if isinstance(node.judge, LLMJudge):
            judge = node.judge
            groups = {judge.files}
            if judge.mode == "individual":
                groups = {c.files or judge.files for c in node.criteria}
                if judge.guard != "off":
                    groups.add(
                        tuple(
                            dict.fromkeys(
                                f
                                for c in node.criteria
                                for f in (c.files or judge.files)
                            )
                        )
                    )
            contents = [
                {
                    "files": files,
                    "blocks": _build_judge_blocks(
                        files, judge.reference, guard=judge.guard != "off"
                    ),
                }
                for files in sorted(groups)
            ]
        exported[node.name] = {
            "config": config.model_dump(mode="json", by_alias=True, exclude_unset=True)
            if config
            else None,
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
            "contents": contents,
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
    """Restore only data; task Python and file discovery never run on the host."""
    judges = {c.__name__: c for c in (LLMJudge, AgentJudge, JevJudge)}
    rewards = {}
    for name, spec in data["rewards"].items():
        judge = spec["judge"]
        config = JudgeTomlConfig.model_validate(spec["config"]) if judge else None
        reward = (NativeReward if judge else Reward)(
            criteria=_build_criteria_from_toml(config.criterion, Path(name))
            if config
            else [],
            weights=[c.weight for c in config.criterion] if config else None,
            judge=judges[judge["type"]](**judge["value"]) if judge else None,
            **spec["settings"],
        )
        reward.scores = [Score.model_validate(s) for s in spec["scores"]]
        reward.warnings = spec["warnings"]
        reward.contents = {tuple(c["files"]): c["blocks"] for c in spec["contents"]}
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


if __name__ == "__main__":
    plan = asyncio.run(discover_plan(sys.argv[1], sys.argv[2]))
    Path(sys.argv[3]).write_text(json.dumps(plan, allow_nan=False))
