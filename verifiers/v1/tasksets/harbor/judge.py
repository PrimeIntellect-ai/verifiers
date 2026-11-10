"""Run RewardKit-authored grading through native Verifiers judges."""

import asyncio
import json
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from openai import APITimeoutError

from verifiers.v1.configs.judge import JudgeConfig
from verifiers.v1.errors import TaskError
from verifiers.v1.judge import Judge, JudgeView
from verifiers.v1.types import SystemMessage, UserMessage

MAX_PLAN_BYTES = 32 * 1024 * 1024


class HarborJudgeConfig(JudgeConfig):
    view: JudgeView = "last_reply"
    """Native evidence selection, independent of the authored rubric."""
    include_hidden_reasoning: bool = False
    """Include reasoning in full_trace; opaque provider state is always excluded."""


async def prepare_authored(runtime, env):
    """Resolve the worker's pinned dependencies without activating them in children."""
    interpreter, script = await runtime.prepare_uv_script(
        Path(__file__).with_name("rewardkit.py").read_bytes(), env, activate=False
    )
    return [interpreter, "-I", script]


async def prepare_judge(task, runtime, trace, reward, raw, workspace=None):
    from rewardkit.agents import AgentAttempt
    from rewardkit.models import AgentJudge, JudgeTomlConfig, LLMJudge

    from .rewardkit import fence_submission

    authored = JudgeTomlConfig.model_validate(raw).judge
    if unsupported := {"version", "cwd", "atif_trajectory"} & authored.model_fields_set:
        raise TaskError(f"Unsupported native judge options: {sorted(unsupported)}")
    if reward.judge.timeout <= 0:
        raise TaskError("judge.timeout must be positive")
    if isinstance(reward.judge, AgentJudge):
        from .agent import prepare_agent_judge

        await prepare_agent_judge(task, runtime, trace, reward, authored, workspace)
        return
    if not isinstance(reward.judge, LLMJudge):
        raise TaskError("This native judge requires a RewardKit LLM rubric")
    if "model" in authored.model_fields_set:
        raise TaskError(
            "LLM rubrics select their model with judge.judge; judge.model requires an agent judge"
        )
    if authored.mcp_servers or "isolated" in authored.model_fields_set:
        raise TaskError("judge.mcp_servers and judge.isolated require an agent judge")
    settings = (
        task.config.judge.model_dump(exclude_unset=True) if task.config.judge else {}
    )
    settings.setdefault("model", reward.judge.model)
    settings.setdefault("name", f"harbor/{reward.name}")
    view = settings.pop("view", "last_reply")
    include_reasoning = settings.pop("include_hidden_reasoning", False)
    settings["sampling"] = {
        "reasoning_effort": reward.judge.reasoning_effort
    } | settings.get("sampling", {})
    judge = Judge(JudgeConfig(**settings))
    reward.judge = reward.judge.model_copy(update={"model": judge.config.model})
    response = [{"type": "text", "text": trace.last_reply}]
    if view == "full_trace":
        exclude = {"provider_state"}
        if not include_reasoning:
            exclude.add("reasoning_content")
        response = [
            {
                "type": "text",
                "text": (
                    "Native trajectory: all recorded nodes, once each. Parent and semantic_parents "
                    "identify graph links; sampled marks model responses. Content follows each node's metadata."
                ),
            }
        ]
        for index, node in enumerate(trace.nodes):
            metadata = node.model_dump(
                mode="json",
                include={"parent", "semantic_parents", "sampled", "message"},
                exclude={"message": exclude | {"content"}},
            )
            response.append(
                {
                    "type": "text",
                    "text": json.dumps({"node": index, **metadata}, ensure_ascii=False),
                }
            )
            content = node.message.content
            if content:
                response.extend(
                    [{"type": "text", "text": content}]
                    if isinstance(content, str)
                    else [
                        part.model_dump(mode="json", exclude_none=True)
                        for part in content
                    ]
                )
    if reward.judge.guard != "off":
        response = fence_submission(response)

    async def call(prompt, schema, blocks):
        sampling: dict[str, Any] = {}
        if reward.judge.response_format == "json-schema":
            sampling["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "judge_response",
                    "schema": schema,
                    "strict": True,
                },
            }
        elif reward.judge.response_format == "json-object":
            sampling["response_format"] = {"type": "json_object"}
        content = [
            {"type": "text", "text": f"Task:\n{task.data.prompt_text}\n\nResponse:"},
            *response,
            *blocks,
        ]
        try:
            async with asyncio.timeout(reward.judge.timeout):
                result = await judge.complete(
                    [
                        SystemMessage(content=prompt),
                        UserMessage.model_validate(
                            {"role": "user", "content": content}
                        ),
                    ],
                    trace=trace,
                    **sampling,
                )
        except (TimeoutError, APITimeoutError):
            return AgentAttempt(timed_out=True)
        return AgentAttempt(output=result.text)

    reward.call = call


async def grade_authored(task, runtime, trace, env):
    from rewardkit.models import AgentJudge

    from verifiers.v1.utils.artifacts import ARTIFACTS_DIR, Artifact, collect

    from .rewardkit import evaluate_layout, restore_plan

    if not runtime.backend.SUPPORTS_TASK_IMAGES:
        raise TaskError("Authored Harbor grading requires a sandbox runtime")
    # These conventional inputs also serve authored programmatic criteria.
    await runtime.run(
        ["rm", "-f", "/logs/verifier/instruction.md", "/logs/verifier/response.txt"], {}
    )
    await runtime.write("/logs/verifier/instruction.md", task.data.prompt_text.encode())
    await runtime.write("/logs/verifier/response.txt", trace.last_reply.encode())
    prefix = f"/tmp/vf-rewardkit-{uuid.uuid4().hex}"
    trajectory = "/logs/agent/trajectory.json"
    await runtime.run(["rm", "-f", trajectory], {})
    await runtime.write(
        trajectory,
        trace.to_json(
            exclude={
                "ok": True,
                "nodes": {
                    "__all__": {"message": {"reasoning_content", "provider_state"}}
                },
            }
        ),
    )
    worker = await prepare_authored(runtime, env)
    result = await runtime.run(
        [
            *worker,
            "/tests",
            runtime.config.workdir or "/app",
            prefix + ".json",
        ],
        env,
    )
    trace.info.setdefault("scoring_diagnostics", {})["harbor_programmatic"] = {
        "exit_code": result.exit_code,
        "stdout": result.stdout[-16384:],
        "stderr": result.stderr[-16384:],
    }
    if result.exit_code:
        raise TaskError(
            f"Harbor criterion worker failed (exit {result.exit_code}): {result.stderr[-1000:]}"
        )
    plan = json.loads(await runtime.read(prefix + ".json", max_bytes=MAX_PLAN_BYTES))
    layout, rewards = restore_plan(plan)
    names = {*layout.dimensions, *(spec.name for spec in layout.specs if spec.name)}
    if task.config.reward_key not in names:
        raise TaskError(
            f"Harbor reward key {task.config.reward_key!r} is missing: {sorted(names)}"
        )
    workspace = None
    if any(isinstance(r.judge, AgentJudge) for r in rewards.values()):
        workdir = PurePosixPath(runtime.config.workdir or "/app")
        roots = {
            PurePosixPath("/tests"),
            PurePosixPath("/logs/verifier"),
            PurePosixPath(ARTIFACTS_DIR),
        }
        # A root workdir is valid, but cannot be copied as an artifact. Capture
        # the explicit submission and judge inputs instead of the whole image.
        if workdir != PurePosixPath("/"):
            roots.add(workdir)
        roots.update(workdir / p for p in trace.state.artifacts)
        roots.update(workdir / a.source for a in task.data.artifacts)
        baseline = tuple(roots)
        for spec in plan["rewards"].values():
            # Resolve aliases in the verifier so symlinked evidence captures its
            # actual target, including targets outside the working directory.
            for entry in spec["inputs"]:
                path, resolved = (
                    PurePosixPath(entry["path"]),
                    PurePosixPath(entry["resolved"]),
                )
                if path != resolved and not any(
                    path.is_relative_to(root) for root in baseline
                ):
                    raise TaskError(
                        f"Judge input alias {path} must be inside a captured workspace root"
                    )
                roots.add(resolved)
        # All samples and retries start from this immutable, post-programmatic state.
        workspace = await collect(
            runtime,
            [Artifact(source=str(root), required=False) for root in sorted(roots)],
            max_bytes=512 * 1024 * 1024,
            sweep=False,
            workspace=True,
        )
    # Check every judge before making any inference calls.
    for name, reward in rewards.items():
        if reward.judge is not None:
            await prepare_judge(
                task, runtime, trace, reward, plan["rewards"][name]["config"], workspace
            )
    for name, reward in rewards.items():
        if reward.judge is not None:
            await reward.arun()
        prefix = f"harbor/{name}"
        trace.record_metrics(
            {
                prefix: reward.score,
                **{f"{prefix}/{s.name}": s.value for s in reward.scores},
            }
        )
        if reward.judge is None:
            continue
        diagnostics = {f"criteria/{s.name}": s for s in reward.scores}
        if (guard := reward.guard_score) is not None:
            diagnostics["guard"] = guard
            if guard.error is None:
                trace.record_metric(
                    f"{prefix}/reliability/guard/flagged", guard.value > 0
                )
        for label, score in diagnostics.items():
            trace.record_metrics(
                {
                    f"{prefix}/reliability/{label}/{field}": getattr(score, field)
                    for field in ("agreement", "valid_samples")
                    if getattr(score, field) is not None
                }
            )
            trace.record_metric(
                f"{prefix}/reliability/{label}/error", score.error is not None
            )
    scores, details = evaluate_layout(layout)
    trace.info["harbor_judge"] = details
    await runtime.write(
        "/logs/verifier/reward.json", json.dumps(scores, allow_nan=False).encode()
    )
    await runtime.write(
        "/logs/verifier/reward-details.json",
        json.dumps(details, allow_nan=False).encode(),
    )
    return scores
