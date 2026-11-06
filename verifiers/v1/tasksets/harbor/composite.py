"""Runtime discovery and native judge execution over RewardKit's own reward graph."""

import json
import uuid
from pathlib import Path

from verifiers.v1.errors import TaskError

from .judge import prepare_judge


async def grade_composite(task, runtime, trace, env):
    from .rewardkit import evaluate_layout, restore_plan

    # Discovery imports task Python. It must run inside the selected verifier,
    # even when a file only registers criteria and does not itself score anything.
    if not runtime.backend.SUPPORTS_TASK_IMAGES:
        raise TaskError("Composite Harbor discovery requires a sandbox runtime")
    prefix = f"/tmp/vf-criteria-{uuid.uuid4().hex}"
    source = Path(__file__).parent
    # RewardKit's conventional trajectory path carries the native v1 record.
    # Replace solver-owned files/symlinks before staging harness evidence.
    trajectory = "/logs/agent/trajectory.json"
    await runtime.run(["rm", "-f", trajectory], {})
    await runtime.write(
        trajectory,
        trace.to_json(
            exclude={
                "nodes": {
                    "__all__": {"message": {"reasoning_content", "provider_state"}}
                }
            }
        ),
    )
    await runtime.write(prefix + ".py", (source / "programmatic.py").read_bytes())
    await runtime.write(prefix + "-compat.py", (source / "rewardkit.py").read_bytes())
    result = await runtime.run(
        [
            "python3",
            "-I",
            prefix + ".py",
            prefix + "-compat.py",
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
    plan = json.loads(await runtime.read(prefix + ".json", max_bytes=16 * 1024 * 1024))
    layout, rewards = restore_plan(plan)
    prepared = {
        name: await prepare_judge(
            task,
            runtime,
            trace,
            f"/tests/{name}.toml",
            raw=plan["rewards"][name]["config"],
            reward=reward,
            prefix=f"harbor/{name}",
        )
        for name, reward in rewards.items()
        if reward.judge is not None
    }
    # Every judge's inputs and execution options are checked before inference.
    for name, judge in prepared.items():
        await judge.run()
        for field in (
            "scores",
            "guard_score",
            "warnings",
            "usage",
            "judge_output",
            "judge_logs",
        ):
            setattr(rewards[name], field, getattr(judge.reward, field))
    scores, details = evaluate_layout(layout)
    trace.record_metrics(
        {f"harbor/{name}": reward.score for name, reward in rewards.items()}
    )
    trace.record_metrics(
        {
            f"harbor/{name}/{score.name}": score.value
            for name, reward in rewards.items()
            for score in reward.scores
        }
    )
    trace.info["harbor_composite"] = details
    await runtime.write(
        "/logs/verifier/reward.json", json.dumps(scores, allow_nan=False).encode()
    )
    await runtime.write(
        "/logs/verifier/reward-details.json",
        json.dumps(details, allow_nan=False).encode(),
    )
    return scores
