"""Canonical conversion of a v1 `Trace` to the platform's (v0) eval-sample dict.

Shared so the same record shape reaches the platform from every producer: verifiers'
eval `uv run eval --push` client and the prime-rl monitor's sample upload both call
`trace_to_sample`, so a `--push` run, a `prime eval push`, and a training-run sample all
render identically. The conversation is the unit — no prompt/completion split (meaningless
mid-branch): `completion` is the final branch's messages and `trajectory` carries one
message list per branch.
"""

from typing import Any

from verifiers.v1.trace import Trace


def trace_to_sample(trace: Trace, rollout_number: int = 1) -> dict[str, Any]:
    """One rollout -> the platform's sample dict (the "old" v0 eval-sample format).

    `rollout_number` is this rollout's 1-based index within its task's group (callers that
    don't group rollouts can leave it at the default)."""

    def dump(messages):
        return [m.model_dump(mode="json", exclude_none=True) for m in messages]

    task = trace.task.model_dump(mode="json", exclude_none=True)
    branches = trace.branches
    # The platform stores a fixed set of sample columns and folds everything else into `info`;
    # per-rollout sub-rewards / metrics are read back from `info["reward_signals"]`, so the
    # per-reward-function and metric breakdown goes there (a flat top-level key would just be
    # buried in `info` unread).
    info = dict(trace.info)
    reward_signals = {**trace.rewards, **trace.metrics}
    if reward_signals:
        info["reward_signals"] = reward_signals
    return {
        "sample_id": trace.id,
        "example_id": trace.task.idx,
        "rollout_number": rollout_number,
        "task": task,
        "prompt": [],
        "completion": dump(branches[-1].messages) if branches else [],
        "answer": task.get("answer"),
        "reward": trace.reward,
        "timing": trace.timing.model_dump(mode="json", exclude_none=True),
        "is_completed": trace.is_completed,
        "is_truncated": trace.is_truncated,
        "metrics": trace.metrics,
        "error": trace.error.model_dump(mode="json", exclude_none=True)
        if trace.error
        else None,
        "stop_condition": trace.stop_condition,
        "trajectory": [
            {
                "messages": dump(branch.messages),
                "reward": trace.reward,
                "num_input_tokens": branch.num_input_tokens,
                "num_output_tokens": branch.num_output_tokens,
            }
            for branch in branches
        ],
        "token_usage": trace.usage.model_dump(mode="json", exclude_none=True)
        if trace.usage
        else None,
        "info": info or None,
    }
