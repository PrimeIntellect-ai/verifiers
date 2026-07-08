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

# The platform promotes a numeric sample field to a per-rollout reward column only when its
# name ends in one of these (the v0 reward-function naming convention); any other field folds
# into the sample's `info`.
_REWARD_COLUMN_SUFFIXES = ("reward", "reward_func", "score")


def reward_column_name(name: str) -> str:
    """A reward-function name in the form the platform renders as a per-rollout column: v0
    reward functions already end in `_reward_func`; give any v1 name that lacks a recognized
    suffix one so it lands as a column instead of buried in `info`."""
    return name if name.endswith(_REWARD_COLUMN_SUFFIXES) else f"{name}_reward_func"


def trace_to_sample(trace: Trace, rollout_number: int = 1) -> dict[str, Any]:
    """One rollout -> the platform's sample dict (the "old" v0 eval-sample format).

    `rollout_number` is this rollout's 1-based index within its task's group (callers that
    don't group rollouts can leave it at the default)."""

    def dump(messages):
        return [m.model_dump(mode="json", exclude_none=True) for m in messages]

    task = trace.task.model_dump(mode="json", exclude_none=True)
    branches = trace.branches
    sample = {
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
        "info": dict(trace.info) or None,
    }
    # Flatten each sub-reward onto the sample as a top-level `*_reward_func`/`*_score` key so the
    # platform renders it as a per-rollout reward column. Env metrics stay in the nested `metrics`
    # field (they aren't reward columns — the platform keeps them inside `info`, as it does for v0).
    for name, value in trace.rewards.items():
        sample.setdefault(reward_column_name(name), value)
    return sample
