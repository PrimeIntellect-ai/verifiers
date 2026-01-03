from unittest.mock import MagicMock

import pytest

from verifiers.utils.rlm_metrics_utils import add_metrics_to_state, get_rlm_rubrics


def _make_response(prompt_tokens: int, completion_tokens: int):
    response = MagicMock()
    response.usage = MagicMock(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )
    return response


def _make_step(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    depth: int,
    batch_id: str,
    request_id: str,
    tool_calls: int,
):
    return {
        "prompt": [{"role": "user", "content": "x"}],
        "completion": [{"role": "assistant", "content": "y"}],
        "response": _make_response(prompt_tokens, completion_tokens),
        "tokens": None,
        "reward": None,
        "advantage": None,
        "is_truncated": False,
        "trajectory_id": "sub",
        "extras": {
            "is_sub_llm_call": True,
            "sub_llm_depth": depth,
            "batch_id": batch_id,
            "request_id": request_id,
            "tool_call_count": tool_calls,
        },
    }


@pytest.mark.asyncio
async def test_add_metrics_to_state_with_sub_llm_steps():
    sub_steps = [
        _make_step(
            prompt_tokens=10,
            completion_tokens=5,
            depth=1,
            batch_id="b1",
            request_id="r1",
            tool_calls=2,
        ),
        _make_step(
            prompt_tokens=20,
            completion_tokens=10,
            depth=1,
            batch_id="b1",
            request_id="r1",
            tool_calls=0,
        ),
        _make_step(
            prompt_tokens=5,
            completion_tokens=5,
            depth=2,
            batch_id="b2",
            request_id="r2",
            tool_calls=1,
        ),
    ]
    state = {
        "trajectory": [
            {
                "prompt": [{"role": "user", "content": "main"}],
                "completion": [{"role": "assistant", "content": "main"}],
                "response": _make_response(7, 3),
                "tokens": None,
                "reward": None,
                "advantage": None,
                "is_truncated": False,
                "trajectory_id": "main",
                "extras": {},
            }
        ]
    }

    await add_metrics_to_state(state, sub_llm_steps=sub_steps)

    assert state["sub_llm_call_count"] == 2
    assert state["sub_llm_total_turns"] == 3
    assert state["sub_llm_prompt_tokens"] == 35
    assert state["sub_llm_completion_tokens"] == 20
    assert state["sub_llm_total_tool_calls"] == 3

    assert state["sub_llm_depth_max"] == 2
    assert state["sub_llm_depth_mean"] == 1.5
    assert state["sub_llm_depth_gt1_frac"] == 0.5

    assert state["sub_llm_prompt_tokens_per_call"] == 17.5
    assert state["sub_llm_completion_tokens_per_call"] == 10.0
    assert state["sub_llm_tool_calls_per_call"] == 1.5
    assert state["sub_llm_turns_per_call"] == 1.5

    assert state["main_rlm_turns"] == 1
    assert state["main_rlm_prompt_tokens"] == 7
    assert state["main_rlm_completion_tokens"] == 3


@pytest.mark.asyncio
async def test_add_metrics_to_state_empty():
    state = {"trajectory": []}
    await add_metrics_to_state(state)

    assert state["sub_llm_call_count"] == 0
    assert state["sub_llm_total_turns"] == 0
    assert state["sub_llm_prompt_tokens"] == 0
    assert state["sub_llm_completion_tokens"] == 0
    assert state["sub_llm_depth_max"] == 0
    assert state["sub_llm_depth_mean"] == 0.0
    assert state["sub_llm_depth_gt1_frac"] == 0.0
    assert state["main_rlm_turns"] == 0
    assert state["repl_call_count"] == 0


def test_get_rlm_rubrics_appends_metrics():
    def base_metric(*, state, **kwargs):
        return 0.0

    funcs, weights = get_rlm_rubrics(base=[base_metric], base_weights=[1.0], weight=0.0)

    names = {func.__name__ for func in funcs}
    assert "sub_llm_call_count" in names
    assert "main_rlm_turns" in names
    assert "repl_mean_time_seconds" in names
    assert base_metric.__name__ in names
    assert len(funcs) == len(weights)
