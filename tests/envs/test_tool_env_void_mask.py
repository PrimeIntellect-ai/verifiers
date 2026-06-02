import json

import pytest

import verifiers as vf
from verifiers.envs.tool_env import ToolEnv


def constant_reward(**kwargs) -> float:
    return 2.0


def good_tool() -> str:
    return "ok"


def bad_tool() -> str:
    raise RuntimeError("boom")


def make_env(*, mask: bool) -> ToolEnv:
    return ToolEnv(
        tools=[good_tool, bad_tool],
        mask_all_failed_tool_calls=mask,
        rubric=vf.Rubric(funcs=[constant_reward]),
    )


def make_tool_call(name: str, index: int) -> vf.ToolCall:
    return vf.ToolCall(id=f"call_{index}", name=name, arguments=json.dumps({}))


async def run_tool_calls(env: ToolEnv, tool_names: list[str]) -> vf.State:
    tool_calls = [make_tool_call(name, index) for index, name in enumerate(tool_names)]
    assistant = vf.AssistantMessage(content=None, tool_calls=tool_calls)
    state = vf.State(
        prompt=[vf.UserMessage(content="use tools")],
        completion=[],
        trajectory=[],
    )
    tool_messages = await env.env_response([assistant], state)
    state["completion"] = [assistant, *tool_messages]
    env._apply_tool_call_mask(state)
    return state


def run_assistant_only_no_tools(env: ToolEnv) -> vf.State:
    assistant = vf.AssistantMessage(content="done")
    state = vf.State(
        prompt=[vf.UserMessage(content="answer")],
        completion=[assistant],
        trajectory=[],
    )
    env._apply_tool_call_mask(state)
    return state


@pytest.mark.asyncio
async def test_flag_off_records_outcomes_without_masking():
    env = make_env(mask=False)

    state = await run_tool_calls(env, ["bad_tool", "bad_tool"])
    await env.rubric.score_rollout(state)

    assert state.get("masked") in (None, False)
    assert state["tool_call_outcomes"] == ["error", "error"]
    assert state["reward"] == 2.0


@pytest.mark.asyncio
async def test_flag_on_all_errors_masked_and_scores_zero():
    env = make_env(mask=True)

    state = await run_tool_calls(env, ["bad_tool", "bad_tool"])
    await env.rubric.score_rollout(state)

    assert state["masked"] is True
    assert state["tool_call_outcomes"] == ["error", "error"]
    assert state["reward"] == 0.0
    assert state["metrics"]["constant_reward"] == 0.0
    assert state["metrics"]["void_turn_rollouts"] == 1.0


@pytest.mark.asyncio
async def test_flag_on_mixed_outcomes_unmasked():
    env = make_env(mask=True)

    state = await run_tool_calls(env, ["good_tool", "bad_tool"])

    assert state["masked"] is False
    assert state["tool_call_outcomes"] == ["ok", "error"]


def test_flag_on_no_tool_calls_unmasked():
    env = make_env(mask=True)

    state = run_assistant_only_no_tools(env)

    assert state["masked"] is False
    assert state.get("tool_call_outcomes") in (None, [])


@pytest.mark.asyncio
async def test_stateful_tool_env_tracks_outcomes_for_masking():
    class ExampleStatefulToolEnv(vf.StatefulToolEnv):
        def update_tool_args(
            self,
            tool_name: str,
            tool_args: dict,
            messages: vf.Messages,
            state: vf.State,
            **kwargs,
        ) -> dict:
            return tool_args

    env = ExampleStatefulToolEnv(
        tools=[good_tool, bad_tool],
        mask_all_failed_tool_calls=True,
        rubric=vf.Rubric(funcs=[constant_reward]),
    )

    state = await run_tool_calls(env, ["bad_tool"])

    assert state["masked"] is True
    assert state["tool_call_outcomes"] == ["error"]
