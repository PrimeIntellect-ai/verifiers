from __future__ import annotations

import time
import uuid
from typing import Any

from verifiers.types import (
    AssistantMessage,
    Messages,
    Response,
    ResponseMessage,
    ResponseTokens,
    State,
    ToolCall,
    ToolMessage,
    TrajectoryStep,
)


def _resolve_gym_config(resources_server: str) -> str:
    from verifiers.envs.integrations.nemo_gym.utils import _resolve_resources_servers_root
    root = _resolve_resources_servers_root()
    path = root / resources_server / "configs" / f"{resources_server}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find NeMo Gym config for '{resources_server}': {path}"
        )
    return str(path)


def _reward_from_nemo(state: State, **kwargs: Any) -> float:
    return float(state.get("nemo_reward", 0.0) or 0.0)


def _nemo_item_to_assistant_message(item: dict[str, Any]) -> AssistantMessage:
    item_type = item.get("type")

    if item_type == "message":
        content_blocks = item.get("content") or []
        text = "\n".join(
            c.get("text", "")
            for c in content_blocks
            if isinstance(c, dict) and c.get("type") == "output_text"
        )
        return AssistantMessage(role="assistant", content=text or None)

    if item_type == "function_call":
        tool_call = ToolCall(
            id=str(item.get("call_id") or item.get("id") or uuid.uuid4().hex[:8]),
            name=str(item.get("name", "")),
            arguments=str(item.get("arguments", "{}")),
        )
        return AssistantMessage(role="assistant", content=None, tool_calls=[tool_call])

    return AssistantMessage(role="assistant", content=str(item))


def _make_synthetic_response(
    msg: AssistantMessage,
    model: str,
    gen_ids: list[int],
    logprobs: list[float],
    prompt_ids: list[int],
) -> Response:
    tokens = ResponseTokens(
        prompt_ids=prompt_ids,
        prompt_mask=[1] * len(prompt_ids),
        completion_ids=gen_ids,
        completion_mask=[1] * len(gen_ids),
        completion_logprobs=logprobs,
        routed_experts=None,
    )
    return Response(
        id=f"nemo-agent-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=model,
        usage=None,
        message=ResponseMessage(
            role="assistant",
            content=msg.content,
            tool_calls=msg.tool_calls,
            finish_reason="tool_calls" if msg.tool_calls else "stop",
            is_truncated=False,
            tokens=tokens,
        ),
    )


def _build_trajectory_from_nemo(
    output_items: list[dict[str, Any]],
    initial_prompt: Messages,
    model: str,
    trajectory_id: str,
) -> tuple[list[TrajectoryStep], Messages]:
    # Items with generation_token_ids are assistant turns; function_call_output
    # items are env responses. Tool-response tokens live in the next step's
    # prompt_ids only and are never trained on.
    trajectory: list[TrajectoryStep] = []
    completion_messages: list = []
    all_messages: list = list(initial_prompt)

    for item in output_items:
        if item.get("type") == "function_call_output":
            tool_msg = ToolMessage(
                role="tool",
                tool_call_id=str(item.get("call_id", "")),
                content=str(item.get("output", "")),
            )
            all_messages.append(tool_msg)
            completion_messages.append(tool_msg)
            continue

        if "generation_token_ids" not in item:
            continue

        prompt_ids: list[int] = list(item.get("prompt_token_ids") or [])
        gen_ids: list[int] = list(item.get("generation_token_ids") or [])
        logprobs: list[float] = list(
            item.get("generation_log_probs") or [0.0] * len(gen_ids)
        )

        step_prompt: Messages = list(all_messages)
        assistant_msg = _nemo_item_to_assistant_message(item)
        all_messages.append(assistant_msg)
        completion_messages.append(assistant_msg)

        trajectory.append({
            "prompt": step_prompt,
            "completion": [assistant_msg],
            "response": _make_synthetic_response(
                assistant_msg, model, gen_ids, logprobs, prompt_ids
            ),
            "tokens": {
                "prompt_ids": prompt_ids,
                "prompt_mask": [1] * len(prompt_ids),
                "completion_ids": gen_ids,
                "completion_mask": [1] * len(gen_ids),
                "completion_logprobs": logprobs,
                "overlong_prompt": False,
                "is_truncated": False,
                "routed_experts": None,
            },
            "reward": None,
            "advantage": None,
            "is_truncated": False,
            "trajectory_id": trajectory_id,
            "extras": {},
        })

    return trajectory, completion_messages


def _map_nemo_result_to_state(state: State, nemo_result: Any, model: str) -> None:
    import verifiers as vf

    if not isinstance(nemo_result, dict) or nemo_result.get("error"):
        error_detail = (
            nemo_result.get("error", "unknown error")
            if isinstance(nemo_result, dict)
            else repr(nemo_result)
        )
        state["error"] = vf.InfraError(
            f"NeMo Gym agent server rollout failed: {error_detail}"
        )
        state["nemo_reward"] = 0.0
        state["completion"] = []
        return

    state["nemo_reward"] = float(nemo_result.get("reward", 0.0) or 0.0)
    state["nemo_result"] = nemo_result

    output_items: list[dict[str, Any]] = (
        nemo_result.get("response") or {}
    ).get("output") or []

    trajectory, completion_messages = _build_trajectory_from_nemo(
        output_items=output_items,
        initial_prompt=state["prompt"],
        model=model,
        trajectory_id=state["trajectory_id"],
    )

    state["trajectory"] = trajectory
    state["completion"] = completion_messages
    state["is_truncated"] = False
