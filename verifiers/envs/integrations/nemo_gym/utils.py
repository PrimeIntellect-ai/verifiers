from __future__ import annotations

import importlib.util
import json
import time
import uuid
from pathlib import Path
from typing import Any

from datasets import Dataset

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

def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return _json_dumps(value)
    except (TypeError, ValueError):
        return str(value)


def _resolve_resources_servers_root() -> Path:
    resources_spec = importlib.util.find_spec("resources_servers")
    if resources_spec and resources_spec.submodule_search_locations:
        root = Path(next(iter(resources_spec.submodule_search_locations))).resolve()
        if root.exists():
            return root

    nemo_spec = importlib.util.find_spec("nemo_gym")
    if nemo_spec and nemo_spec.origin:
        nemo_root = Path(nemo_spec.origin).resolve().parent
        sibling = nemo_root.parent / "resources_servers"
        if sibling.exists():
            return sibling

    raise RuntimeError(
        "Unable to locate NeMo Gym resources_servers package. "
        "Install `nemo-gym` or pass `dataset_path` explicitly."
    )


def _build_dataset(
    resources_server: str,
    dataset_split: str,
    dataset_path: str | None = None,
    dataset_limit: int | None = None,
) -> tuple[Dataset, Path]:
    if dataset_path is not None:
        path = Path(dataset_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"dataset_path does not exist: {path}")
    else:
        root = _resolve_resources_servers_root()
        path = root / resources_server / "data" / f"{dataset_split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"Could not find dataset for '{resources_server}' split '{dataset_split}': {path}"
            )

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Row {line_no} in {path} is not an object")
            if "responses_create_params" not in row:
                raise ValueError(
                    f"Row {line_no} in {path} is missing 'responses_create_params'"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"Dataset file {path} contains no rows")

    if dataset_limit is not None:
        if dataset_limit <= 0:
            raise ValueError("dataset_limit must be > 0")
        rows = rows[:dataset_limit]

    dataset_rows: list[dict[str, Any]] = []
    for row in rows:
        rcp = row["responses_create_params"]
        raw_input = rcp.get("input", [])
        if isinstance(raw_input, str):
            prompt = [{"role": "user", "content": raw_input}]
        elif isinstance(raw_input, list):
            prompt = raw_input
        else:
            prompt = [{"role": "user", "content": _stringify(raw_input)}]
        dataset_rows.append(
            {
                "prompt": prompt,
                "answer": _stringify(row.get("answer", "")),
                "task": resources_server,
                "info": {"dataset_row_json": _json_dumps(row)},
            }
        )

    return Dataset.from_list(dataset_rows), path


def _resolve_gym_config(resources_server: str) -> str:
    root = _resolve_resources_servers_root()
    path = root / resources_server / "configs" / f"{resources_server}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find NeMo Gym config for '{resources_server}': {path}"
        )
    return str(path)

# this may silenty break things in multi-env runs if agent_ref is not set in the dataset! 
# TODO: should discuss removing it, or at least documenting it
def _resolve_agent_name(gym_config_path: str) -> str:
    import yaml
    with open(gym_config_path) as f:
        config = yaml.safe_load(f)
    for key, value in config.items():
        if isinstance(value, dict) and "responses_api_agents" in value:
            return key
    raise RuntimeError(
        f"Could not find a responses_api_agents entry in {gym_config_path}"
    )


def _reward_from_nemo_gym(state: State, **kwargs: Any) -> float:
    return float(state.get("nemo_gym_reward", 0.0) or 0.0)


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


def _make_response(
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
        id=f"nemo_gym-{uuid.uuid4().hex[:8]}",
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
            "response": _make_response(
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


def _map_nemo_gym_result_to_state(state: State, nemo_gym_result: Any, model: str) -> None:
    import verifiers as vf

    if not isinstance(nemo_gym_result, dict) or nemo_gym_result.get("error"):
        error_detail = (
            nemo_gym_result.get("error", "unknown error")
            if isinstance(nemo_gym_result, dict)
            else repr(nemo_gym_result)
        )
        state["error"] = vf.InfraError(
            f"NeMo Gym agent server rollout failed: {error_detail}"
        )
        state["nemo_gym_reward"] = 0.0
        state["completion"] = []
        return

    state["nemo_gym_reward"] = float(nemo_gym_result.get("reward", 0.0) or 0.0)
    state["nemo_gym_result"] = nemo_gym_result

    output_items: list[dict[str, Any]] = (
        nemo_gym_result.get("response") or {}
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
