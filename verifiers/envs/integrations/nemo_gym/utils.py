from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from typing import Any, cast

from datasets import Dataset

from verifiers.types import AssistantMessage, Messages, State, ToolMessage


_ALLOWED_DATASET_SPLITS = {"example", "train", "validation"}


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


def _sanitize_json_schema(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, raw_child in value.items():
            if raw_child is None:
                continue
            child = _sanitize_json_schema(raw_child)
            if child is None:
                continue
            sanitized[key] = child

        properties = sanitized.get("properties")
        if isinstance(properties, dict):
            sanitized["properties"] = {
                name: schema
                for name, schema in properties.items()
                if isinstance(schema, (dict, bool))
            }
            required = sanitized.get("required")
            if isinstance(required, list):
                allowed = set(sanitized["properties"].keys())
                sanitized["required"] = [
                    name
                    for name in required
                    if isinstance(name, str) and name in allowed
                ]

        return sanitized

    if isinstance(value, list):
        return [
            child
            for item in value
            if (child := _sanitize_json_schema(item)) is not None
        ]

    return value


def _normalize_parameters_schema(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {"type": "object", "properties": {}}
    return _sanitize_json_schema(value)


def _nemo_tools_to_tool_defs(raw_tools: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_tools, list):
        return []

    tool_defs: list[dict[str, Any]] = []
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, dict):
            continue

        # OpenAI Chat Completions-style tool schema.
        if raw_tool.get("type") == "function" and isinstance(
            raw_tool.get("function"), dict
        ):
            fn = cast(dict[str, Any], raw_tool["function"])
            name = fn.get("name")
            if not isinstance(name, str) or not name:
                continue
            tool_def: dict[str, Any] = {
                "name": name,
                "description": _stringify(fn.get("description", "")),
                "parameters": _normalize_parameters_schema(fn.get("parameters")),
            }
            strict = fn.get("strict", raw_tool.get("strict"))
            if isinstance(strict, bool):
                tool_def["strict"] = strict
            tool_defs.append(tool_def)
            continue

        # OpenAI Responses API function tool schema.
        tool_type = raw_tool.get("type")
        if tool_type not in (None, "function"):
            continue

        name = raw_tool.get("name")
        if not isinstance(name, str) or not name:
            continue

        tool_def = {
            "name": name,
            "description": _stringify(raw_tool.get("description", "")),
            "parameters": _normalize_parameters_schema(raw_tool.get("parameters")),
        }
        strict = raw_tool.get("strict")
        if isinstance(strict, bool):
            tool_def["strict"] = strict
        tool_defs.append(tool_def)

    return tool_defs


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


def _resolve_dataset_path(
    resource_server: str,
    dataset_split: str,
    dataset_path: str | None,
) -> Path:
    if dataset_path is not None:
        path = Path(dataset_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"dataset_path does not exist: {path}")
        return path

    resources_root = _resolve_resources_servers_root()
    path = resources_root / resource_server / "data" / f"{dataset_split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            "Could not find dataset file for server "
            f"'{resource_server}' split '{dataset_split}': {path}"
        )
    return path


def _load_rows_from_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} line {line_no}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"Row {line_no} in {path} is not an object")
            if "responses_create_params" not in row:
                raise ValueError(
                    f"Row {line_no} in {path} is missing required key "
                    "'responses_create_params'"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"Dataset file {path} contains no rows")
    return rows


def _build_dataset(
    resource_server: str,
    dataset_split: str,
    dataset_path: str | None = None,
    dataset_limit: int | None = None,
) -> tuple[Dataset, Path]:
    resolved_path = _resolve_dataset_path(resource_server, dataset_split, dataset_path)
    rows = _load_rows_from_jsonl(resolved_path)

    if dataset_limit is not None:
        if dataset_limit <= 0:
            raise ValueError("dataset_limit must be > 0 when provided")
        rows = rows[:dataset_limit]

    dataset_rows: list[dict[str, Any]] = []
    for row in rows:
        responses_create_params = row.get("responses_create_params")
        if not isinstance(responses_create_params, dict):
            raise ValueError("responses_create_params must be an object")

        raw_input = responses_create_params.get("input", [])
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
                "task": resource_server,
                "info": {
                    "dataset_row_json": _json_dumps(row),
                    "resource_server": resource_server,
                },
            }
        )

    return Dataset.from_list(dataset_rows), resolved_path


def _completion_to_nemo_response(
    completion: Messages,
    model_name: str,
    trajectory_id: str,
    responses_create_params: dict[str, Any],
) -> dict[str, Any]:
    output: list[dict[str, Any]] = []
    message_idx = 0

    for msg in completion:
        if isinstance(msg, AssistantMessage):
            text = msg.content or ""
            if isinstance(text, list):
                text = "\n".join(getattr(p, "text", str(p)) for p in text)
            if text:
                output.append(
                    {
                        "id": f"msg_{message_idx}",
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": text, "annotations": []}
                        ],
                    }
                )
                message_idx += 1

            for tc in msg.tool_calls or []:
                output.append(
                    {
                        "id": tc.id,
                        "type": "function_call",
                        "call_id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }
                )
                message_idx += 1

        elif isinstance(msg, ToolMessage):
            content = msg.content
            if isinstance(content, list):
                content = "\n".join(getattr(p, "text", str(p)) for p in content)
            output.append(
                {
                    "type": "function_call_output",
                    "call_id": msg.tool_call_id,
                    "output": content or "",
                }
            )

    return {
        "id": f"verifiers-{trajectory_id}",
        "created_at": int(time.time()),
        "model": model_name,
        "object": "response",
        "output": output,
        "parallel_tool_calls": bool(
            responses_create_params.get("parallel_tool_calls", False)
        ),
        "tool_choice": responses_create_params.get("tool_choice", "none"),
        "tools": responses_create_params.get("tools", []),
    }


def _reward_from_verify(state: State, **kwargs: Any) -> float:
    verify_response = state.get("verify_response")
    if not isinstance(verify_response, dict):
        return 0.0
    return float(verify_response.get("reward", 0.0) or 0.0)
