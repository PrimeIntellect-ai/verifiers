import json
from typing import Any


def export_verifiers_env(env: Any, path: str) -> None:
    """Export a verifiers ToolEnv as an ART-compatible JSON config.

    Notes:
    - Only exports tool names/descriptions/parameters when available.
    - Implementations are not serialized; ART side should plug in real code.
    """
    # tools
    tools = []
    for tool in getattr(env, "tools", []) or []:
        schema = getattr(tool, "__art_schema__", None)
        if schema is None:
            # best-effort: name + empty schema
            schema = {
                "name": getattr(tool, "__name__", "tool"),
                "description": getattr(tool, "__doc__", "") or "",
                "parameters": {"type": "object", "properties": {}},
            }
        tools.append({
            "name": schema.get("name"),
            "description": schema.get("description", ""),
            "parameters": schema.get("parameters", {"type": "object", "properties": {}}),
            "implementation": None,
        })

    config = {
        "name": getattr(env, "env_id", "verifiers_env"),
        "tools": tools,
        "completion_tool_name": "submit_answer",
        "system_prompt": getattr(env, "system_prompt", None) or "Use tools to solve the task.",
    }
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


