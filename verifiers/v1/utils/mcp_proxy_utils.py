from __future__ import annotations

import json
import os
import shlex
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import cast

MCP_PROXY_PATH = "/tmp/vf_mcp_tools.py"
MCP_PACKAGE = "mcp>=1.14.1"


def validate_tool_protocol(value: object) -> str:
    if value not in {"callable", "mcp"}:
        raise ValueError("tool_protocol must be 'callable' or 'mcp'.")
    return cast(str, value)


def proxy_program(program: Mapping[str, object]) -> dict[str, object]:
    files = dict(cast(Mapping[str, object], program.get("files") or {}))
    env = dict(cast(Mapping[str, object], program.get("env") or {}))
    if MCP_PROXY_PATH in files and files[MCP_PROXY_PATH] != proxy_source():
        raise ValueError(f"program.files cannot override {MCP_PROXY_PATH}.")
    files[MCP_PROXY_PATH] = proxy_source()
    env.update(proxy_env(MCP_PROXY_PATH, ["python3", MCP_PROXY_PATH]))
    return {**dict(program), "files": files, "env": env}


def proxy_sandbox(sandbox_config: Mapping[str, object]) -> dict[str, object]:
    config = dict(sandbox_config)
    packages = package_list(config.get("packages"))
    if not any(str(package).startswith("mcp") for package in packages):
        packages.append(MCP_PACKAGE)
    config["packages"] = packages
    return config


def local_proxy_program(
    program: Mapping[str, object],
) -> tuple[dict[str, object], Path]:
    fd, path_text = tempfile.mkstemp(prefix="vf_mcp_tools_", suffix=".py")
    path = Path(path_text)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(proxy_source())
    env = dict(cast(Mapping[str, object], program.get("env") or {}))
    env.update(proxy_env(str(path), [sys.executable, str(path)]))
    return {**dict(program), "env": env}, path


def proxy_env(path: str, command: list[str]) -> dict[str, str]:
    return {
        "VF_TOOL_PROTOCOL": "mcp",
        "VF_MCP_TOOL_PATH": path,
        "VF_MCP_TOOL_COMMAND_JSON": json.dumps(command),
        "VF_MCP_TOOL_COMMAND": shlex.join(command),
    }


def package_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, list):
        return [str(item) for item in value]
    raise TypeError("sandbox.packages must be a list or string.")


def proxy_source() -> str:
    return r"""
from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.parse
import urllib.request

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool


def tool_base_url() -> str:
    value = os.environ.get("VF_TOOL_BASE_URL")
    if not value:
        raise RuntimeError("VF_TOOL_BASE_URL is required.")
    return value.rstrip("/")


def auth_headers() -> dict[str, str]:
    token = os.environ.get("VF_TOOL_API_KEY") or os.environ.get("VF_ENDPOINT_API_KEY")
    headers = {"content-type": "application/json"}
    if token:
        headers["authorization"] = f"Bearer {token}"
    return headers


def get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers=auth_headers())
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(detail) from exc


def post_json(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers=auth_headers(),
    )
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(detail) from exc


def tool_text(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


server = Server("verifiers-tools")


@server.list_tools()
async def list_tools() -> list[Tool]:
    url = tool_base_url() + "?" + urllib.parse.urlencode({"protocol": "vf"})
    payload = await asyncio.to_thread(get_json, url)
    tools = []
    for item in payload.get("tools") or []:
        tools.append(
            Tool(
                name=str(item["name"]),
                description=str(item.get("description") or ""),
                inputSchema=item.get("parameters") or {"type": "object", "properties": {}},
            )
        )
    return tools


@server.call_tool(validate_input=False)
async def call_tool(name: str, arguments: dict) -> CallToolResult:
    payload = await asyncio.to_thread(
        post_json,
        f"{tool_base_url()}/{name}",
        {"arguments": arguments or {}},
    )
    if "error" in payload:
        return CallToolResult(
            content=[TextContent(type="text", text=str(payload["error"]))],
            isError=True,
        )
    result = payload.get("result")
    structured = result if isinstance(result, dict) else None
    return CallToolResult(
        content=[TextContent(type="text", text=tool_text(result))],
        structuredContent=structured,
        isError=False,
    )


async def main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
"""
