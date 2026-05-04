from __future__ import annotations

import json
import shlex
from collections.abc import Mapping
from typing import cast

MCP_PROXY_PATH = "/tmp/vf_mcp_tools.py"
MCP_PROXY_CONFIG_PATH = "/tmp/vf_mcp_tools.json"
MCP_PACKAGE = "mcp>=1.14.1"


def validate_program_tools(value: object) -> str | None:
    if value is None:
        return None
    if value != "mcp":
        raise ValueError("program.tools must be 'mcp'.")
    return cast(str, value)


def proxy_program(
    program: Mapping[str, object], tool_base_url: str, tool_api_key: str
) -> dict[str, object]:
    files = dict(cast(Mapping[str, object], program.get("files") or {}))
    if MCP_PROXY_PATH in files and files[MCP_PROXY_PATH] != proxy_source():
        raise ValueError(f"program.files cannot override {MCP_PROXY_PATH}.")
    config = {
        "tool_base_url": tool_base_url.rstrip("/"),
        "tool_api_key": tool_api_key,
    }
    config_json = json.dumps(config)
    if MCP_PROXY_CONFIG_PATH in files and files[MCP_PROXY_CONFIG_PATH] != config_json:
        raise ValueError(f"program.files cannot override {MCP_PROXY_CONFIG_PATH}.")
    files[MCP_PROXY_PATH] = proxy_source()
    files[MCP_PROXY_CONFIG_PATH] = config_json
    return {**dict(program), "files": files}


def proxy_command() -> list[str]:
    return ["python3", MCP_PROXY_PATH, MCP_PROXY_CONFIG_PATH]


def proxy_sandbox(sandbox_config: Mapping[str, object]) -> dict[str, object]:
    config = dict(sandbox_config)
    packages = package_list(config.get("packages"))
    if not any(str(package).startswith("mcp") for package in packages):
        packages.append(MCP_PACKAGE)
    config["packages"] = packages
    return config


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
import sys
import urllib.error
import urllib.parse
import urllib.request

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool

CONFIG = None


def config() -> dict:
    global CONFIG
    if CONFIG is None:
        if len(sys.argv) != 2:
            raise RuntimeError("MCP proxy requires a config path argument.")
        with open(sys.argv[1]) as f:
            CONFIG = json.load(f)
    return CONFIG


def tool_base_url() -> str:
    value = config().get("tool_base_url")
    if not value:
        raise RuntimeError("tool_base_url is required.")
    return str(value).rstrip("/")


def auth_headers() -> dict[str, str]:
    token = config().get("tool_api_key")
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
