from __future__ import annotations

import json
import shlex
from collections.abc import Mapping
from typing import Literal, cast

MCP_PROXY_PATH = "/tmp/vf_mcp_tools.py"
MCP_PROXY_CONFIG_PATH = "/tmp/vf_mcp_tools.json"
MCP_PACKAGE = "mcp>=1.14.1"
REQUESTS_PACKAGE = "requests"


ProgramToolType = Literal["callable", "mcp"]
PROGRAM_TOOL_TYPES = {"callable", "mcp"}
PROGRAM_TOOL_METADATA = {"priority"}


def validate_program_tool_types(value: object) -> tuple[ProgramToolType, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        if value not in PROGRAM_TOOL_TYPES:
            raise ValueError("program.tools must be 'callable' or 'mcp'.")
        return (cast(ProgramToolType, value),)
    if isinstance(value, list):
        result: list[ProgramToolType] = []
        for item in value:
            for tool_type in validate_program_tool_types(item):
                if tool_type in result:
                    raise ValueError(
                        f"program.tools defines {tool_type!r} more than once."
                    )
                result.append(tool_type)
        return tuple(result)
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("program.tools mapping keys must be strings.")
        spec = cast(Mapping[str, object], value)
        unknown = sorted(set(spec) - PROGRAM_TOOL_TYPES - PROGRAM_TOOL_METADATA)
        if unknown:
            raise ValueError(f"program.tools has unknown tool interface: {unknown}.")
        if "priority" in spec:
            priority = spec["priority"]
            if not isinstance(priority, int) or isinstance(priority, bool):
                raise TypeError("program.tools priority must be an integer.")
        result = [
            cast(ProgramToolType, key) for key in spec if key in PROGRAM_TOOL_TYPES
        ]
        if not result:
            raise ValueError("program.tools mapping must define a tool interface.")
        return tuple(result)
    raise TypeError("program.tools must be a string, mapping, or list.")


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
    if not any(str(package).startswith("requests") for package in packages):
        packages.append(REQUESTS_PACKAGE)
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

import requests

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
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "python-requests/2.32.3",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def get_json(url: str, params: dict | None = None) -> dict:
    try:
        response = requests.get(
            url,
            params=params,
            headers=auth_headers(),
            timeout=300,
        )
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(response.text) from exc
    except requests.RequestException as exc:
        raise RuntimeError(str(exc)) from exc
    return response.json()


def post_json(url: str, payload: dict) -> dict:
    try:
        response = requests.post(
            url,
            json=payload,
            headers=auth_headers(),
            timeout=300,
        )
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(response.text) from exc
    except requests.RequestException as exc:
        raise RuntimeError(str(exc)) from exc
    return response.json()


def tool_text(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


server = Server("verifiers-tools")


@server.list_tools()
async def list_tools() -> list[Tool]:
    payload = await asyncio.to_thread(get_json, tool_base_url(), {"protocol": "vf"})
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
