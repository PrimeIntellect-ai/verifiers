"""Fetch MCP server and CLI utility.

This module serves two purposes:
- Local development CLI (`python fetch_mcp_server.py --url ...`) that prints a JSON
  payload describing the fetch result.
- MCP-compliant stdio server (`python fetch_mcp_server.py --run-server`) which
  exposes a `fetch` tool returning both structured JSON and text content.

The structured payload includes: status, headers, body_text/body_json, hash of the
truncated body, final URL, and bookkeeping flags (`bytes`, `truncated`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple
from urllib.parse import urlparse

import anyio
import httpx
import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

DEFAULT_TIMEOUT_S = 8.0
DEFAULT_MAX_BYTES = 200_000


def _normalize_hosts(hosts: Iterable[str]) -> list[str]:
    seen: list[str] = []
    for host in hosts:
        cleaned = host.strip().lower()
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return seen


def _collect_allowed_hosts(cli_hosts: Optional[Sequence[str]] = None) -> list[str]:
    hosts: list[str] = []
    env_hosts = os.getenv("MCP_FETCH_ALLOWED_HOSTS")
    if env_hosts:
        hosts.extend([h for h in env_hosts.split(",") if h.strip()])
    if cli_hosts:
        hosts.extend(cli_hosts)
    return _normalize_hosts(hosts)


def _parse_bool_env(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _ensure_allowed_host(url: str, allowed_hosts: Sequence[str], allow_any: bool) -> None:
    if allow_any or not allowed_hosts:
        return

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http:// or https:// URLs are permitted")
    if not parsed.hostname:
        raise ValueError("URL must include a hostname")

    host = parsed.hostname.lower()
    port = parsed.port
    host_key = f"{host}:{port}" if port else host

    candidates = {host_key, host}
    if not any(candidate in allowed_hosts for candidate in candidates):
        raise ValueError(f"Host '{host_key}' not in allowed list")


def _parse_kv_pairs(pairs: Sequence[str], what: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for pair in pairs:
        if ":" in pair:
            key, value = pair.split(":", 1)
        elif "=" in pair:
            key, value = pair.split("=", 1)
        else:
            raise ValueError(f"Invalid {what} format '{pair}'. Use key:value or key=value.")
        parsed[key.strip()] = value.strip()
    return parsed


async def fetch_url_async(
    url: str,
    *,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, str]] = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Dict[str, Any]:
    headers = {str(k): str(v) for k, v in (headers or {}).items()}
    params_dict = None
    if params:
        # Only pass params to httpx when the caller provided explicit overrides;
        # passing an empty dict would strip query strings already present in the URL.
        params_dict = {str(k): str(v) for k, v in params.items()}

    method = method.upper()
    if method not in {"GET", "HEAD"}:
        raise ValueError("Only GET and HEAD are supported")

    if max_bytes <= 0:
        raise ValueError("max_bytes must be > 0")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be > 0")

    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout_s) as client:
        if method == "HEAD":
            response = await client.head(url, headers=headers, params=params_dict)
            raw_body = b""
        else:
            response = await client.get(url, headers=headers, params=params_dict)
            raw_body = response.content

    body_bytes = raw_body[:max_bytes]
    content_type = response.headers.get("Content-Type", "")

    body_text: Optional[str] = None
    body_json: Optional[Any] = None
    if method != "HEAD" and body_bytes:
        if "json" in content_type.lower():
            try:
                body_json = json.loads(body_bytes.decode("utf-8", errors="replace"))
            except Exception:
                body_text = body_bytes.decode("utf-8", errors="replace")
        else:
            body_text = body_bytes.decode("utf-8", errors="replace")

    structured: Dict[str, Any] = {
        "status": response.status_code,
        "headers": dict(response.headers.items()),
        "body_text": body_text,
        "body_json": body_json,
        "hash": hashlib.sha256(body_bytes).hexdigest(),
        "final_url": str(response.url),
        "bytes": len(body_bytes),
        "truncated": len(body_bytes) < len(raw_body),
        "method": method,
        "content_type": content_type,
    }

    return structured


def _build_tool_schema() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    input_schema = {
        "type": "object",
        "required": ["url"],
        "properties": {
            "url": {"type": "string", "description": "URL to fetch"},
            "method": {
                "type": "string",
                "enum": ["GET", "HEAD", "get", "head"],
                "default": "GET",
                "description": "HTTP method (GET or HEAD)",
            },
            "headers": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": "Optional HTTP headers",
            },
            "params": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": "Optional query parameters",
            },
            "timeout_s": {
                "type": "number",
                "minimum": 0.1,
                "maximum": 60.0,
                "default": DEFAULT_TIMEOUT_S,
                "description": "Request timeout in seconds",
            },
            "max_bytes": {
                "type": "integer",
                "minimum": 1,
                "maximum": 1_000_000,
                "default": DEFAULT_MAX_BYTES,
                "description": "Maximum bytes to read from the response body",
            },
        },
    }

    output_schema = {
        "type": "object",
        "required": ["status", "headers", "hash", "final_url", "bytes", "truncated"],
        "properties": {
            "status": {"type": "integer"},
            "headers": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "body_text": {"type": ["string", "null"]},
            "body_json": {"type": ["object", "array", "null"]},
            "hash": {"type": "string"},
            "final_url": {"type": "string"},
            "bytes": {"type": "integer"},
            "truncated": {"type": "boolean"},
            "method": {"type": "string"},
            "content_type": {"type": ["string", "null"]},
        },
    }

    return input_schema, output_schema


def create_server(
    *,
    allowed_hosts: Sequence[str],
    allow_any_host: bool,
) -> Server:
    server = Server("mcp-fetch")
    input_schema, output_schema = _build_tool_schema()

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="fetch",
                title="HTTP Fetch",
                description="Fetches a URL using GET or HEAD and returns metadata plus body.",
                inputSchema=input_schema,
                outputSchema=output_schema,
            )
        ]

    @server.call_tool()
    async def call_tool(
        name: str, arguments: dict[str, Any]
    ) -> Tuple[list[types.ContentBlock], Dict[str, Any]]:
        if name != "fetch":
            raise ValueError(f"Unknown tool '{name}'")

        url = arguments.get("url")
        if not isinstance(url, str) or not url:
            raise ValueError("Argument 'url' must be a non-empty string")

        method = str(arguments.get("method", "GET"))
        headers = arguments.get("headers") or {}
        params = arguments.get("params") or {}
        timeout_s = float(arguments.get("timeout_s", DEFAULT_TIMEOUT_S))
        max_bytes = int(arguments.get("max_bytes", DEFAULT_MAX_BYTES))

        _ensure_allowed_host(url, allowed_hosts, allow_any_host)

        structured = await fetch_url_async(
            url,
            method=method,
            headers=headers,
            params=params,
            timeout_s=timeout_s,
            max_bytes=max_bytes,
        )
        text = json.dumps(structured, indent=2, ensure_ascii=False, sort_keys=True)
        return [types.TextContent(type="text", text=text)], structured

    return server


def run_server(*, allowed_hosts: Sequence[str], allow_any_host: bool) -> int:
    server = create_server(
        allowed_hosts=allowed_hosts,
        allow_any_host=allow_any_host,
    )

    async def _run() -> None:
        async with stdio_server() as streams:
            await server.run(streams[0], streams[1], server.create_initialization_options())

    anyio.run(_run)
    return 0


def run_cli(args: argparse.Namespace, allowed_hosts: Sequence[str], allow_any_host: bool) -> int:
    if not args.url:
        print("Missing required argument --url", file=sys.stderr)
        return 2

    headers = _parse_kv_pairs(args.header or [], "header")
    params = _parse_kv_pairs(args.param or [], "param")

    try:
        _ensure_allowed_host(args.url, allowed_hosts, allow_any_host)
        result = anyio.run(
            fetch_url_async,
            args.url,
            method=args.method,
            headers=headers,
            params=params,
            timeout_s=args.timeout_s,
            max_bytes=args.max_bytes,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-server", action="store_true", help="Run as an MCP stdio server")
    parser.add_argument("--transport", choices=["stdio"], default="stdio", help="Transport when running the server")
    parser.add_argument("--allowed-host", dest="allowed_hosts", action="append", help="Allowlist host (repeatable)")
    parser.add_argument("--allow-any-host", action="store_true", help="Disable host allowlist checks")

    parser.add_argument("--url", help="URL to fetch (CLI mode)")
    parser.add_argument("--method", default="GET", help="HTTP method (CLI mode)")
    parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S, help="Timeout in seconds (CLI mode)")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES, help="Maximum body bytes to retain")
    parser.add_argument("--header", action="append", help="Custom header key:value (CLI mode)")
    parser.add_argument("--param", action="append", help="Query parameter key:value (CLI mode)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.transport != "stdio":
        raise SystemExit("Only stdio transport is supported in this server build")

    allowed_hosts = _collect_allowed_hosts(args.allowed_hosts)
    allow_any_host = args.allow_any_host or _parse_bool_env("MCP_FETCH_ALLOW_ANY_HOST")
    if allow_any_host:
        allowed_hosts = []

    if args.run_server:
        return run_server(allowed_hosts=allowed_hosts, allow_any_host=allow_any_host)
    return run_cli(args, allowed_hosts=allowed_hosts, allow_any_host=allow_any_host)


if __name__ == "__main__":
    raise SystemExit(main())
