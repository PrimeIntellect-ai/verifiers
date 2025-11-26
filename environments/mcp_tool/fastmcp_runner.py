import argparse
import asyncio
import json
from typing import Any, Dict, List
from fastmcp import Client


def _json_print(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False))


def _join_text(content_blocks: Any) -> str:
    parts: List[str] = []
    try:
        for c in (content_blocks or []):
            if hasattr(c, "text") and c.text is not None:
                parts.append(str(c.text))
            elif isinstance(c, dict) and "text" in c:
                parts.append(str(c["text"]))
    except Exception:
        pass
    return "\n".join([p for p in parts if p is not None])


def _schema_to_dict(schema: Any) -> Dict[str, Any]:
    if schema is None:
        return {}
    if isinstance(schema, dict):
        return schema
    if hasattr(schema, "model_dump"):
        try:
            return schema.model_dump()  # pydantic v2
        except Exception:
            pass
    if hasattr(schema, "dict"):
        try:
            return schema.dict()  # pydantic v1
        except Exception:
            pass
    try:
        return json.loads(json.dumps(schema, default=lambda o: getattr(o, "__dict__", str(o))))
    except Exception:
        return {}


def _build_client(server: Dict[str, Any]) -> Client:
    """
    Build a FastMCP Client from a server payload:
      - {"type":"script", "path": "/path/to/server.py", "env":{}, "cwd": "..."}
      - {"type":"http", "url": "https://...", "headers": {...}}
      - {"type":"stdio", "command": "python", "args": ["./server.py", "--flag"], "env":{}, "cwd": "..."}
    """
    stype = server.get("type")
    if stype == "script":
        path = server["path"]
        # Always use explicit config for scripts to avoid transport inference issues
        # Use venv Python to run scripts with FastMCP available
        cfg = {
            "mcpServers": {
                "server": {
                    "transport": "stdio",
                    "command": "/opt/mcp-venv/bin/python",
                    "args": [path],
                    "env": server.get("env", {}),
                    "cwd": server.get("cwd"),
                }
            }
        }
        client = Client(cfg)
        return client

    if stype == "http":
        url = server["url"]
        # Client(url) infers HTTP/SSE transport; headers supported via config
        if server.get("headers"):
            cfg = {
                "mcpServers": {
                    "server": {
                        "transport": "http",
                        "url": url,
                        "headers": server.get("headers", {}),
                    }
                }
            }
            return Client(cfg)
        return Client(url)

    if stype == "stdio":
        command = server["command"]
        args = server.get("args", [])
        cfg = {
            "mcpServers": {
                "server": {
                    "transport": "stdio",
                    "command": command,
                    "args": args,
                    "env": server.get("env", {}),
                    "cwd": server.get("cwd"),
                }
            }
        }
        return Client(cfg)

    raise ValueError(f"Unknown server type: {stype}")


async def _list_tools(server: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    client = _build_client(server)
    async with client:
        await client.ping()
        tools = await asyncio.wait_for(client.list_tools(), timeout=timeout_s)
        serial = []
        for t in tools:
            entry = {
                "name": t.name,
                "description": t.description or f"MCP tool: {t.name}",
                "inputSchema": _schema_to_dict(getattr(t, "inputSchema", None)),
            }
            meta = getattr(t, "meta", None)
            if meta:
                entry["meta"] = meta
            serial.append(entry)
        return {"ok": True, "tools": serial}


async def _call_tool(server: Dict[str, Any], tool_name: str, arguments: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    client = _build_client(server)
    async with client:
        await client.ping()
        result = await client.call_tool(tool_name, arguments=arguments, timeout=timeout_s, raise_on_error=False)
        payload = {
            "data": getattr(result, "data", None),
            "structured_content": getattr(result, "structured_content", None),
            "text": _join_text(getattr(result, "content", None)),
            "is_error": getattr(result, "is_error", False),
        }
        return {"ok": True, "result": payload}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-file", required=True, help="Path to JSON request file")
    args = ap.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        req = json.load(f)

    mode = req.get("mode")
    server = req["server"]
    timeout_s = float(req.get("timeout", 20))

    try:
        if mode == "list-tools":
            out = asyncio.run(_list_tools(server, timeout_s))
        elif mode == "call-tool":
            tool_name = req["tool_name"]
            arguments = req.get("arguments", {})
            out = asyncio.run(_call_tool(server, tool_name, arguments, timeout_s))
        else:
            out = {"ok": False, "error": f"unknown mode: {mode}"}
    except Exception as e:
        out = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    _json_print(out)


if __name__ == "__main__":
    main()