"""Tool bundle runtime: one stdlib file shipped into a task's box.

A bundle is a directory with a `tools.py` whose `@tool` functions take the bundle's
JSON state document plus keyword arguments and return JSON. This file is:

- the `@tool` decorator, `ToolError` and `new_id` that bundles import;
- `Bundle` and `Session`: load a bundle and call its tools on a state, in-process
  (hosts use them for replay and tests);
- an MCP server over streamable HTTP (JSON responses, stateless);
- a CLI: `serve`, `seed`, `dump`, `call`, `schemas`.

It needs python3 >= 3.9 and nothing else. Tools must be deterministic: ids come from
`new_id` (a counter in the state) and the clock is `state["meta"]["now"]`.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import inspect
import json
import os
import re
import sys
import threading
import typing
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = "2025-06-18"
INTERFACES = ("direct", "search_execute")
ENDPOINTS_FILE = os.environ.get(
    "VF_TOOLS_ENDPOINTS", "/usr/local/lib/vf-tools/endpoints.json"
)


class ToolError(Exception):
    """A tool's refusal: the caller sees the message as a tool error."""


class ArgumentError(ToolError):
    """The arguments do not match the tool's schema."""


def tool(fn=None, *, name=None, app=None, schema=None, description=None):
    """Mark a function `(state, **arguments) -> JSON` as a tool of its bundle.

    `schema` (a JSON Schema object for the arguments) replaces the one derived from
    the signature; import passes the schema it observed so a mock matches exactly."""

    def mark(fn):
        fn._vf_tool = {
            "name": name or fn.__name__,
            "app": app,
            "schema": schema,
            "description": description,
        }
        return fn

    return mark(fn) if fn is not None else mark


def new_id(state, prefix):
    """The next id for `prefix`, from a counter kept in `state["meta"]`."""
    counters = state.setdefault("meta", {}).setdefault("counters", {})
    counters[prefix] = counters.get(prefix, 0) + 1
    return f"{prefix}_{counters[prefix]}"


_JSON_TYPES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    tuple: "array",
    dict: "object",
}


def _json_schema(annotation):
    if annotation is inspect.Parameter.empty or annotation is typing.Any:
        return {}
    origin = typing.get_origin(annotation)
    args = [a for a in typing.get_args(annotation) if a is not type(None)]
    if origin is typing.Union or (
        origin is not None and type(None) in typing.get_args(annotation)
    ):
        return _json_schema(args[0]) if len(args) == 1 else {}
    if origin is typing.Literal:
        return {"enum": list(typing.get_args(annotation))}
    if origin in (list, tuple):
        schema = {"type": "array"}
        if args:
            item = _json_schema(args[0])
            if item:
                schema["items"] = item
        return schema
    if origin is dict:
        return {"type": "object"}
    if annotation in _JSON_TYPES:
        return {"type": _JSON_TYPES[annotation]}
    return {}


def derive_schema(fn):
    """The arguments' JSON Schema from the signature (the first parameter is the state)."""
    try:
        hints = typing.get_type_hints(fn)
    except Exception:  # noqa: BLE001 - an annotation this Python cannot evaluate: leave it untyped
        hints = {}
    params = list(inspect.signature(fn).parameters.values())[1:]
    properties, required = {}, []
    for param in params:
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise TypeError(f"tool {fn.__name__}: *args and **kwargs are not allowed")
        properties[param.name] = _json_schema(
            hints.get(param.name, inspect.Parameter.empty)
        )
        if param.default is inspect.Parameter.empty:
            required.append(param.name)
        else:
            properties[param.name]["default"] = param.default
    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


class Tool:
    def __init__(self, fn):
        meta = fn._vf_tool
        self.fn = fn
        self.name = meta["name"]
        self.app = meta["app"]
        self.description = (meta["description"] or inspect.getdoc(fn) or "").strip()
        self.schema = meta["schema"] or derive_schema(fn)

    def spec(self):
        spec = {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.schema,
        }
        if self.app:
            spec["app"] = self.app
        return spec


_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
    "null": lambda v: v is None,
}


def check_arguments(schema, arguments):
    """Refuse missing, unknown and mistyped top-level arguments."""
    if not isinstance(arguments, dict):
        raise ArgumentError("arguments must be a JSON object")
    properties = schema.get("properties", {})
    missing = [name for name in schema.get("required", []) if name not in arguments]
    if missing:
        raise ArgumentError(
            "missing required argument(s): {}".format(", ".join(missing))
        )
    if schema.get("additionalProperties", False) is False and properties:
        unknown = sorted(set(arguments) - set(properties))
        if unknown:
            raise ArgumentError("unknown argument(s): {}".format(", ".join(unknown)))
    for name, value in arguments.items():
        expected = properties.get(name, {}).get("type")
        if expected is None:
            continue
        types = expected if isinstance(expected, list) else [expected]
        if not any(_TYPE_CHECKS.get(t, lambda v: True)(value) for t in types):
            raise ArgumentError(
                "argument {!r} must be {}".format(name, " or ".join(types))
            )


class Bundle:
    """A loaded bundle: its tools by name, and the interface it advertises."""

    def __init__(self, name, tools, interface="direct"):
        if interface not in INTERFACES:
            raise ValueError(
                "interface must be one of {}".format(", ".join(INTERFACES))
            )
        self.name = name
        self.tools = tools
        self.interface = interface

    @classmethod
    def load(cls, directory, name=None, interface="direct"):
        directory = os.path.abspath(directory)
        sys.modules.setdefault("vf_tools", sys.modules[__name__])
        if directory not in sys.path:
            sys.path.insert(0, directory)
        name = name or os.path.basename(directory)
        module_name = "vf_bundle_" + re.sub(r"\W", "_", name)
        spec = importlib.util.spec_from_file_location(
            module_name, os.path.join(directory, "tools.py")
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        tools = {}
        for value in vars(module).values():
            if callable(value) and hasattr(value, "_vf_tool"):
                item = Tool(value)
                if item.name in tools or item.name in ("search_api", "execute"):
                    raise ValueError(f"duplicate or reserved tool name {item.name!r}")
                tools[item.name] = item
        if not tools:
            raise ValueError(f"bundle {name} defines no @tool functions")
        return cls(name, tools, interface)

    def advertised(self):
        """The MCP tool list the agent sees."""
        if self.interface == "direct":
            return [t.spec() for t in self.tools.values()]
        apps = sorted({t.app for t in self.tools.values() if t.app})
        return [
            {
                "name": "search_api",
                "description": (
                    "Search the available APIs%s by keyword. Returns each matching "
                    "API's name, app, description and argument schema."
                    % (" (apps: {})".format(", ".join(apps)) if apps else "")
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "default": 10},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "execute",
                "description": "Call one API by name with its arguments object.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "api": {"type": "string"},
                        "arguments": {"type": "object"},
                    },
                    "required": ["api"],
                },
            },
        ]

    def search(self, query, limit=10):
        words = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored = []
        for t in self.tools.values():
            text = " ".join([t.name, t.app or "", t.description]).lower()
            tokens = set(re.findall(r"[a-z0-9]+", text))
            score = len(words & tokens) if words else 1
            if score:
                scored.append((-score, t.name, t))
        scored.sort(key=lambda item: item[:2])
        return [t.spec() for _, _, t in scored[: max(1, int(limit))]]


class Session:
    """A bundle bound to one state document. Calls are serialized; a call that raises
    leaves the state unchanged."""

    def __init__(self, bundle, state, on_commit=None):
        self.bundle = bundle
        self.state = state
        self.on_commit = on_commit
        self.lock = threading.Lock()

    def call(self, name, arguments=None):
        """Call an advertised tool or an API by name. Returns (ok, result): result is
        the JSON value, or the error message when ok is False."""
        arguments = {} if arguments is None else arguments
        try:
            if name == "search_api" and self.bundle.interface == "search_execute":
                check_arguments(self.bundle.advertised()[0]["inputSchema"], arguments)
                return True, self.bundle.search(
                    arguments["query"], arguments.get("limit", 10)
                )
            if name == "execute" and self.bundle.interface == "search_execute":
                check_arguments(self.bundle.advertised()[1]["inputSchema"], arguments)
                name, arguments = arguments["api"], arguments.get("arguments") or {}
            item = self.bundle.tools.get(name)
            if item is None:
                raise ToolError(f"unknown API {name!r}")
            check_arguments(item.schema, arguments)
            with self.lock:
                work = copy.deepcopy(self.state)
                result = item.fn(work, **arguments)
                json.dumps(result)
                if work != self.state:
                    self.state = work
                    if self.on_commit is not None:
                        self.on_commit(work)
            return True, result
        except ToolError as error:
            return False, str(error)
        except Exception as error:  # noqa: BLE001 - a bundle bug: the caller sees it as a tool error
            return False, f"internal error: {type(error).__name__}: {error}"


def read_state(path):
    with open(path) as handle:
        return json.load(handle)


def write_state(path, state):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as handle:
        json.dump(state, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def handle_rpc(session, message, server_name="vf-tools"):
    """One JSON-RPC message; None for a notification."""
    method = message.get("method")
    params = message.get("params") or {}
    if "id" not in message:
        return None
    reply = {"jsonrpc": "2.0", "id": message["id"]}
    if method == "initialize":
        reply["result"] = {
            "protocolVersion": params.get("protocolVersion") or PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": server_name, "version": "1"},
        }
    elif method == "ping":
        reply["result"] = {}
    elif method == "tools/list":
        tools = [
            {k: v for k, v in spec.items() if k != "app"}
            for spec in session.bundle.advertised()
        ]
        reply["result"] = {"tools": tools}
    elif method == "tools/call":
        ok, result = session.call(params.get("name"), params.get("arguments"))
        text = json.dumps(result) if ok else result
        reply["result"] = {
            "content": [{"type": "text", "text": text}],
            "isError": not ok,
        }
    else:
        reply["error"] = {"code": -32601, "message": f"method not found: {method}"}
    return reply


def make_server(session, host, port, server_name="vf-tools"):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            return

        def _send(self, status, body=None):
            data = b"" if body is None else json.dumps(body).encode()
            self.send_response(status)
            if body is not None:
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if self.path.split("?")[0].rstrip("/") != "/mcp":
                return self._send(404, {"error": "not found"})
            try:
                length = int(self.headers.get("Content-Length") or 0)
                message = json.loads(self.rfile.read(length) or b"null")
            except ValueError:
                return self._send(
                    400,
                    {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": -32700, "message": "parse error"},
                    },
                )
            if isinstance(message, list):
                replies = [
                    r
                    for r in (handle_rpc(session, m, server_name) for m in message)
                    if r
                ]
                return self._send(200, replies) if replies else self._send(202)
            if not isinstance(message, dict):
                return self._send(400, {"error": "invalid request"})
            reply = handle_rpc(session, message, server_name)
            return self._send(202) if reply is None else self._send(200, reply)

        def do_GET(self):
            self._send(405, {"error": "no server-initiated stream"})

        def do_DELETE(self):
            self._send(405, {"error": "stateless server"})

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def rpc(url, method, params=None, timeout=60.0):
    body = json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
    )
    request = urllib.request.Request(
        url,
        data=body.encode(),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        reply = json.loads(response.read())
    if "error" in reply:
        raise RuntimeError(reply["error"].get("message"))
    return reply["result"]


def _endpoint(args):
    if args.url:
        return args.url
    with open(ENDPOINTS_FILE) as handle:
        endpoints = json.load(handle)
    if args.bundle:
        return endpoints[args.bundle]
    if len(endpoints) != 1:
        raise SystemExit(
            "several bundles are mounted; pass --bundle ({})".format(
                ", ".join(sorted(endpoints))
            )
        )
    return next(iter(endpoints.values()))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="vf-tools")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve")
    serve.add_argument("--bundle", required=True)
    serve.add_argument("--name")
    serve.add_argument("--state", required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, required=True)
    serve.add_argument("--interface", default="direct", choices=INTERFACES)
    seed = sub.add_parser("seed")
    seed.add_argument("--state", required=True)
    seed.add_argument("--seed")
    dump = sub.add_parser("dump")
    dump.add_argument("--state", required=True)
    dump.add_argument("--out")
    call = sub.add_parser("call")
    call.add_argument("api")
    call.add_argument("arguments", nargs="?", default="{}")
    call.add_argument("--bundle")
    call.add_argument("--url")
    ping = sub.add_parser("ping")
    ping.add_argument("--bundle")
    ping.add_argument("--url")
    schemas = sub.add_parser("schemas")
    schemas.add_argument("--bundle", required=True)
    schemas.add_argument("--interface", default="direct", choices=INTERFACES)
    args = parser.parse_args(argv)

    if args.command == "serve":
        bundle = Bundle.load(args.bundle, args.name, args.interface)
        session = Session(
            bundle, read_state(args.state), lambda s: write_state(args.state, s)
        )
        server = make_server(session, args.host, args.port, bundle.name)
        server.serve_forever()
    elif args.command == "seed":
        state = read_state(args.seed) if args.seed else {}
        if not isinstance(state, dict):
            raise SystemExit("a seed must be a JSON object")
        write_state(args.state, state)
    elif args.command == "dump":
        state = read_state(args.state)
        text = json.dumps(state, indent=2, sort_keys=True)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w") as handle:
                handle.write(text + "\n")
        else:
            print(text)
    elif args.command == "call":
        try:
            arguments = json.loads(args.arguments)
        except ValueError as error:
            raise SystemExit(f"arguments are not JSON: {error}")
        result = rpc(
            _endpoint(args), "tools/call", {"name": args.api, "arguments": arguments}
        )
        text = result["content"][0]["text"]
        print(text)
        return 1 if result.get("isError") else 0
    elif args.command == "ping":
        rpc(_endpoint(args), "tools/list", timeout=5.0)
    elif args.command == "schemas":
        bundle = Bundle.load(args.bundle, interface=args.interface)
        print(
            json.dumps(
                {
                    "advertised": bundle.advertised(),
                    "apis": [t.spec() for t in bundle.tools.values()],
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    sys.modules.setdefault("vf_tools", sys.modules[__name__])
    sys.exit(main())
