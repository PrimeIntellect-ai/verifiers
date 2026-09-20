"""Mediate Hermes tool execution through the runner's private socket."""

import json
import os
import socket
import sys
import threading
from contextvars import ContextVar
from uuid import uuid4


def register(ctx):
    try:
        connection = socket.socket(socket.AF_UNIX)
        connection.settimeout(40)
        connection.connect(os.environ.pop("VF_TOOL_INTERCEPTION_SOCKET"))
        wire = connection.makefile("rwb")
        lock = threading.Lock()
        parent = ContextVar[str | None]("vf_execute_code", default=None)

        def code_mode(request, **kwargs):
            from tools.code_execution_tool import SANDBOX_ALLOWED_TOOLS

            tools = request.get("tools") or []
            names = [tool.get("function", tool).get("name") for tool in tools]
            if "execute_code" in names:
                # Keep tools available to the Python bridge while hiding their direct schemas.
                return {
                    "request": {
                        **request,
                        "tools": [
                            tool
                            for name, tool in zip(names, tools)
                            if name not in SANDBOX_ALLOWED_TOOLS
                        ],
                    }
                }

        def policy(phase, message, call):
            with lock:
                request_id = f"{phase}:{message['tool_call_id']}"
                body = {"phase": phase, "content": "nonempty_text", "message": message}
                if call is not None:
                    body.update(detachedParent="execute_code", toolCall=call)
                wire.write(
                    json.dumps({"id": request_id, "body": body}).encode() + b"\n"
                )
                wire.flush()
                response = json.loads(wire.readline())
                if response.get("id") != request_id or response.get("error"):
                    raise RuntimeError("Tool interception failed")
                decision = response["decision"]
                if decision.get("action") not in {"allow", "rewrite", "stop"}:
                    raise RuntimeError("Invalid tool interception decision")
                return decision

        def execute(tool_name, tool_call_id, args, next_call, **kwargs):
            token = None
            try:
                call = None
                if not tool_call_id:
                    # Hermes copies the cell's context into its tool RPC threads.
                    if parent.get() is None:
                        raise RuntimeError("Hermes omitted the tool call ID")
                    tool_call_id = f"vf-{parent.get()}-{uuid4().hex}"
                    call = {
                        "id": tool_call_id,
                        "name": tool_name,
                        "type": "function",
                        "arguments": json.dumps(args),
                    }
                if tool_name == "execute_code":
                    token = parent.set(tool_call_id)
                message = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": "",
                }
                for phase in ("before", "after"):
                    decision = policy(phase, message, call)
                    if decision["action"] == "stop":
                        os._exit(70)
                    if decision["action"] == "rewrite":
                        if call is not None:
                            # The tool RPC decodes this value before returning to Python.
                            json.loads(decision["message"]["content"])
                        return decision["message"]["content"]
                    if phase == "before":
                        message["content"] = next_call(args)
                return message["content"]
            except BaseException as error:  # noqa: BLE001 - native middleware must fail closed
                # Hermes resumes the original call/result when middleware raises.
                print(f"Tool interception failed: {error}", file=sys.stderr, flush=True)
                os._exit(70)
            finally:
                if token is not None:
                    parent.reset(token)

        ctx.register_middleware("llm_request", code_mode)
        ctx.register_middleware("tool_execution", execute)
    except BaseException as error:  # noqa: BLE001 - plugin loading must fail closed
        # Plugin load errors otherwise leave Hermes running without the monitor.
        print(f"Tool interception setup failed: {error}", file=sys.stderr, flush=True)
        os._exit(70)
