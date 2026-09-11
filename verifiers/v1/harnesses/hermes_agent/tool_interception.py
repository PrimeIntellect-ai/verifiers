"""Mediate Hermes tool execution through the runner's private socket."""

import json
import os
import socket
import sys
import threading


def register(ctx):
    try:
        connection = socket.socket(socket.AF_UNIX)
        connection.settimeout(40)
        connection.connect(os.environ.pop("VF_TOOL_INTERCEPTION_SOCKET"))
        wire = connection.makefile("rwb")
        lock = threading.Lock()

        def policy(phase, message):
            with lock:
                request_id = f"{phase}:{message['tool_call_id']}"
                body = {"phase": phase, "content": "nonempty_text", "message": message}
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
            try:
                message = {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": "",
                }
                for phase in ("before", "after"):
                    decision = policy(phase, message)
                    if decision["action"] == "stop":
                        os._exit(70)
                    if decision["action"] == "rewrite":
                        return decision["message"]["content"]
                    if phase == "before":
                        message["content"] = next_call(args)
                return message["content"]
            except BaseException as error:  # noqa: BLE001 - native middleware must fail closed
                # Hermes resumes the original call/result when middleware raises.
                print(f"Tool interception failed: {error}", file=sys.stderr, flush=True)
                os._exit(70)

        ctx.register_middleware("tool_execution", execute)
    except BaseException as error:  # noqa: BLE001 - plugin loading must fail closed
        # Plugin load errors otherwise leave Hermes running without the monitor.
        print(f"Tool interception setup failed: {error}", file=sys.stderr, flush=True)
        os._exit(70)
