"""Clients and execution adapters for split harnesses."""

import asyncio
import json
import shlex

from verifiers.v1.runtimes import Runtime
from verifiers.v1.types import ToolCall, ToolMessage

# Embedded in Python harness programs so their native environment adapters use
# the same authenticated policy decision and idempotency contract as /tool.
REMOTE_TOOL_CLIENT_SOURCE = r"""
def remote_tool_decision(url, secret, name, arguments, call_id=None):
    import json
    import time
    import urllib.error
    import urllib.request
    import uuid
    body = json.dumps({"tool_call_id": call_id or uuid.uuid4().hex,
                       "name": name, "arguments": arguments}).encode()
    for attempt in range(3):
        request = urllib.request.Request(url, data=body, headers={
            "Authorization": "Bearer " + secret, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=86400) as response:
                decision = json.load(response)
            break
        except urllib.error.HTTPError:
            raise
        except (OSError, urllib.error.URLError):
            if attempt == 2:
                raise
            time.sleep(0.2 * (attempt + 1))
    if decision["action"] == "stop":
        raise RuntimeError(decision.get("reason", "Tool execution stopped"))
    return decision

def remote_tool(url, secret, name, arguments, call_id=None):
    decision = remote_tool_decision(url, secret, name, arguments, call_id)
    if decision["action"] == "allow":
        raise RuntimeError("Remote tool gate permitted local execution")
    return decision["message"]["content"]

def remote_shell_result(content):
    import json
    try:
        result = json.loads(content)
        if isinstance(result, dict) and result.get("_vf_shell_result") is True:
            return result
    except (TypeError, ValueError):
        pass
    # vf.intercept may supply a synthetic tool result without executing a shell.
    # Preserve it as stdout; an explicit shell-result envelope carries failures.
    return {"stdout": content if isinstance(content, str) else json.dumps(content),
            "stderr": "", "returncode": 0, "exception_info": ""}
"""


def remote_shell_tools(runtime: Runtime):
    """Preserve native environment result fields for shell-only harnesses."""

    async def execute(call: ToolCall) -> ToolMessage:
        args = json.loads(call.arguments or "{}")
        if call.name != "bash" or not isinstance(args.get("command"), str):
            raise ValueError("remote environment requires a bash command")
        command = args["command"]
        if cwd := args.get("cwd"):
            command = f"cd -- {shlex.quote(cwd)} && {command}"
        if args.get("combine_output"):
            command = "exec 2>&1\n" + command
        env = args.get("env") or {}
        if not isinstance(env, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in env.items()
        ):
            raise TypeError("remote command environment must contain strings")
        try:
            async with asyncio.timeout(args.get("timeout") or 86400):
                result = await runtime.run(["sh", "-c", command], env)
            output = {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.exit_code,
                "exception_info": "",
            }
        except TimeoutError:
            output = {
                "stdout": "",
                "stderr": "",
                "returncode": -1,
                "exception_info": "Command timed out",
            }
        return ToolMessage(
            tool_call_id=call.id,
            name=call.name,
            content=json.dumps({"_vf_shell_result": True, **output}),
        )

    execute.defer_until_gate = True
    return execute


REMOTE_RUNTIME_SOURCE = r"""
import argparse
import asyncio
import base64
import json
import os
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

parser = argparse.ArgumentParser()
parser.add_argument("--url", required=True)
parser.add_argument("--secret", required=True)
parser.add_argument("--worker", required=True)
parser.add_argument("--protocol", choices=("raw", "rlm", "prime", "codex"), default="raw")
parser.add_argument("--tool-url", default="")
parser.add_argument("--mcp-configs", default="")
args = parser.parse_args()
mcp_configs = json.loads(args.mcp_configs) if args.mcp_configs else {}
process_id = uuid.uuid4().hex
execution_url, execution_secret = args.url, args.secret
execution_opener = urllib.request.build_opener()
failed = threading.Event()
stdout_lock = threading.Lock()
history_receipts = {}
history_lock = threading.Lock()

def emit(data):
    with stdout_lock:
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()

def pack(message):
    data = json.dumps(message).encode()
    return struct.pack("!I", len(data)) + data

def frames(buffer):
    while len(buffer) >= 4:
        size = struct.unpack("!I", buffer[:4])[0]
        if size > 16 * 1024 * 1024:
            raise ValueError("execution frame exceeds 16 MiB")
        if len(buffer) < size + 4:
            return
        packet = bytes(buffer[:size + 4])
        del buffer[:size + 4]
        yield packet

_prime_loop = None
_prime_lock = threading.Lock()

def _dispatch_prime_mcp(req_id, req_type, req_data):
    global _prime_loop
    import rlm, rlm.mcp as _mcp

    with _prime_lock:
        if _prime_loop is None:
            async def _host_req(rt, payload):
                if rt == "mcp.config":
                    return mcp_configs.get(payload.get("server"))
                if rt == "mcp.list_connections":
                    return {"connections": [{"connectionId": n, "label": n, "service": n, "status": "connected"} for n in mcp_configs]}
                if rt in ("mcp.list_plugins", "mcp.search_plugins"):
                    return {"plugins": [], "nextCursor": None}
                raise RuntimeError("Supervisor callbacks are unavailable in split execution")

            rlm.host_request = _mcp.host_request = _host_req
            loop = asyncio.new_event_loop()
            threading.Thread(target=loop.run_forever, daemon=True).start()
            _prime_loop = loop

    async def _call():
        if req_type == "mcp.tools":
            return await _mcp.list_tools(req_data["server"])
        if req_type == "mcp.call":
            return await _mcp.call_tool(req_data["server"], req_data["tool"], req_data.get("arguments"))
        if req_type == "mcp.describe":
            return await _mcp.describe_tool(req_data["server"], req_data["tool"])
        if req_type == "mcp.search":
            return await _mcp.search_tools(req_data["query"], req_data.get("server"), req_data.get("limit", 20))
        return await _mcp.host_request(req_type, req_data)

    def run():
        try:
            res = asyncio.run_coroutine_threadsafe(_call(), _prime_loop).result(timeout=630)
            reply_data = {"status": "ok", "result": res}
        except Exception as exc:
            reply_data = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        reply = {"type": "host_reply", "id": req_id, "data": reply_data}
        rpc("write", data=base64.b64encode((json.dumps(reply) + "\n").encode()).decode())

    threading.Thread(target=run, daemon=True).start()


def prime_output(buffer):
    while b"\n" in buffer:
        at = buffer.index(b"\n")
        packet = bytes(buffer[:at + 1])
        del buffer[:at + 1]
        message = json.loads(packet)
        if message.get("event") == "host_request":
            req_data = message.get("data") or {}
            req_type = req_data.get("type", "")
            if mcp_configs and req_type in (
                "mcp.tools",
                "mcp.call",
                "mcp.describe",
                "mcp.search",
                "mcp.list_connections",
                "mcp.list_plugins",
                "mcp.search_plugins",
            ):
                _dispatch_prime_mcp(message["id"], req_type, req_data)
                continue
            # The kernel is arbitrary task code. It cannot drive the supervisor's
            # model, filesystem, harness-state, or delegation callbacks.
            reply = {
                "type": "host_reply",
                "id": message["id"],
                "data": {
                    "status": "error",
                    "error": "Supervisor callbacks are unavailable in split execution",
                },
            }
            rpc("write", data=base64.b64encode((json.dumps(reply) + "\n").encode()).decode())
            continue
        if message.get("event") == "display":
            # Native bash activity carries executor PIDs. Never let B name a PID
            # for A's cleanup, or claim to have sent supervisor agent messages.
            safe = {"application/vnd.prime-agent.diff+json",
                    "application/vnd.prime-agent.attachment+json",
                    "application/vnd.prime-agent.bash-command+json"}
            message["data"] = {key: value for key, value in (message.get("data") or {}).items() if key in safe}
            packet = (json.dumps(message) + "\n").encode()
        emit(packet)
    if len(buffer) > 16 * 1024 * 1024:
        raise ValueError("Prime Agent execution frame exceeds 16 MiB")

def codex_output(buffer):
    while b"\n" in buffer:
        at = buffer.index(b"\n")
        packet = bytes(buffer[:at + 1])
        del buffer[:at + 1]
        message = json.loads(packet)
        if "method" in message and "id" in message:
            # Exec-server can ask its supervisor for network policy decisions.
            # B's Runtime owns that policy; expose no reverse A capability.
            reply = {"id": message["id"], "error": {"code": -32601,
                "message": "Supervisor callbacks are unavailable in split execution"}}
            rpc("write", data=base64.b64encode((json.dumps(reply) + "\n").encode()).decode())
            continue
        emit(packet)
    if len(buffer) > 16 * 1024 * 1024:
        raise ValueError("Codex execution frame exceeds 16 MiB")

def rlm_input(packet):
    message = json.loads(packet[4:])
    method = message.get("method", "")
    if method == "host.events":
        # Status polling must survive a denied call so existing jobs can finish.
        # It launches no work; jobs.start and services.start still cross the gate.
        return packet
    if method != "tool" and method.split(".", 1)[0] not in ("host", "jobs", "services", "watch"):
        return packet
    params = message.get("params") or {}
    body = {"tool_call_id": "rlm:" + str(message["id"]),
            "name": params["name"] if method == "tool" else method,
            "arguments": params["args"] if method == "tool" else params}
    request = urllib.request.Request(args.tool_url, data=json.dumps(body).encode(), headers={
        "Authorization": "Bearer " + args.secret, "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=3600) as response:
        decision = json.load(response)
    if decision["action"] == "allow":
        return packet
    content = decision.get("reason") if decision["action"] == "stop" else decision.get("message", {}).get("content", "")
    if not isinstance(content, str):
        content = json.dumps(content)
    if method == "tool":
        # The supervisor advances its ledger cursor even when a tool is replaced.
        # Keep B's history cursor in sync without executing the refused tool.
        history_id = "vf-history-" + uuid.uuid4().hex
        receipt = {"ready": threading.Event()}
        with history_lock:
            history_receipts[history_id] = receipt
        history = pack({"id": history_id,
                        "method": "history", "params": params["history"]})
        try:
            rpc("write", data=base64.b64encode(history).decode())
            if not receipt["ready"].wait(30) or receipt.get("error"):
                raise RuntimeError("execution worker did not accept blocked-call history")
        finally:
            with history_lock:
                history_receipts.pop(history_id, None)
        reply = {"id": message["id"], "result": {"output": content,
                 "omitted": None, "metric_events": [], "notices": [], "tool_log": ""}}
    else:
        reply = {"id": message["id"], "error": content or "execution blocked by policy"}
    emit(pack(reply))
    return None

def rpc(operation, **values):
    message = {"operation": operation, "process_id": process_id,
               "request_id": uuid.uuid4().hex, **values}
    body = json.dumps(message).encode()
    for attempt in range(3):
        request = urllib.request.Request(execution_url, data=body, headers={
            "Authorization": "Bearer " + execution_secret,
            "Content-Type": "application/json",
        })
        try:
            with execution_opener.open(request, timeout=3600) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"execution bridge returned HTTP {error.code}") from error
        except (OSError, urllib.error.URLError):
            if attempt == 2:
                raise
            time.sleep(0.2 * (attempt + 1))

def copy_input():
    try:
        pending = bytearray()
        while data := os.read(sys.stdin.fileno(), 65536):
            if args.protocol == "rlm":
                pending.extend(data)
                for packet in frames(pending):
                    if packet := rlm_input(packet):
                        rpc("write", data=base64.b64encode(packet).decode())
            else:
                rpc("write", data=base64.b64encode(data).decode())
        rpc("kill")
    except Exception as error:
        failed.set()
        print(f"execution bridge input failed: {type(error).__name__}: {error}", file=sys.stderr)
        try:
            rpc("kill")
        except Exception:
            pass

def copy_output(stream, output):
    try:
        pending = bytearray()
        while True:
            result = rpc("read", stream=stream)
            data = base64.b64decode(result.get("data") or "")
            if data:
                if stream == "stdout" and args.protocol == "codex":
                    pending.extend(data)
                    codex_output(pending)
                elif stream == "stdout" and args.protocol == "prime":
                    pending.extend(data)
                    prime_output(pending)
                elif stream == "stdout" and args.protocol == "rlm":
                    pending.extend(data)
                    for packet in frames(pending):
                        message = json.loads(packet[4:])
                        if "method" in message and message["method"] != "broker":
                            # Only nano-rlm's scoped broker may call back into A.
                            # RemoteREPL checks the session capability, and the
                            # supervisor checks the active cell and operation.
                            reply = pack({"id": message["id"], "error":
                                          "Supervisor callbacks are unavailable in split execution"})
                            rpc("write", data=base64.b64encode(reply).decode())
                            continue
                        with history_lock:
                            receipt = history_receipts.get(message.get("id"))
                        if receipt is not None:
                            receipt["error"] = message.get("error")
                            receipt["ready"].set()
                            continue
                        emit(packet)
                else:
                    output.write(data)
                    output.flush()
            if result.get("eof"):
                return
    except Exception as error:
        failed.set()
        print(f"execution bridge {stream} failed: {type(error).__name__}: {error}", file=sys.stderr)
        try:
            rpc("kill")
        except Exception:
            pass

rpc("open", worker=args.worker)
readers = [threading.Thread(target=copy_output, args=(name, output), daemon=True)
           for name, output in (("stdout", sys.stdout.buffer), ("stderr", sys.stderr.buffer))]
try:
    for reader in readers:
        reader.start()
    threading.Thread(target=copy_input, daemon=True).start()
    result = rpc("wait")
    for reader in readers:
        reader.join(timeout=10)
    code = result["exit_code"]
    if failed.is_set() or any(reader.is_alive() for reader in readers):
        code = code or 1
finally:
    try:
        rpc("kill")
    except Exception:
        pass
raise SystemExit(code)
"""


async def prepare_remote_bridge(
    runtime: Runtime,
    url: str,
    secret: str,
    worker: str,
    *,
    protocol: str = "raw",
    tool_url: str = "",
    mcp_configs: dict[str, dict] | None = None,
    python: str | None = None,
) -> list[str]:
    """Connect the harness to a workspace process registered by the evaluator."""
    args = [
        python or await runtime.resolve_python(),
        "-P",
        "-s",
        "-u",
        "-c",
        REMOTE_RUNTIME_SOURCE,
        f"--url={url}",
        f"--secret={secret}",
        f"--worker={worker}",
        f"--protocol={protocol}",
        f"--tool-url={tool_url}",
    ]
    if mcp_configs:
        args.append(f"--mcp-configs={json.dumps(mcp_configs)}")
    return args
