"""Mediate Codex Code Mode's gRPC calls inside the ACP runner."""

import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

MAX_FRAME_BYTES = 64 * 1024 * 1024
# {codex_protocol}


def find_host(launcher: str) -> str:
    launcher = Path(launcher).resolve()
    node_modules = next(
        (parent for parent in launcher.parents if parent.name == "node_modules"), None
    )
    if node_modules is None:
        raise RuntimeError(f"Cannot locate Codex package from {launcher}")
    matches = list(
        node_modules.glob("@openai/codex-*/vendor/*/bin/codex-code-mode-host")
    )
    if len(matches) != 1:
        raise RuntimeError(f"Found {len(matches)} Codex Code Mode hosts")
    return str(matches[0])


async def intercept(
    policy: Callable[[dict], Awaitable[dict]],
    phase: str,
    call_id: str,
    name: str,
    content,
    detached_parent: str,
    tool_call: dict | None = None,
) -> dict:
    decision = await policy(
        {
            "phase": phase,
            "content": "any",
            "detachedParent": detached_parent,
            "toolCall": tool_call,
            "message": {
                "role": "tool",
                "tool_call_id": call_id,
                "content": content,
                "name": name,
            },
        }
    )
    if decision["action"] == "stop":
        # Never resume JavaScript after a stop: even a caught tool error can
        # otherwise cause another tool call with an unapproved result.
        raise RuntimeError(decision.get("reason") or "Tool interception stopped")
    if decision["action"] == "rewrite":
        message = decision.get("message")
        if not isinstance(message, dict):
            raise ValueError("Tool interception omitted its replacement")
        policy_call_id = decision.get("toolCallId", call_id)
        if message.get("tool_call_id") != policy_call_id or message.get("name") != name:
            raise ValueError("Tool interception changed the Code Mode tool identity")
    return decision


def response_content(response):
    variant = response.WhichOneof("outcome")
    if variant not in {"yielded", "terminated", "completed"}:
        raise ValueError("Code Mode omitted its execution outcome")
    parts = []
    for item in response.content_items:
        if item.WhichOneof("item") == "text":
            parts.append({"type": "text", "text": item.text.text})
        elif item.WhichOneof("item") == "image":
            parts.append(
                {"type": "image_url", "image_url": {"url": item.image.image_url}}
            )
        else:
            raise ValueError("Code Mode interception does not support this result")
    if variant == "completed" and response.completed.HasField("error_text"):
        parts.append(
            {"type": "text", "text": f"Script error:\n{response.completed.error_text}"}
        )
    if not parts:
        return ""
    if len(parts) == 1 and parts[0]["type"] == "text":
        return parts[0]["text"]
    return parts


def wire_content(content) -> list[dict]:
    parts = [{"type": "text", "text": content}] if isinstance(content, str) else content
    result = []
    for part in parts:
        if part.get("type") == "text":
            result.append({"text": {"text": part.get("text", "")}})
        elif part.get("type") == "image_url":
            result.append(
                {"image": {"image_url": (part.get("image_url") or {}).get("url", "")}}
            )
        else:
            raise ValueError("Tool interception returned unsupported content")
    return result


async def apply_result_policy(proxy, response, session_id, call_id, name) -> None:
    async with proxy.tool_lock:
        if response.WhichOneof("outcome") in {"completed", "terminated"}:
            # Terminal outcomes can overtake cancellation events on the lease
            # stream. Retire this cell's delegates before delivering its result.
            proxy.closed_cells.add((session_id, response.cell_id))
            for key, (cell_id, call) in list(proxy.calls.items()):
                if key[0] == session_id and cell_id == response.cell_id:
                    proxy.cancelled.add(key)
                    await intercept(
                        proxy.policy, "cancel", call["id"], call["name"], "", name
                    )
                    del proxy.calls[key]
        decision = await intercept(
            proxy.policy, "after", call_id, name, response_content(response), name
        )
    if decision["action"] == "rewrite":
        del response.content_items[:]
        for item in wire_content(decision["message"]["content"]):
            response.content_items.add(**item)
        if response.WhichOneof("outcome") == "completed":
            response.completed.ClearField("error_text")


class CodexProxy:
    def __init__(self, channel, proto, services, policy):
        self.channel = channel
        self.proto = proto
        self.host = services.CodeModeHostStub(channel)
        self.policy = policy
        self.failure = asyncio.get_running_loop().create_future()
        self.peer: str | None = None
        self.parents: dict[str, str] = {}
        self.calls: dict[tuple[str, str], tuple[str, dict]] = {}
        self.cancelled: set[tuple[str, str]] = set()
        self.closed_cells: set[tuple[str, str]] = set()
        self.tool_lock = asyncio.Lock()

    async def stream(self, method, request, context):
        if method == "OpenSession" and self.peer is None:
            self.peer = context.peer()
        if context.peer() != self.peer:
            raise RuntimeError("Code Mode host is already connected")
        upstream = None
        try:
            blocked = False
            outcome_received = False
            session_id = None
            if method == "Execute":
                self.parents[request.session_id] = "exec"
                decision = await intercept(
                    self.policy, "before", request.tool_call_id, "exec", "", "exec"
                )
                blocked = decision["action"] == "rewrite"
                if blocked:
                    # Let the host retain ownership of cell IDs and lifecycle
                    # events, executing only the approved literal replacement.
                    request.source = "\n".join(
                        f"{kind}({json.dumps(value)});"
                        for item in wire_content(decision["message"]["content"])
                        for kind, values in item.items()
                        for value in values.values()
                    )
                    del request.enabled_tools[:]
            upstream = getattr(self.host, method)(request)
            # Codex waits for subscription headers before starting executions;
            # waiting for the first invocation here would deadlock startup.
            await context.send_initial_metadata(await upstream.initial_metadata())
            async for response in upstream:
                if method == "OpenSession":
                    if response.WhichOneof("event") == "opened":
                        session_id = response.opened.session_id
                    elif response.WhichOneof("event") == "tool_call_cancelled":
                        key = (session_id, response.tool_call_cancelled.invocation_id)
                        # Remember cancellation even if its invocation has not
                        # arrived, or its pre hook is still awaiting policy.
                        self.cancelled.add(key)
                        async with self.tool_lock:
                            pending = self.calls.pop(key, None)
                            if pending is not None:
                                _, call = pending
                                await intercept(
                                    self.policy,
                                    "cancel",
                                    call["id"],
                                    call["name"],
                                    "",
                                    "",
                                )
                elif method == "SubscribeToToolCalls":
                    key = (response.session_id, response.invocation_id)
                    call = {
                        "id": f"vf-{response.session_id}-{response.invocation_id}",
                        "name": response.tool_name.name,
                        "type": "custom" if response.tool_kind == 2 else "function",
                        "arguments": (
                            json.loads(response.input_json)
                            if response.tool_kind == 2
                            else response.input_json.decode() or "null"
                        ),
                    }
                    async with self.tool_lock:
                        if (
                            key in self.cancelled
                            or (response.session_id, response.cell_id)
                            in self.closed_cells
                        ):
                            continue
                        if key in self.calls:
                            raise ValueError("Code Mode repeated a nested invocation")
                        decision = await intercept(
                            self.policy,
                            "before",
                            call["id"],
                            call["name"],
                            "",
                            self.parents[response.session_id],
                            call,
                        )
                        if decision["action"] == "allow":
                            self.calls[key] = (response.cell_id, call)
                    if key in self.cancelled:
                        continue
                    if decision["action"] == "rewrite":
                        # Nested results are JSON values consumed by JavaScript,
                        # so replacements must use the same JSON representation.
                        content = json.dumps(json.loads(decision["message"]["content"]))
                        await self.host.CompleteToolCall(
                            self.proto.CompleteToolCallRequest(
                                session_id=response.session_id,
                                invocation_id=response.invocation_id,
                                succeeded={"output_json": content.encode()},
                            )
                        )
                        continue
                elif method == "Execute" and response.WhichOneof("event") == "outcome":
                    outcome_received = True
                    if not blocked:
                        await apply_result_policy(
                            self,
                            response.outcome,
                            request.session_id,
                            request.tool_call_id,
                            "exec",
                        )
                yield response
            if method == "Execute" and not outcome_received:
                raise ValueError("Code Mode execution omitted its tool result")
        except Exception as error:
            # Dropping the host lease cancels every active cell and delegate.
            # A single failed RPC must not leave other streams executing tools.
            if not self.failure.done():
                self.failure.set_result(error)
            await self.channel.close()
            raise
        finally:
            if upstream is not None:
                upstream.cancel()

    async def unary(self, method, request, context):
        if context.peer() != self.peer:
            raise RuntimeError("Code Mode host is already connected")
        try:
            if method in {"Wait", "Terminate"}:
                self.parents[request.session_id] = "wait"
            elif method == "CompleteToolCall":
                key = (request.session_id, request.invocation_id)
                async with self.tool_lock:
                    if key in self.cancelled:
                        return self.proto.CompleteToolCallResponse()
                    _, call = self.calls.pop(key)
                    outcome = request.WhichOneof("outcome")
                    if outcome not in {"succeeded", "failed"}:
                        raise ValueError("Code Mode omitted its nested tool result")
                    content = (
                        request.succeeded.output_json.decode()
                        if outcome == "succeeded"
                        else json.dumps({"error": request.failed.message})
                    )
                    json.loads(content)
                    decision = await intercept(
                        self.policy,
                        "after",
                        call["id"],
                        call["name"],
                        content,
                        self.parents[request.session_id],
                        call,
                    )
                    if decision["action"] == "rewrite":
                        content = json.dumps(json.loads(decision["message"]["content"]))
                        request.succeeded.output_json = content.encode()
                if key in self.cancelled:
                    return self.proto.CompleteToolCallResponse()
            response = await getattr(self.host, method)(request)
            if method in {"Wait", "Terminate"}:
                state = response.WhichOneof("state")
                if state not in {"live_cell", "missing_cell"}:
                    raise ValueError("Code Mode continuation omitted its tool result")
                await apply_result_policy(
                    self,
                    getattr(response, state),
                    request.session_id,
                    f"vf-wait-{getattr(request, 'wait_id', request.cell_id)}",
                    "wait",
                )
            return response
        except Exception as error:
            if not self.failure.done():
                self.failure.set_result(error)
            await self.channel.close()
            raise


@asynccontextmanager
async def running_codex_proxy(launcher: str, policy):
    import grpc

    # Compile the pinned upstream schema in the runner, without generated stubs
    # or gRPC dependencies in the importing Verifiers process.
    with TemporaryDirectory(prefix="vf-codex-proto-") as directory:
        (Path(directory) / "vf_codex.proto").write_text(CODEX_PROTOCOL)  # noqa: F821
        sys.path.insert(0, directory)
        try:
            proto, services = grpc.protos_and_services("vf_codex.proto")
        finally:
            sys.path.remove(directory)
    host = await asyncio.create_subprocess_exec(
        find_host(launcher),
        "--listen",
        "grpc://127.0.0.1:0",
        stdout=asyncio.subprocess.PIPE,
        stderr=None,
    )
    assert host.stdout is not None
    try:
        host_url = (
            (await asyncio.wait_for(host.stdout.readline(), timeout=15))
            .decode()
            .strip()
        )
        if not host_url.startswith("http://127.0.0.1:"):
            raise RuntimeError(
                f"Codex Code Mode host returned an invalid endpoint: {host_url!r}"
            )
        options = [
            ("grpc.max_receive_message_length", MAX_FRAME_BYTES),
            ("grpc.max_send_message_length", MAX_FRAME_BYTES),
        ]
        async with grpc.aio.insecure_channel(
            host_url.removeprefix("http://"), options=options
        ) as channel:
            proxy = CodexProxy(channel, proto, services, policy)
            server = grpc.aio.server(options=options)
            service = proto.DESCRIPTOR.services_by_name["CodeModeHost"]
            server.add_generic_rpc_handlers(
                (
                    grpc.method_handlers_generic_handler(
                        service.full_name,
                        {
                            method.name: (
                                grpc.unary_stream_rpc_method_handler
                                if method.server_streaming
                                else grpc.unary_unary_rpc_method_handler
                            )(
                                partial(
                                    proxy.stream
                                    if method.server_streaming
                                    else proxy.unary,
                                    method.name,
                                ),
                                request_deserializer=getattr(
                                    proto, method.input_type.name
                                ).FromString,
                                response_serializer=getattr(
                                    proto, method.output_type.name
                                ).SerializeToString,
                            )
                            for method in service.methods
                        },
                    ),
                )
            )
            port = server.add_insecure_port("127.0.0.1:0")
            await server.start()
            try:
                yield f"http://127.0.0.1:{port}", proxy.failure
            finally:
                await server.stop(0)
    finally:
        if host.returncode is None:
            host.terminate()
            try:
                await asyncio.wait_for(host.wait(), timeout=5)
            except TimeoutError:
                host.kill()
                await host.wait()
