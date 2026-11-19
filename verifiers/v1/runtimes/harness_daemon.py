"""Persistent multi-tenant daemon on pooled Prime harness VMs (`VM A`).

Consolidates per-rollout Python workers onto one long-lived process per `VM A`:
1. Multi-tenant local `/tool` gate (`127.0.0.1:<gate_port>/tool`), routed by `Bearer <secret>`.
2. In-process ACP client driver + slot-indexed localhost HTTP router (`127.0.0.1:<router_port>/slot/<id>/...`)
   with a background standby pool of pre-initialized `claude-agent-acp` / `codex-acp` processes.
3. Fast file staging and directory cleanup over the open `ProcessRPC` pipe.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import os
import re
import secrets
import shlex
import shutil
import sys
import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from contextlib import AsyncExitStack, suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any

from verifiers.v1.errors import HarnessError
from verifiers.v1.harnesses.utils.compaction import bound_tool_message
from verifiers.v1.harnesses.utils.sandbox_tools import (
    _match_queued_item,
    _resolve_factory,
)
from verifiers.v1.runtimes import Runtime
from verifiers.v1.runtimes.base import RuntimeProcess
from verifiers.v1.runtimes.prepared import load_prepared, record_arguments
from verifiers.v1.runtimes.stdio_rpc import ProcessRPC
from verifiers.v1.types import ToolCall

STANDBY_TARGET = 2
_ACP_PKG = "agent-client-protocol==0.9.0"
_ENDPOINT_FILE_RE = re.compile(r"--endpoint-file=([^\s\"']+)")


async def ensure_daemon_python(runtime: Runtime) -> str:
    """Return `.vf-venv/bin/python` on `runtime` with `verifiers` and `acp` installed."""
    from verifiers.v1.harnesses.utils.remote import NativeRemoteTools
    from verifiers.v1.mcp.launch import install_in_runtime

    root_rt = runtime.backend.bind(env=runtime.env).with_user(None)
    python = await install_in_runtime(
        NativeRemoteTools, root_rt, label="harness daemon"
    )
    await load_prepared(root_rt)
    venv_dir = str(Path(python).parent.parent)
    install_key = f"install:{venv_dir}:acp-daemon-0.9.0"
    if install_key in root_rt.backend.cache.installed:
        return python
    async with root_rt.backend.cache.setup_lock:
        if install_key in root_rt.backend.cache.installed:
            return python
        ready = f"{shlex.quote(python)} -c 'import acp' >/dev/null 2>&1"
        rec = shlex.join(
            [
                python,
                *record_arguments(
                    root_rt, install_key, {"install": venv_dir, "ready": ready}
                ),
            ]
        )
        cmd = (
            'set -e; export PATH="$HOME/.local/bin:$PATH"; '
            f"({ready}) || uv pip install --python {shlex.quote(venv_dir)} {shlex.quote(_ACP_PKG)} -q; "
            f"{rec}"
        )
        res = await root_rt.run(["sh", "-c", cmd], {})
        if res.exit_code != 0:
            raise HarnessError(
                f"harness daemon dependency install failed: {(res.stderr or res.stdout).strip()[-1000:]}"
            )
        root_rt.backend.cache.installed.add(install_key)
    return python


class HarnessDaemonClient:
    """Host-side RPC handle to one persistent `harness_daemon` process on `VM A`."""

    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime
        self.local_gate_url: str = ""
        self.router_port: int = 0
        self._rpc: ProcessRPC | None = None
        self._stack = AsyncExitStack()
        self._closed = False

    @property
    def alive(self) -> bool:
        return not self._closed and self._rpc is not None and self._rpc._error is None

    async def start(self) -> None:
        from verifiers.v1.runtimes.controller import current_job

        root_rt = self.runtime.backend.bind(env={}).with_user(None)
        python = await ensure_daemon_python(root_rt)
        argv = [python, "-I", "-u", "-m", __name__]
        process: RuntimeProcess
        if current_job.get() and root_rt.type == "prime":
            from verifiers.v1.runtimes.lan_execution import LanExecution
            from verifiers.v1.runtimes.private_network import lan_host

            transport = LanExecution(root_rt, lan_host(root_rt.backend), direct=True)
            self._stack.push_async_callback(transport.close)
            await transport.start(root_rt)
            process = await transport.exec(root_rt, argv, {})
        else:
            process = await root_rt.exec(argv, {})

        rpc = ProcessRPC(process)
        ready: asyncio.Future[dict] = asyncio.get_running_loop().create_future()
        rpc._pending["__ready__"] = ready
        try:
            info = await asyncio.wait_for(ready, timeout=45.0)
        except BaseException:
            await rpc.close()
            await self._stack.aclose()
            raise
        finally:
            rpc._pending.pop("__ready__", None)
        self._rpc = rpc
        self.local_gate_url = f"http://127.0.0.1:{info['gate_port']}/tool"
        self.router_port = int(info["router_port"])

    async def write_file(self, path: str, data: bytes) -> None:
        assert self._rpc is not None
        await self._rpc.call(
            "write_file", path=path, data_b64=base64.b64encode(data).decode("ascii")
        )

    async def remove_dir(self, path: str) -> None:
        assert self._rpc is not None
        await self._rpc.call("remove_dir", path=path)

    async def register_tool_slot(
        self,
        *,
        slot_id: str,
        module: str,
        factory: str,
        kwargs: dict,
        endpoint: dict | None,
        command: list[str] | None,
        interception_url: str | None,
        secret: str,
    ) -> None:
        assert self._rpc is not None
        await self._rpc.call(
            "register_tool_slot",
            slot_id=slot_id,
            module=module,
            factory=factory,
            kwargs=kwargs,
            endpoint=endpoint,
            command=command,
            interception_url=interception_url,
            secret=secret,
        )

    async def execute_tool_slot(self, slot_id: str, call_dict: dict) -> dict:
        assert self._rpc is not None
        return await self._rpc.call(
            "execute_tool_slot", slot_id=slot_id, call=call_dict
        )

    async def unregister_tool_slot(self, slot_id: str) -> None:
        if not self.alive or self._rpc is None:
            return
        with contextlib.suppress(Exception):
            await self._rpc.call("unregister_tool_slot", slot_id=slot_id)

    async def acp_warm(
        self,
        *,
        session_key: str,
        env: dict[str, str],
        command: list[str],
        cwd: str | None,
        client_capabilities: dict[str, Any] | None = None,
    ) -> None:
        assert self._rpc is not None
        await self._rpc.call(
            "acp_warm",
            session_key=session_key,
            env=env,
            command=command,
            cwd=cwd,
            client_capabilities=client_capabilities or {},
        )

    async def acp_prompt(
        self,
        *,
        session_key: str,
        env: dict[str, str],
        config: dict[str, Any],
        secret: str,
    ) -> dict[str, Any]:
        assert self._rpc is not None
        return await self._rpc.call(
            "acp_prompt",
            session_key=session_key,
            env=env,
            config=config,
            secret=secret,
        )

    async def acp_steer(
        self, session_key: str, message: str, message_id: str | None = None
    ) -> dict[str, Any]:
        assert self._rpc is not None
        return await self._rpc.call(
            "acp_steer",
            session_key=session_key,
            message=message,
            message_id=message_id,
        )

    async def acp_close(
        self, session_key: str, *, graceful: bool = False
    ) -> dict[str, Any]:
        if not self.alive or self._rpc is None:
            return {}
        with contextlib.suppress(Exception):
            return await self._rpc.call(
                "acp_close", session_key=session_key, graceful=graceful
            )
        return {}

    async def aclose(self) -> None:
        self._closed = True
        try:
            if self._rpc is not None:
                with contextlib.suppress(Exception):
                    await self._rpc.process.abandon()
                await super(ProcessRPC, self._rpc).close()
                self._rpc = None
        finally:
            await self._stack.aclose()


# ---------------------------------------------------------------------------
# In-VM multi-tenant daemon (`python -I -u -m verifiers.v1.runtimes.harness_daemon`)
# ---------------------------------------------------------------------------


def _build_codex_standby_files(
    standby_home: str, router_port: int, slot_id: str, slot_token: str
) -> None:
    import tomli_w

    from verifiers.v1.harnesses.codex.harness import GATE_HANDLER, GATE_HOOK
    from verifiers.v1.harnesses.utils.remote_runtime import REMOTE_RUNTIME_SOURCE

    os.makedirs(standby_home, exist_ok=True)
    bridge_script = f"{standby_home}/bridge.py"
    Path(bridge_script).write_text(REMOTE_RUNTIME_SOURCE)

    environments = {
        "default": "task",
        "include_local": False,
        "environments": [
            {
                "id": "task",
                "program": sys.executable,
                "args": [
                    "-I",
                    "-u",
                    bridge_script,
                    f"--url=http://127.0.0.1:{router_port}/slot/{slot_id}/execution",
                    f"--secret={slot_token}",
                    "--worker=codex",
                    "--protocol=codex",
                    f"--endpoint-file={standby_home}/endpoint.json",
                ],
            }
        ],
    }
    Path(f"{standby_home}/environments.toml").write_bytes(
        tomli_w.dumps(environments).encode()
    )

    cfg_path = f"{standby_home}/config.toml"
    identity = {"event_name": "pre_tool_use", "hooks": [GATE_HANDLER]}
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    candidates = [cfg_path]
    if cfg_path.startswith("/tmp/"):
        candidates.append(f"/private{cfg_path}")
    settings = {
        "features": {"mcp_2026_07_28": True},
        "hooks": {
            "PreToolUse": [{"hooks": [GATE_HANDLER]}],
            "state": {
                f"{c}:pre_tool_use:0:0": {"trusted_hash": f"sha256:{digest}"}
                for c in candidates
            },
        },
    }
    Path(cfg_path).write_bytes(tomli_w.dumps(settings).encode())
    gate_url = f"http://127.0.0.1:{router_port}/slot/{slot_id}/codex-gate"
    Path(f"{standby_home}/tool-gate.mjs").write_text(
        GATE_HOOK.replace("__URL__", json.dumps(gate_url)).replace(
            "__SECRET__", json.dumps(slot_token)
        )
    )


async def main() -> None:
    import httpx
    from acp import PROTOCOL_VERSION, RequestError, spawn_agent_process
    from acp.schema import ClientCapabilities
    from aiohttp import web

    from verifiers.v1.acp import runner as acp_runner
    from verifiers.v1.runtimes.remote_backend import RemoteBackend
    from verifiers.v1.runtimes.rpc_wire import encode, serve

    http = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
        timeout=httpx.Timeout(None, connect=30.0),
        trust_env=False,
    )
    closing = False
    factory_cache: dict[tuple[str, str], Callable] = {}

    # 1. Multi-tenant /tool slots keyed by slot_id and secret
    tool_slots_by_id: dict[str, dict[str, Any]] = {}
    tool_slots_by_secret: dict[str, dict[str, Any]] = {}

    # 2. Localhost slot router (`127.0.0.1:<router_port>/slot/<slot_id>/...`) & standby ACP pool
    router_slots: dict[str, dict[str, Any]] = {}
    standby_queues: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    standby_inflight: dict[tuple, int] = defaultdict(int)
    acp_sessions: dict[str, dict[str, Any]] = {}
    bg_tasks: set[asyncio.Task] = set()

    def track_bg(coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        bg_tasks.add(task)
        task.add_done_callback(bg_tasks.discard)
        return task

    def post_slot_bg(interception_url: str, secret: str, payload: dict) -> None:
        async def send() -> None:
            headers = {"Authorization": f"Bearer {secret}"}
            for attempt in range(2):
                try:
                    resp = await http.post(
                        interception_url, json=payload, headers=headers
                    )
                    if resp.status_code < 500:
                        return
                except httpx.HTTPError:
                    if attempt == 0:
                        await asyncio.sleep(0.05)

        track_bg(send())

    async def poll_slot(slot: dict[str, Any]) -> None:
        url = slot["interception_url"]
        headers = {"Authorization": f"Bearer {slot['secret']}"}
        while not closing and not slot["closed"]:
            try:
                resp = await http.post(url, json={"op": "poll"}, headers=headers)
            except httpx.HTTPError:
                if closing or slot["closed"]:
                    return
                await asyncio.sleep(0.05)
                continue
            if resp.status_code != 200:
                slot["session_closed"] = True
                slot["batch_event"].set()
                return
            data = resp.json()
            if data.get("closed"):
                slot["session_closed"] = True
                slot["batch_event"].set()
                return
            new_items = data.get("items") or []
            if new_items:
                slot["queued_items"].extend(new_items)
                slot["batch_event"].set()

    async def handle_gate_http(request: web.Request) -> web.Response:
        secret = request.headers.get("Authorization", "").removeprefix("Bearer ")
        slot = tool_slots_by_secret.get(secret)
        if not secret or slot is None:
            return web.json_response({"error": "unauthorized"}, status=401)
        body = await request.json()
        tool_call_id = str(body.get("tool_call_id", ""))
        name = body.get("name")
        arguments = body.get("arguments")
        queued = slot["queued_items"]
        completed = slot["completed_verdicts"]
        batch_ev: asyncio.Event = slot["batch_event"]
        deadline = time.monotonic() + 1.5
        while True:
            matches = _match_queued_item(
                queued, completed, tool_call_id, name, arguments
            )
            if matches or slot["session_closed"]:
                break
            rem = deadline - time.monotonic()
            if rem <= 0:
                break
            batch_ev.clear()
            matches = _match_queued_item(
                queued, completed, tool_call_id, name, arguments
            )
            if matches or slot["session_closed"]:
                break
            with suppress(TimeoutError):
                await asyncio.wait_for(batch_ev.wait(), timeout=rem)

        url = slot["interception_url"]
        if (
            not url
            or slot["session_closed"]
            or len(matches) != 1
            or (name is not None and name != matches[0]["call"]["name"])
        ):
            resp = await http.post(
                url,
                json={
                    "tool_call_id": tool_call_id,
                    "name": name,
                    "arguments": arguments,
                },
                headers={"Authorization": f"Bearer {secret}"},
            )
            return web.json_response(resp.json(), status=resp.status_code)

        item = matches[0]
        item_key = (str(item["key"][0]), str(item["key"][1]))
        call_dict = item["call"]
        alias_prefix = str(item["alias_prefix"])
        lock = slot["key_locks"].setdefault(item_key, asyncio.Lock())
        async with lock:
            if cached := completed.get(item_key):
                return web.json_response({"action": "deny", "message": cached})
            if item["denied"] is not None:
                verdict_dict = item["denied"]
                completed[item_key] = verdict_dict
                post_slot_bg(
                    url,
                    secret,
                    {
                        "op": "ack_denied",
                        "key": item["key"],
                        "alias_prefix": alias_prefix,
                        "call": call_dict,
                        "arguments": arguments,
                    },
                )
                return web.json_response({"action": "deny", "message": verdict_dict})
            verdict = await slot["adapter"](ToolCall.model_validate(call_dict))
            verdict_dict = bound_tool_message(verdict.model_dump(exclude_none=True))
            completed[item_key] = verdict_dict
            post_slot_bg(
                url,
                secret,
                {
                    "op": "pin",
                    "key": item["key"],
                    "alias_prefix": alias_prefix,
                    "call": call_dict,
                    "message": verdict_dict,
                    "arguments": arguments,
                },
            )
            return web.json_response({"action": "deny", "message": verdict_dict})

    async def handle_router_http(request: web.Request) -> web.StreamResponse:
        slot_id = request.match_info["slot_id"]
        tail = request.match_info.get("tail", "")
        route = router_slots.get(slot_id)
        if route is None:
            return web.json_response({"error": "inactive slot"}, status=404)
        expected_tok = route["slot_token"]
        in_x_key = request.headers.get("x-api-key")
        in_auth = request.headers.get("Authorization", "")
        if in_x_key != expected_tok and in_auth != f"Bearer {expected_tok}":
            return web.json_response({"error": "unauthorized slot"}, status=401)

        raw_body = await request.read()
        active_secret = route["secret"]
        if tail == "codex-gate":
            target_url = route["tool_url"]
            fwd_headers = {
                "Authorization": f"Bearer {active_secret}",
                "Content-Type": "application/json",
            }
            resp = await http.post(target_url, content=raw_body, headers=fwd_headers)
            return web.Response(
                body=resp.content,
                status=resp.status_code,
                content_type="application/json",
            )

        qs = f"?{request.query_string}" if request.query_string else ""
        target_url = f"{route['upstream_base']}/{tail.lstrip('/')}{qs}"
        fwd_headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower()
            not in ("host", "content-length", "transfer-encoding", "connection")
        }
        if in_x_key is not None:
            fwd_headers["x-api-key"] = active_secret
        if in_auth:
            fwd_headers["Authorization"] = f"Bearer {active_secret}"

        async with http.stream(
            request.method, target_url, content=raw_body, headers=fwd_headers
        ) as upstream:
            ctype = upstream.headers.get("content-type", "application/json")
            if "text/event-stream" in ctype:
                stream_resp = web.StreamResponse(
                    status=upstream.status_code,
                    headers={"Content-Type": ctype},
                )
                await stream_resp.prepare(request)
                async for chunk in upstream.aiter_raw():
                    if chunk:
                        await stream_resp.write(chunk)
                await stream_resp.write_eof()
                return stream_resp
            data = await upstream.aread()
            return web.Response(
                body=data,
                status=upstream.status_code,
                headers={"Content-Type": ctype},
            )

    gate_app = web.Application(client_max_size=64 * 1024 * 1024)
    gate_app.router.add_post("/tool", handle_gate_http)
    gate_runner = web.AppRunner(gate_app, access_log=None)
    await gate_runner.setup()
    gate_site = web.TCPSite(gate_runner, "127.0.0.1", 0)
    await gate_site.start()
    gate_port = gate_site._server.sockets[0].getsockname()[1]

    router_app = web.Application(client_max_size=64 * 1024 * 1024)
    router_app.router.add_route("*", "/slot/{slot_id}/{tail:.*}", handle_router_http)
    router_runner = web.AppRunner(router_app, access_log=None)
    await router_runner.setup()
    router_site = web.TCPSite(router_runner, "127.0.0.1", 0)
    await router_site.start()
    router_port = router_site._server.sockets[0].getsockname()[1]

    def classify_standby(
        command: list[str], env: dict[str, str], config: dict[str, Any]
    ) -> tuple[str, tuple] | None:
        if config.get("mcp_urls"):
            return None
        cmd_tuple = tuple(command)
        if any("claude-agent-acp" in part for part in cmd_tuple):
            cfg_dir = env.get("CLAUDE_CONFIG_DIR", "")
            cwd_base = config.get("cwd") or os.getcwd()
            if cfg_dir and (
                os.path.isdir(f"{cfg_dir}/skills")
                or os.path.isdir(os.path.join(cwd_base, cfg_dir, "skills"))
            ):
                return None
            # Custom user env keys beyond standard ClaudeCodeHarness keys skip standby
            std_keys = {
                "ANTHROPIC_BASE_URL",
                "ANTHROPIC_API_KEY",
                "ANTHROPIC_MODEL",
                "CLAUDE_CODE_EXECUTABLE",
                "CLAUDE_CONFIG_DIR",
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
                "ENABLE_TOOL_SEARCH",
                "DISABLE_AUTOUPDATER",
                "IS_SANDBOX",
            }
            extra = tuple(sorted((k, v) for k, v in env.items() if k not in std_keys))
            tkey = (
                "claude",
                cmd_tuple,
                env.get("ANTHROPIC_MODEL", ""),
                env.get("CLAUDE_CODE_EXECUTABLE", ""),
                extra,
                json.dumps(config.get("client_capabilities") or {}, sort_keys=True),
            )
            return "claude", tkey
        if any("codex-acp" in part for part in cmd_tuple):
            home = env.get("CODEX_HOME", "")
            if not home or os.path.isdir(f"{home}/skills"):
                return None
            env_toml = Path(f"{home}/environments.toml")
            if not env_toml.is_file() or "--endpoint-file=" not in env_toml.read_text():
                return None
            std_keys = {
                "CODEX_CONFIG",
                "CODEX_HOME",
                "DEFAULT_AUTH_REQUEST",
                "INITIAL_AGENT_MODE",
                "NO_BROWSER",
            }
            extra = tuple(sorted((k, v) for k, v in env.items() if k not in std_keys))
            tkey = (
                "codex",
                cmd_tuple,
                env.get("CODEX_CONFIG", ""),
                env.get("INITIAL_AGENT_MODE", ""),
                extra,
                json.dumps(config.get("client_capabilities") or {}, sort_keys=True),
            )
            return "codex", tkey
        return None

    async def spawn_standby_one(
        kind: str,
        tkey: tuple,
        command: list[str],
        template_env: dict[str, str],
        client_caps: dict[str, Any],
    ) -> dict[str, Any]:
        slot_id = uuid.uuid4().hex[:12]
        slot_token = secrets.token_urlsafe(24)
        sb_dir = f"/tmp/vf-{kind}-sb-{slot_id}"
        os.makedirs(sb_dir, exist_ok=True)
        proc_env = {**os.environ, **template_env}
        if kind == "claude":
            Path(f"{sb_dir}/settings.json").write_text(
                json.dumps({"permissions": {"ask": ["*"]}})
            )
            proc_env["ANTHROPIC_BASE_URL"] = (
                f"http://127.0.0.1:{router_port}/slot/{slot_id}"
            )
            proc_env["ANTHROPIC_API_KEY"] = slot_token
            proc_env["CLAUDE_CONFIG_DIR"] = sb_dir
        else:
            _build_codex_standby_files(sb_dir, router_port, slot_id, slot_token)
            proc_env["CODEX_HOME"] = sb_dir
            proc_env["DEFAULT_AUTH_REQUEST"] = json.dumps(
                {
                    "methodId": "gateway",
                    "_meta": {
                        "gateway": {
                            "baseUrl": f"http://127.0.0.1:{router_port}/slot/{slot_id}/v1",
                            "headers": {"Authorization": f"Bearer {slot_token}"},
                            "providerName": "Verifiers",
                        }
                    },
                }
            )

        stack = AsyncExitStack()
        client = acp_runner.VerifiersACPClient()
        try:
            agent_proc = await stack.enter_async_context(
                spawn_agent_process(
                    client,
                    command[0],
                    *command[1:],
                    env=proc_env,
                    transport_kwargs={"stderr": None},
                )
            )
            conn = agent_proc[0]
            initialized = await conn.initialize(
                protocol_version=PROTOCOL_VERSION,
                client_capabilities=ClientCapabilities.model_validate(client_caps),
            )
        except BaseException:
            with suppress(BaseException):
                await stack.aclose()
            shutil.rmtree(sb_dir, ignore_errors=True)
            raise

        return {
            "kind": kind,
            "tkey": tkey,
            "slot_id": slot_id,
            "slot_token": slot_token,
            "sb_dir": sb_dir,
            "stack": stack,
            "client": client,
            "connection": conn,
            "initialized": initialized,
        }

    def replenish_standby(
        kind: str,
        tkey: tuple,
        command: list[str],
        template_env: dict[str, str],
        client_caps: dict[str, Any],
    ) -> None:
        needed = STANDBY_TARGET - (len(standby_queues[tkey]) + standby_inflight[tkey])
        for idx in range(max(0, needed)):
            standby_inflight[tkey] += 1

            async def build_one(delay: float) -> None:
                try:
                    await asyncio.sleep(delay)
                    if closing:
                        return
                    with suppress(Exception):
                        entry = await spawn_standby_one(
                            kind, tkey, command, template_env, client_caps
                        )
                        if closing:
                            await entry["stack"].aclose()
                            shutil.rmtree(entry["sb_dir"], ignore_errors=True)
                            return
                        standby_queues[tkey].append(entry)
                finally:
                    standby_inflight[tkey] = max(0, standby_inflight[tkey] - 1)

            track_bg(
                build_one(
                    0.0 if (idx == 0 and not standby_queues[tkey]) else 0.15 * (idx + 1)
                )
            )

    async def open_acp_session(
        session_key: str,
        env: dict[str, str],
        config: dict[str, Any],
        secret: str,
    ) -> dict[str, Any]:
        command = list(config["command"])
        client_caps = config.get("client_capabilities") or {}
        classified = classify_standby(command, env, config)
        standby_entry: dict[str, Any] | None = None

        if classified is not None:
            kind, tkey = classified
            queue = standby_queues[tkey]
            while queue:
                cand = queue.pop(0)
                standby_entry = cand
                break
            if standby_entry is None:
                standby_entry = await spawn_standby_one(
                    kind, tkey, command, env, client_caps
                )
            replenish_standby(kind, tkey, command, env, client_caps)

            slot_id = standby_entry["slot_id"]
            slot_token = standby_entry["slot_token"]
            gate_cfg = config.get("tool_interception")
            tool_url = gate_cfg["url"] if isinstance(gate_cfg, dict) else ""
            if kind == "claude":
                upstream_base = env["ANTHROPIC_BASE_URL"].rstrip("/")
            else:
                orig_home = env["CODEX_HOME"]
                env_toml_text = Path(f"{orig_home}/environments.toml").read_text()
                m = _ENDPOINT_FILE_RE.search(env_toml_text)
                if m:
                    orig_ep = m.group(1)
                    if os.path.exists(orig_ep):
                        os.replace(orig_ep, f"{standby_entry['sb_dir']}/endpoint.json")
                auth_gw = json.loads(env["DEFAULT_AUTH_REQUEST"])["_meta"]["gateway"]
                upstream_base = auth_gw["baseUrl"].removesuffix("/v1").rstrip("/")

            router_slots[slot_id] = {
                "upstream_base": upstream_base,
                "secret": secret,
                "tool_url": tool_url,
                "slot_token": slot_token,
            }
            conn = standby_entry["connection"]
            initialized = standby_entry["initialized"]
            client = standby_entry["client"]
            stack = standby_entry["stack"]
            sb_dir = standby_entry["sb_dir"]
        else:
            slot_id = ""
            sb_dir = ""
            stack = AsyncExitStack()
            client = acp_runner.VerifiersACPClient()
            agent_proc = await stack.enter_async_context(
                spawn_agent_process(
                    client,
                    command[0],
                    *command[1:],
                    env={**os.environ, **env},
                    transport_kwargs={"stderr": None},
                )
            )
            conn = agent_proc[0]
            initialized = await conn.initialize(
                protocol_version=PROTOCOL_VERSION,
                client_capabilities=ClientCapabilities.model_validate(client_caps),
            )

        try:
            required = config.get("required_agent_metadata")
            if required and required not in (initialized.field_meta or {}):
                raise ValueError(f"ACP agent does not advertise {required}")
            capabilities = initialized.agent_capabilities
            supports_steering = (initialized.field_meta or {}).get("steering", {}).get(
                "supported"
            ) is True
            gate_cfg = config.get("tool_interception")
            if gate_cfg and client.gate is None:
                client.gate = acp_runner.ToolGate(gate_cfg["url"], gate_cfg["secret"])
            sess = await conn.new_session(
                cwd=config.get("cwd") or os.getcwd(),
                mcp_servers=acp_runner.mcp_servers(config),
                **(config.get("session_meta") or {}),
            )
            if ready_file := config.get("ready_file"):
                dl = asyncio.get_running_loop().time() + 30
                while not os.path.isfile(ready_file):
                    if asyncio.get_running_loop().time() >= dl:
                        raise RuntimeError(
                            "Native tool gate did not activate before the first prompt"
                        )
                    await asyncio.sleep(0.05)
        except BaseException:
            if slot_id:
                router_slots.pop(slot_id, None)
            with suppress(BaseException):
                await stack.aclose()
            if sb_dir:
                shutil.rmtree(sb_dir, ignore_errors=True)
            raise

        record = {
            "stack": stack,
            "client": client,
            "connection": conn,
            "capabilities": capabilities,
            "supports_steering": supports_steering,
            "session_id": sess.session_id,
            "is_new": True,
            "slot_id": slot_id,
            "sb_dir": sb_dir,
            "lock": asyncio.Lock(),
        }
        acp_sessions[session_key] = record
        return record

    async def close_acp_session(session_key: str, graceful: bool) -> dict[str, Any]:
        rec = acp_sessions.pop(session_key, None)
        if rec is None:
            return {}
        if rec["slot_id"]:
            router_slots.pop(rec["slot_id"], None)
        meta: dict[str, Any] = {}
        try:
            if graceful and rec["connection"] is not None and rec["session_id"]:
                sc = rec["capabilities"] and rec["capabilities"].session_capabilities
                if sc and sc.close is not None:
                    with suppress(Exception):
                        r = await rec["connection"].close_session(
                            session_id=rec["session_id"]
                        )
                        meta = dict(r.field_meta or {})
        finally:
            with suppress(BaseException):
                await rec["stack"].aclose()
            if rec["client"].gate is not None:
                with suppress(Exception):
                    await rec["client"].gate.client.aclose()
                rec["client"].gate = None
            if rec["sb_dir"]:
                shutil.rmtree(rec["sb_dir"], ignore_errors=True)
        return meta

    async def handle_rpc(message: dict[str, Any]) -> Any:
        op = message["operation"]
        if op == "write_file":
            p = Path(message["path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(f"{p.name}.{uuid.uuid4().hex[:8]}.tmp")
            tmp.write_bytes(base64.b64decode(message["data_b64"]))
            os.replace(tmp, p)
            return None

        if op == "remove_dir":
            shutil.rmtree(message["path"], ignore_errors=True)
            return None

        if op == "register_tool_slot":
            slot_id = str(message["slot_id"])
            mod_name, fac_name = str(message["module"]), str(message["factory"])
            factory = factory_cache.get((mod_name, fac_name))
            if factory is None:
                factory = _resolve_factory(mod_name, fac_name)
                factory_cache[(mod_name, fac_name)] = factory
            if message.get("endpoint") is not None:
                backend = RemoteBackend(endpoint=message["endpoint"])
            else:
                backend = RemoteBackend(message["command"])
            await backend.start()
            adapter = factory(backend.bind(), **(message.get("kwargs") or {}))
            secret = str(message.get("secret") or "")
            interception_url = message.get("interception_url")
            slot: dict[str, Any] = {
                "slot_id": slot_id,
                "secret": secret,
                "backend": backend,
                "adapter": adapter,
                "interception_url": str(interception_url) if interception_url else "",
                "queued_items": [],
                "completed_verdicts": {},
                "key_locks": {},
                "batch_event": asyncio.Event(),
                "session_closed": False,
                "closed": False,
                "poller": None,
            }
            tool_slots_by_id[slot_id] = slot
            if secret and interception_url:
                tool_slots_by_secret[secret] = slot
                slot["poller"] = asyncio.create_task(poll_slot(slot))
            return None

        if op == "execute_tool_slot":
            slot = tool_slots_by_id.get(str(message["slot_id"]))
            if slot is None:
                raise HarnessError("tool slot is not registered")
            return (
                await slot["adapter"](ToolCall.model_validate(message["call"]))
            ).model_dump()

        if op == "unregister_tool_slot":
            slot = tool_slots_by_id.pop(str(message["slot_id"]), None)
            if slot is None:
                return None
            slot["closed"] = True
            if slot["secret"]:
                tool_slots_by_secret.pop(slot["secret"], None)
            if slot["poller"] is not None:
                slot["poller"].cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await slot["poller"]
            try:
                if close_fn := getattr(slot["adapter"], "aclose", None):
                    await close_fn()
            finally:
                await slot["backend"].stop()
            return None

        if op == "acp_warm":
            env = message["env"]
            cmd = list(message["command"])
            caps = message.get("client_capabilities") or {}
            classified = classify_standby(
                cmd, env, {"client_capabilities": caps, "cwd": message.get("cwd")}
            )
            if classified is not None:
                kind, tkey = classified
                if len(standby_queues[tkey]) + standby_inflight[tkey] < STANDBY_TARGET:
                    replenish_standby(kind, tkey, cmd, env, caps)
            return None

        if op == "acp_prompt":
            skey = str(message["session_key"])
            env = message["env"]
            cfg = message["config"]
            secret = str(message["secret"])
            rec = acp_sessions.get(skey)
            try:
                if rec is None:
                    rec = await open_acp_session(skey, env, cfg, secret)
                async with rec["lock"]:
                    turn = await acp_runner.prompt(
                        rec["client"],
                        rec["connection"],
                        rec["capabilities"],
                        rec["session_id"],
                        cfg,
                        is_new=rec["is_new"],
                    )
                    rec["is_new"] = False
                return {"ok": True, "result": asdict(turn)}
            except Exception as error:  # noqa: BLE001
                resp: dict[str, Any] = {
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                }
                if isinstance(error, RequestError) and isinstance(error.data, dict):
                    resp["error"] = error.data.get("details") or str(error)
                    resp["error_data"] = error.data
                if rec is not None:
                    resp["result"] = asdict(rec["client"].turn_result())
                return resp

        if op == "acp_steer":
            rec = acp_sessions.get(str(message["session_key"]))
            if rec is None or rec["connection"] is None or not rec["session_id"]:
                return {"outcome": "promptRequired", "reason": "noRunningTurn"}
            if not rec["supports_steering"]:
                raise NotImplementedError(
                    "ACP agent does not advertise steering support"
                )
            params: dict[str, Any] = {
                "sessionId": rec["session_id"],
                "prompt": [{"type": "text", "text": message["message"]}],
                "_meta": {"steering": {"idleBehavior": "promptRequired"}},
            }
            if message.get("message_id") is not None:
                params["messageId"] = message["message_id"]
            return await rec["connection"].ext_method("session/steering", params)

        if op == "acp_close":
            meta = await close_acp_session(
                str(message["session_key"]), bool(message.get("graceful"))
            )
            return {"response_metadata": meta}

        raise HarnessError(f"unknown harness daemon operation: {op!r}")

    sys.stdout.buffer.write(
        encode(
            {
                "id": "__ready__",
                "result": {"gate_port": gate_port, "router_port": router_port},
            }
        )
    )
    sys.stdout.buffer.flush()

    try:
        await serve(handle_rpc)
    finally:
        closing = True
        if bg_tasks:
            await asyncio.gather(*list(bg_tasks), return_exceptions=True)
        for skey in list(acp_sessions):
            with suppress(Exception):
                await close_acp_session(skey, False)
        for q in standby_queues.values():
            for entry in q:
                with suppress(Exception):
                    await entry["stack"].aclose()
                shutil.rmtree(entry["sb_dir"], ignore_errors=True)
        for sid in list(tool_slots_by_id):
            with suppress(Exception):
                await handle_rpc({"operation": "unregister_tool_slot", "slot_id": sid})
        await gate_runner.cleanup()
        await router_runner.cleanup()
        await http.aclose()


if __name__ == "__main__":
    import importlib as _imp  # ensure stdlib ready

    _ = _imp
    asyncio.run(main())
