"""Persistent multi-tenant daemon on split harness sandboxes (`VM A`).

Consolidates per-rollout Python workers onto one long-lived process per `VM A`:
1. Multi-tenant local `/tool` gate (`127.0.0.1:<gate_port>/tool`), routed by `Bearer <secret>`.
2. In-process `ACPSession` driver (sharing `verifiers.v1.acp.runner`).
3. Fast file staging and directory cleanup over the open `ProcessRPC` pipe.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import importlib
import json
import os
import shlex
import shutil
import sys
import time
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import Any

from verifiers.v1.errors import HarnessError
from verifiers.v1.harnesses.utils.compaction import bound_tool_message
from verifiers.v1.runtimes import Runtime
from verifiers.v1.runtimes.base import RuntimeBackend
from verifiers.v1.runtimes.stdio_rpc import ProcessRPC
from verifiers.v1.types import ToolCall

_ACP_PKG = "agent-client-protocol==0.12.1"


def _resolve_factory(module_name: str, qualname: str) -> Callable:
    target: object = importlib.import_module(module_name)
    for part in qualname.split("."):
        target = getattr(target, part)
    assert callable(target)
    return target


def _same_arguments(raw_call: dict, arguments: object) -> bool:
    raw_args = raw_call.get("arguments", "")
    try:
        return json.dumps(json.loads(raw_args), sort_keys=True) == json.dumps(
            arguments, sort_keys=True
        )
    except (json.JSONDecodeError, TypeError):
        return raw_args == arguments


def match_queued_item(
    items: list[dict],
    completed: dict[tuple[str, str], dict],
    tool_call_id: str,
    name: str | None,
    arguments: object,
) -> list[dict]:
    keys = {tool_call_id}
    if isinstance(arguments, dict):
        keys.update(
            arguments[field]
            for field in ("tool_call_id", "call_id", "toolCallId")
            if isinstance(arguments.get(field), str)
        )

    def id_matches(pool: list[dict]) -> list[dict]:
        return [
            item
            for item in pool
            if any(
                key == (cid := item["call"]["id"])
                or key.startswith(f"{cid}|")
                or key.endswith(f":{cid}")
                for key in keys
            )
        ]

    uncompleted = [
        item
        for item in items
        if (str(item["key"][0]), str(item["key"][1])) not in completed
    ]
    matches = (
        id_matches(uncompleted)
        or id_matches(items)
        or [
            item
            for item in uncompleted
            if (name is None or item["call"]["name"] == name)
            and _same_arguments(item["call"], arguments)
        ]
    )
    if len(matches) > 1:
        matches = [
            item
            for item in matches
            if (name is None or item["call"]["name"] == name)
            and _same_arguments(item["call"], arguments)
        ] or matches
    return matches


async def ensure_daemon_python(
    runtime: Runtime, extra_cls: Callable | None = None
) -> str:
    """Return a Python interpreter on `runtime` with `verifiers` and `acp` installed."""
    if runtime.type == "subprocess":
        return sys.executable
    from verifiers.v1.harnesses.utils.remote import NativeRemoteTools
    from verifiers.v1.mcp.launch import install_in_runtime

    root_rt = runtime.backend.bind(env=runtime.env).with_user(None)
    python = await install_in_runtime(
        extra_cls or NativeRemoteTools, root_rt, label="harness daemon"
    )
    venv_dir = str(Path(python).parent.parent)
    install_key = f"install:{venv_dir}:{_ACP_PKG}"
    if install_key in root_rt.backend.cache.installed:
        return python
    async with root_rt.backend.cache.setup_lock:
        if install_key in root_rt.backend.cache.installed:
            return python
        ready = (
            f"{shlex.quote(python)} -c "
            "\"import importlib.metadata as m; assert m.version('agent-client-protocol') == '0.12.1'\" "
            ">/dev/null 2>&1"
        )
        cmd = (
            'set -e; export PATH="$HOME/.local/bin:$PATH"; '
            f"({ready}) || uv pip install --python {shlex.quote(venv_dir)} {shlex.quote(_ACP_PKG)} -q"
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
        self._rpc: ProcessRPC | None = None
        self._closed = False

    @property
    def alive(self) -> bool:
        return not self._closed and self._rpc is not None and self._rpc._error is None

    async def start(self, *, extra_cls: Callable | None = None) -> None:
        root_rt = self.runtime.backend.bind(env={}).with_user(None)
        python = await ensure_daemon_python(root_rt, extra_cls=extra_cls)
        process = await root_rt.exec([python, "-I", "-u", "-m", __name__], {})
        rpc = ProcessRPC(process)
        ready: asyncio.Future[dict] = asyncio.get_running_loop().create_future()
        rpc._pending["__ready__"] = ready
        try:
            info = await asyncio.wait_for(ready, timeout=45.0)
        except BaseException:
            await rpc.close()
            raise
        finally:
            rpc._pending.pop("__ready__", None)
        self._rpc = rpc
        self.local_gate_url = f"http://127.0.0.1:{info['gate_port']}/tool"

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

    async def acp_prompt(
        self,
        *,
        session_key: str,
        env: dict[str, str],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        assert self._rpc is not None
        return await self._rpc.call(
            "acp_prompt",
            session_key=session_key,
            env=env,
            config=config,
        )

    async def acp_steer(
        self, session_key: str, message: str, message_id: str | None = None
    ) -> dict[str, Any]:
        assert self._rpc is not None
        res = await self._rpc.call(
            "acp_steer",
            session_key=session_key,
            message=message,
            message_id=message_id,
        )
        if (
            isinstance(res, dict)
            and res.get("ok") is False
            and "not_implemented" in res
        ):
            raise NotImplementedError(res["not_implemented"])
        if isinstance(res, dict) and "result" in res:
            return res["result"]
        return res

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
        if self._rpc is not None:
            with contextlib.suppress(Exception):
                await self._rpc.process.abandon()
            await self._rpc.close(kill=False)
            self._rpc = None


async def ensure_backend_daemon(
    backend: RuntimeBackend, *, extra_cls: Callable | None = None
) -> HarnessDaemonClient | None:
    """Return or start the shared `HarnessDaemonClient` on `backend`, if supported."""
    if getattr(backend, "type", None) not in (
        "prime",
        "docker",
        "subprocess",
        "custom",
    ):
        return None
    if backend.type == "prime" and (
        getattr(backend, "_client", None) is None
        or not hasattr(backend._client, "open_process")
    ):
        return None
    if backend.type != "subprocess" and extra_cls is not None:
        from verifiers.v1.mcp.launch import install_in_runtime

        await install_in_runtime(
            extra_cls,
            backend.bind(env={}).with_user(None),
            label="harness tool adapter",
        )
    existing: HarnessDaemonClient | None = getattr(backend, "_daemon", None)
    if existing is not None and existing.alive:
        return existing
    lock: asyncio.Lock = getattr(backend, "_daemon_lock", None) or asyncio.Lock()
    backend._daemon_lock = lock  # type: ignore[attr-defined]
    async with lock:
        existing = getattr(backend, "_daemon", None)
        if existing is not None and existing.alive:
            return existing
        if existing is not None:
            with contextlib.suppress(Exception):
                await existing.aclose()
            backend._daemon = None  # type: ignore[attr-defined]
        from verifiers.v1.utils.logging import isolated

        with isolated():
            daemon = HarnessDaemonClient(backend.bind())
            await daemon.start(extra_cls=extra_cls)
        backend._daemon = daemon  # type: ignore[attr-defined]
        return daemon


# ---------------------------------------------------------------------------
# In-VM multi-tenant daemon (`python -I -u -m verifiers.v1.runtimes.harness_daemon`)
# ---------------------------------------------------------------------------


async def main() -> None:
    import httpx
    from aiohttp import web

    from verifiers.v1.runtimes.remote_backend import RemoteBackend
    from verifiers.v1.runtimes.rpc_wire import encode, serve

    http = httpx.AsyncClient(
        limits=httpx.Limits(max_connections=256, max_keepalive_connections=64),
        timeout=httpx.Timeout(None, connect=30.0),
        trust_env=False,
    )
    closing = False
    factory_cache: dict[tuple[str, str], Callable] = {}
    tool_slots_by_id: dict[str, dict[str, Any]] = {}
    tool_slots_by_secret: dict[str, dict[str, Any]] = {}
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

    async def decide_gate(secret: str, body: dict) -> tuple[int, dict]:
        slot = tool_slots_by_secret.get(secret)
        if not secret or slot is None:
            return 401, {"error": "unauthorized"}
        tool_call_id = str(body.get("tool_call_id", ""))
        name = body.get("name")
        arguments = body.get("arguments")
        queued = slot["queued_items"]
        completed = slot["completed_verdicts"]
        batch_ev: asyncio.Event = slot["batch_event"]
        deadline = time.monotonic() + 1.5
        while True:
            matches = match_queued_item(
                queued, completed, tool_call_id, name, arguments
            )
            if matches or slot["session_closed"]:
                break
            rem = deadline - time.monotonic()
            if rem <= 0:
                break
            batch_ev.clear()
            matches = match_queued_item(
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
            return resp.status_code, resp.json()

        item = matches[0]
        item_key = (str(item["key"][0]), str(item["key"][1]))
        call_dict = item["call"]
        lock = slot["key_locks"].setdefault(item_key, asyncio.Lock())
        async with lock:
            if cached := completed.get(item_key):
                return 200, {"action": "deny", "message": cached}
            if item["denied"] is not None:
                verdict_dict = item["denied"]
                completed[item_key] = verdict_dict
                post_slot_bg(
                    url,
                    secret,
                    {
                        "op": "ack_denied",
                        "key": item["key"],
                        "arguments": arguments,
                    },
                )
                return 200, {"action": "deny", "message": verdict_dict}
            verdict = await slot["adapter"](ToolCall.model_validate(call_dict))
            verdict_dict = bound_tool_message(verdict.model_dump(exclude_none=True))
            completed[item_key] = verdict_dict
            post_slot_bg(
                url,
                secret,
                {
                    "op": "pin",
                    "key": item["key"],
                    "message": verdict_dict,
                    "arguments": arguments,
                },
            )
            return 200, {"action": "deny", "message": verdict_dict}

    async def handle_gate_http(request: web.Request) -> web.Response:
        secret = request.headers.get("Authorization", "").removeprefix("Bearer ")
        try:
            status, payload = await decide_gate(secret, await request.json())
        except Exception as error:  # noqa: BLE001
            status, payload = 400, {"error": str(error)}
        return web.json_response(payload, status=status)

    gate_app = web.Application(client_max_size=64 * 1024 * 1024)
    gate_app.router.add_post("/tool", handle_gate_http)
    gate_runner = web.AppRunner(gate_app, access_log=None)
    await gate_runner.setup()
    gate_site = web.TCPSite(gate_runner, "127.0.0.1", 0)
    await gate_site.start()
    gate_port = gate_site._server.sockets[0].getsockname()[1]

    async def close_acp_session(session_key: str, graceful: bool) -> dict[str, Any]:
        rec = acp_sessions.pop(session_key, None)
        if rec is None:
            return {}
        rec["closed"] = True
        if task := rec.get("prompt_task"):
            if not task.done():
                task.cancel()
            with suppress(BaseException):
                await task
        return await rec["session"].close(graceful=graceful)

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
            target = Path(message["path"])
            if target.is_symlink() or target.is_file():
                target.unlink(missing_ok=True)
            elif target.exists():
                shutil.rmtree(target)
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

        if op == "acp_prompt":
            from acp import RequestError

            from verifiers.v1.acp.runner import ACPSession

            skey = str(message["session_key"])
            env = message["env"]
            cfg = message["config"]
            rec = acp_sessions.get(skey)
            if rec is None:
                rec = {
                    "session": ACPSession(env=env),
                    "lock": asyncio.Lock(),
                    "prompt_task": None,
                    "closed": False,
                }
                acp_sessions[skey] = rec
            session: ACPSession = rec["session"]
            rec["prompt_task"] = asyncio.current_task()
            try:
                async with rec["lock"]:
                    if rec["closed"]:
                        raise HarnessError("ACP session was closed")
                    turn = await session.run(cfg)
                return {"ok": True, "result": asdict(turn)}
            except Exception as error:  # noqa: BLE001
                resp: dict[str, Any] = {
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                }
                if isinstance(error, RequestError) and isinstance(error.data, dict):
                    resp["error"] = error.data.get("details") or str(error)
                    resp["error_data"] = error.data
                resp["result"] = asdict(session.client.turn_result())
                return resp
            finally:
                rec["prompt_task"] = None

        if op == "acp_steer":
            rec = acp_sessions.get(str(message["session_key"]))
            if rec is None or rec["closed"]:
                return {"outcome": "promptRequired", "reason": "noRunningTurn"}
            try:
                return {
                    "ok": True,
                    "result": await rec["session"].steer(
                        message["message"], message.get("message_id")
                    ),
                }
            except NotImplementedError as error:
                return {"ok": False, "not_implemented": str(error)}

        if op == "acp_close":
            meta = await close_acp_session(
                str(message["session_key"]), bool(message.get("graceful"))
            )
            return {"response_metadata": meta}

        raise HarnessError(f"unknown harness daemon operation: {op!r}")

    sys.stdout.buffer.write(
        encode({"id": "__ready__", "result": {"gate_port": gate_port}})
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
        for sid in list(tool_slots_by_id):
            with suppress(Exception):
                await handle_rpc({"operation": "unregister_tool_slot", "slot_id": sid})
        await gate_runner.cleanup()
        await http.aclose()


if __name__ == "__main__":
    asyncio.run(main())
