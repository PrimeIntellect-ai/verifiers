"""Drive harness ACP sessions and workspace tool adapters over provider stdio."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import shlex
import sys
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from verifiers.v1.errors import HarnessError
from verifiers.v1.runtimes import Runtime
from verifiers.v1.runtimes.base import RuntimeBackend
from verifiers.v1.runtimes.stdio_rpc import ProcessRPC
from verifiers.v1.types import ToolCall, ToolMessage

if TYPE_CHECKING:
    from verifiers.v1.acp.runner import ACPSession

_ACP_PKG = "agent-client-protocol==0.12.1"


def _resolve_factory(module_name: str, qualname: str) -> Callable:
    target: object = importlib.import_module(module_name)
    for part in qualname.split("."):
        target = getattr(target, part)
    assert callable(target)
    return target


async def prepare_daemon_command(
    runtime: Runtime, extra_cls: Callable | None = None
) -> list[str]:
    """Prepare the interpreter command for the harness daemon and its ACP runner."""
    if runtime.type == "subprocess":
        # Overlay ACP on the evaluator's interpreter so the daemon can also
        # import installed harness plugins and editable task packages.
        return [
            "uv",
            "run",
            "--quiet",
            "--no-project",
            "--python",
            sys.executable,
            "--with",
            _ACP_PKG,
            "python",
        ]
    from verifiers.v1.harnesses.utils.remote import NativeRemoteTools
    from verifiers.v1.mcp.launch import install_in_runtime

    root_rt = runtime.backend.bind(env=runtime.env).with_user(None)
    python = await install_in_runtime(
        extra_cls or NativeRemoteTools, root_rt, label="harness daemon"
    )
    venv_dir = str(Path(python).parent.parent)
    install_key = f"install:{venv_dir}:{_ACP_PKG}"
    if install_key in root_rt.backend.cache.installed:
        return [python]
    async with root_rt.backend.cache.setup_lock:
        if install_key in root_rt.backend.cache.installed:
            return [python]
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
    return [python]


class HarnessDaemonClient:
    """Host-side RPC handle to one persistent `harness_daemon` process on `VM A`."""

    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime
        self._rpc: ProcessRPC | None = None
        self._closed = False

    @property
    def alive(self) -> bool:
        return not self._closed and self._rpc is not None and self._rpc._error is None

    async def start(self, *, extra_cls: Callable | None = None) -> None:
        root_rt = self.runtime.backend.bind(env={}).with_user(None)
        command = await prepare_daemon_command(root_rt, extra_cls=extra_cls)
        process = await root_rt.exec([*command, "-I", "-u", "-m", __name__], {})
        rpc = ProcessRPC(process)
        try:
            await asyncio.wait_for(rpc.call("ready"), timeout=45)
        except BaseException:
            await rpc.close()
            raise
        self._rpc = rpc

    async def configure_tools(
        self, factory: Callable, kwargs: dict, command: list[str]
    ) -> None:
        assert self._rpc is not None
        factory_name = getattr(factory, "__qualname__", None)
        if not factory_name or "<locals>" in factory_name:
            raise HarnessError("workspace tool factories must be importable by name")
        await self._rpc.call(
            "configure_tools",
            module=factory.__module__,
            factory=factory_name,
            kwargs=kwargs,
            command=command,
        )

    async def execute_tool(self, call: ToolCall) -> ToolMessage:
        assert self._rpc is not None
        result = await self._rpc.call("execute_tool", call=call.model_dump())
        return ToolMessage.model_validate(result)

    async def close_tools(self) -> None:
        if self.alive and self._rpc is not None:
            with suppress(Exception):
                await self._rpc.call("close_tools")

    async def acp_prompt(self, *, env: dict[str, str], config: dict[str, Any]) -> dict:
        assert self._rpc is not None
        return await self._rpc.call("acp_prompt", env=env, config=config)

    async def acp_steer(
        self, message: str, message_id: str | None = None
    ) -> dict[str, Any]:
        assert self._rpc is not None
        res = await self._rpc.call(
            "acp_steer",
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

    async def acp_close(self, *, graceful: bool) -> dict:
        if self.alive and self._rpc is not None:
            with suppress(Exception):
                return await self._rpc.call("acp_close", graceful=graceful)
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
) -> HarnessDaemonClient:
    """Start the dedicated harness runtime's daemon once per rollout."""
    if backend.type != "subprocess" and extra_cls is not None:
        from verifiers.v1.mcp.launch import install_in_runtime

        await install_in_runtime(
            extra_cls,
            backend.bind(env={}).with_user(None),
            label="harness tool adapter",
        )
    async with backend._daemon_lock:
        if backend.stopped:
            raise HarnessError("the rollout's harness runtime stopped")
        if backend._daemon is None:
            daemon = HarnessDaemonClient(backend.bind())
            await daemon.start(extra_cls=extra_cls)
            backend._daemon = daemon
        elif not backend._daemon.alive:
            raise HarnessError("the rollout's harness daemon stopped")
        return backend._daemon


async def main() -> None:
    from verifiers.v1.harnesses.utils.compaction import bound_tool_message
    from verifiers.v1.runtimes.remote_backend import RemoteBackend
    from verifiers.v1.runtimes.rpc_wire import serve

    tools: tuple[RemoteBackend, Callable] | None = None
    session: ACPSession | None = None
    prompt_task: asyncio.Task | None = None

    async def close_tools() -> None:
        nonlocal tools
        if tools is not None:
            backend, adapter = tools
            tools = None
            try:
                if close := getattr(adapter, "aclose", None):
                    await close()
            finally:
                await backend.stop()

    async def close_session(graceful: bool) -> dict:
        nonlocal session
        if session is None:
            return {}
        active, session = session, None
        if prompt_task is not None:
            prompt_task.cancel()
            with suppress(asyncio.CancelledError):
                await prompt_task
        return await active.close(graceful=graceful)

    async def handle_rpc(message: dict[str, Any]) -> Any:
        nonlocal tools, session, prompt_task
        op = message["operation"]
        if op == "ready":
            return None

        if op == "configure_tools":
            if tools is not None:
                raise HarnessError("workspace tools are already configured")
            factory = _resolve_factory(message["module"], message["factory"])
            backend = RemoteBackend(message["command"])
            tools = (backend, factory(backend.bind(), **message["kwargs"]))
            return None

        if op == "execute_tool":
            if tools is None:
                raise HarnessError("workspace tools are not configured")
            _, adapter = tools
            result = await adapter(ToolCall.model_validate(message["call"]))
            return bound_tool_message(result.model_dump(exclude_none=True))

        if op == "close_tools":
            await close_tools()
            return None

        if op == "acp_prompt":
            from acp import RequestError

            from verifiers.v1.acp.runner import ACPSession

            if prompt_task is not None:
                raise HarnessError("an ACP prompt is already in progress")
            if session is None:
                session = ACPSession(env=message["env"])
            active = session
            prompt_task = asyncio.current_task()
            try:
                turn = await active.run(message["config"])
                return {"ok": True, "result": asdict(turn)}
            except Exception as error:  # noqa: BLE001
                resp: dict[str, Any] = {
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                }
                if isinstance(error, RequestError) and isinstance(error.data, dict):
                    resp["error"] = error.data.get("details") or str(error)
                    resp["error_data"] = error.data
                resp["result"] = asdict(active.client.turn_result())
                return resp
            finally:
                prompt_task = None

        if op == "acp_steer":
            if session is None:
                return {"outcome": "promptRequired", "reason": "noRunningTurn"}
            try:
                return {
                    "ok": True,
                    "result": await session.steer(
                        message["message"], message.get("message_id")
                    ),
                }
            except NotImplementedError as error:
                return {"ok": False, "not_implemented": str(error)}

        if op == "acp_close":
            return await close_session(message["graceful"])

        raise HarnessError(f"unknown harness daemon operation: {op!r}")

    try:
        await serve(handle_rpc)
    finally:
        try:
            await close_session(False)
        finally:
            await close_tools()


if __name__ == "__main__":
    asyncio.run(main())
