"""Public Agent Client Protocol support for harness programs."""

import asyncio
import json
import secrets
from pathlib import Path, PurePosixPath
from weakref import WeakKeyDictionary

from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.types import Messages
from verifiers.v1.utils.aio import run_shielded

ACP_SOURCE = (Path(__file__).resolve().parent / "_runner.py").read_text()
PROBE_UNAVAILABLE_EXIT_CODE = 75

__all__ = ["ACP"]


class ACP:
    """Run an ACP agent."""

    def __init__(self) -> None:
        self._sidecar_locks: WeakKeyDictionary[Runtime, dict[str, asyncio.Lock]] = (
            WeakKeyDictionary()
        )

    def _sidecar_lock(self, runtime: Runtime, sidecar_path: str) -> asyncio.Lock:
        locks = self._sidecar_locks.get(runtime)
        if locks is None:
            locks = {}
            self._sidecar_locks[runtime] = locks
        lock = locks.get(sidecar_path)
        if lock is None:
            lock = asyncio.Lock()
            locks[sidecar_path] = lock
        return lock

    async def setup(self, harness: Harness, runtime: Runtime) -> None:
        await runtime.prepare_uv_script(
            ACP_SOURCE, {**harness.config.resolved_env, "UV_FROZEN": "false"}
        )

    async def run(
        self,
        runtime: Runtime,
        env: dict[str, str],
        command: list[str],
        prompt: str | Messages | None,
        *,
        mcp_urls: dict[str, str] | None = None,
        system_prompt: str | None = None,
        session_path: str | None = None,
        sidecar_path: str | None = None,
    ) -> ProgramResult:
        if prompt is None:
            raise ValueError("ACP requires a prompt")
        messages = (
            [{"role": "user", "content": prompt}]
            if isinstance(prompt, str)
            else [message_to_wire(message) for message in prompt]
        )
        config = {
            "command": command,
            "messages": messages,
            "mcp_urls": mcp_urls or {},
            "system_prompt": system_prompt or "",
            "session_path": session_path,
        }
        program = await runtime.prepare_uv_script(
            ACP_SOURCE, {**env, "UV_FROZEN": "false"}
        )
        sidecar_log = None
        if sidecar_path is not None:
            sidecar_dir = self._sidecar_dir(sidecar_path)
            sidecar_log = f"{sidecar_dir}/acp.log"
            async with self._sidecar_lock(runtime, sidecar_path):
                probe = await runtime.run([*program, "probe", sidecar_path], {})
                if probe.exit_code == PROBE_UNAVAILABLE_EXIT_CODE:
                    removed = await runtime.run(["rm", "-f", sidecar_path], {})
                    if removed.exit_code != 0:
                        raise RuntimeError(
                            "stale ACP sidecar cleanup failed: "
                            f"{removed.stderr.strip()}"
                        )
                    created = await runtime.run(
                        ["mkdir", "-p", "-m", "700", sidecar_dir], {}
                    )
                    if created.exit_code != 0:
                        raise RuntimeError(
                            f"ACP sidecar directory failed: {created.stderr.strip()}"
                        )
                    await runtime.run_background(
                        [*program, "serve", sidecar_path],
                        env,
                        sidecar_log,
                    )
                    ready = await runtime.run(
                        [*program, "probe", sidecar_path, "60"], {}
                    )
                    if ready.exit_code != 0:
                        log = await runtime.run(["tail", "-c", "4000", sidecar_log], {})
                        detail = (
                            ready.stderr.strip()
                            or ready.stdout.strip()
                            or "sidecar did not become ready"
                        )
                        if log.exit_code == 0 and log.stdout:
                            detail = (
                                f"{detail}\n\nACP sidecar log:\n{log.stdout.rstrip()}"
                            )
                        raise RuntimeError(f"ACP sidecar failed to start: {detail}")
                elif probe.exit_code != 0:
                    detail = (
                        probe.stderr.strip()
                        or probe.stdout.strip()
                        or "sidecar did not respond"
                    )
                    raise RuntimeError(f"ACP sidecar probe failed: {detail}")
        directory = f".vf-acp-{secrets.token_hex(8)}"
        created = await runtime.run(["mkdir", "-m", "700", directory], {})
        if created.exit_code != 0:
            raise RuntimeError(f"ACP config directory failed: {created.stderr.strip()}")
        path = f"{directory}/config.json"
        try:
            await runtime.write(path, json.dumps(config).encode())
            command = (
                [*program, "request", path, sidecar_path]
                if sidecar_path is not None
                else [*program, "once", path]
            )
            result = await runtime.run_program(command, env)
            if sidecar_log is not None and result.exit_code != 0:
                log = await runtime.run(["tail", "-c", "4000", sidecar_log], {})
                if log.exit_code == 0 and log.stdout:
                    result = ProgramResult(
                        exit_code=result.exit_code,
                        stdout=result.stdout,
                        stderr=(
                            f"{result.stderr.rstrip()}\n\nACP sidecar log:\n"
                            f"{log.stdout.rstrip()}"
                        ).lstrip(),
                    )
            return result
        finally:
            await run_shielded(runtime.run(["rm", "-rf", directory], {}))

    async def close(
        self, runtime: Runtime, sidecar_path: str, *, remove: bool = True
    ) -> None:
        """Stop a persistent ACP session, optionally keeping its artifacts."""
        sidecar_dir = self._sidecar_dir(sidecar_path)
        exists = await runtime.run(["test", "-S", sidecar_path], {})
        failure = ""
        try:
            if exists.exit_code == 0:
                program = await runtime.prepare_uv_script(
                    ACP_SOURCE, {"UV_FROZEN": "false"}
                )
                result = await runtime.run([*program, "shutdown", sidecar_path], {})
                if result.exit_code != 0:
                    log = await runtime.run(
                        ["tail", "-c", "4000", f"{sidecar_dir}/acp.log"], {}
                    )
                    failure = (
                        result.stderr.strip()
                        or result.stdout.strip()
                        or "ACP sidecar shutdown failed"
                    )
                    if log.exit_code == 0 and log.stdout:
                        failure = (
                            f"{failure}\n\nACP sidecar log:\n{log.stdout.rstrip()}"
                        )
        finally:
            if remove:
                await run_shielded(runtime.run(["rm", "-rf", sidecar_dir], {}))
        if failure:
            raise RuntimeError(failure)

    @staticmethod
    def _sidecar_dir(sidecar_path: str) -> str:
        path = PurePosixPath(sidecar_path)
        parent = str(path.parent)
        if path.is_absolute() or ".." in path.parts or parent in ("", ".", "/"):
            raise ValueError("ACP sidecar must live in a private subdirectory")
        return parent
