"""Explicit whole-agent workload scopes; VM control and grading remain outside."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from verifiers.v1.errors import SandboxError
from verifiers.v1.utils.aio import run_shielded

if TYPE_CHECKING:
    from verifiers.v1.runtimes import Runtime

DIRECTORY = "/run/vf-task-scope"
HELPER = f"{DIRECTORY}/control"

PYTHON_PROBE = """import os, pathlib, stat, sys
path = pathlib.Path(sys.executable).resolve(strict=True)
for item in (path, *path.parents):
    info = item.stat()
    if info.st_uid != 0 or info.st_mode & 0o022:
        raise RuntimeError("interpreter is not root protected")
print(path)
"""


@dataclass(frozen=True)
class ProcessScope:
    control: Runtime

    @classmethod
    async def prepare(cls, runtime: Runtime) -> ProcessScope:
        if (
            not runtime.backend.SUPPORTS_USER_SWITCHING
            or not runtime.backend.SUPPORTS_TASK_IMAGES
        ):
            raise SandboxError(
                "workload scopes require an isolated user-switching runtime"
            )
        control = runtime.with_user(None).with_env({})
        created = await control.run(["sh", "-c", f"umask 077; mkdir {DIRECTORY}"], {})
        if created.exit_code:
            raise SandboxError("workload scope state directory is unavailable")
        interpreter = None
        for candidate in ("/usr/local/bin/python3", "/usr/bin/python3"):
            probe = await control.run([candidate, "-I", "-c", PYTHON_PROBE], {})
            if probe.exit_code == 0:
                resolved = probe.stdout.strip()
                if not resolved.startswith("/") or any(c.isspace() for c in resolved):
                    raise SandboxError("invalid workload scope interpreter path")
                interpreter = resolved
                break
        if interpreter is None:
            raise SandboxError("workload scopes require a root-protected Python 3")
        source = Path(__file__).with_name("_process_scope.py").read_bytes()
        body = source.split(b"\n", 1)[1]
        await control.write(HELPER, f"#!{interpreter} -I\n".encode() + body)
        secured = await control.run(["chmod", "700", HELPER], {})
        if secured.exit_code:
            raise SandboxError("workload scope helper could not be secured")
        scope = cls(control)
        await scope.command("prepare")
        return scope

    @property
    def env(self) -> dict[str, str]:
        return {"VF_TASK_SCOPE_HELPER": HELPER}

    async def command(self, *argv: str) -> None:
        result = await self.control.run([HELPER, *argv], {})
        if result.exit_code:
            raise SandboxError("workload scope operation failed")

    def wrap(self, runtime: Runtime, argv: list[str], env: dict[str, str]):
        if runtime.user is None:
            raise SandboxError("scoped agent launch requires an explicit non-root user")
        child_env = runtime.process_env(env)
        return (
            self.control,
            [HELPER, "run", "agent", "--user", runtime.user, "--", *argv],
            {"VF_SCOPE_CHILD_ENV": json.dumps(child_env)},
        )

    async def finish_verification(self) -> None:
        # The agent stays frozen until scoring and authoritative reward reads finish.
        # Never resume the agent beside a scorer whose termination is unconfirmed.
        try:
            await run_shielded(self.command("kill", "scorer"))
        except BaseException:
            await run_shielded(self.command("kill", "agent"))
            raise
        await run_shielded(self.command("thaw", "agent"))

    async def kill_agent_after_close(self) -> None:
        await run_shielded(self.command("kill", "agent"))
