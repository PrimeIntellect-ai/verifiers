"""Remote E2B sandbox runtime.

E2B boots sandboxes from templates, not images: the first sandbox on an image (at a given
CPU/memory size) builds a template from it, named by a digest of those inputs, and every
later sandbox starts from the cached template in about a second. `expose` (sandbox port ->
public URL) is E2B's per-port host; the reverse direction (a program in the sandbox
reaching a host service) is the shared host-side `Tunnel` (interception.tunnel), not the
runtime's concern. E2B streams process output as text, which cannot carry the
length-prefixed binary ACP packets, so live processes are not supported.
"""

import asyncio
import contextlib
import hashlib
import logging
import shlex
from typing import ClassVar, Literal

from pydantic_config import BaseConfig

from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import (
    SERVICE_PORT,
    BaseRuntimeInfo,
    ProgramResult,
    Runtime,
)
from verifiers.v1.utils.aio import run_shielded

logger = logging.getLogger(__name__)

# One template build per (image, cpu, memory) per process; concurrent rollouts on the same
# image wait for the first build instead of racing it.
_template_locks: dict[str, asyncio.Lock] = {}


class E2BConfig(BaseConfig):
    type: Literal["e2b"] = "e2b"
    image: str = "python:3.11-slim"
    """Docker image to run; built into an E2B template on first use."""
    workdir: str = "/app"
    user: str = "root"
    """User that runs commands and owns written files."""
    network_access: bool = True
    # TaskData.resources uses these units; non-default runtime config values take precedence.
    cpu: float = 1.0
    """CPU cores (E2B allocates whole cores; fixed per template)."""
    memory: float = 2.0
    """Memory in GB (fixed per template)."""
    disk: float = 5.0
    """Disk in GB. E2B sizes the disk from the template, so this is accepted (so a task can
    declare it without a warning) but not enforced."""
    timeout: float = 3600
    """Maximum sandbox lifetime in seconds. E2B caps it at 1 hour on Hobby and 24 hours on
    Pro plans."""


class E2BRuntimeInfo(E2BConfig, BaseRuntimeInfo):
    template: str | None = None
    """Name of the E2B template the sandbox booted from."""


class E2BRuntime(Runtime):
    is_local: ClassVar[bool] = False

    def __init__(self, config: E2BConfig, name: str | None = None) -> None:
        super().__init__(name)
        self.config = config
        self.info = E2BRuntimeInfo(**config.model_dump())
        self._sandbox = None

    @property
    def published_port(self) -> int | None:
        return SERVICE_PORT

    async def start(self) -> None:
        try:
            from e2b import AsyncSandbox
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "E2BRuntime requires the E2B SDK; install `verifiers[e2b]`."
            ) from e

        try:
            template = await self._ensure_template()
            self.info.template = template

            async def create_and_capture() -> None:
                self._sandbox = await AsyncSandbox.create(
                    template=template,
                    timeout=int(self.config.timeout),
                    envs=self.env,
                    metadata={"name": self.name},
                    allow_internet_access=self.config.network_access,
                )
                self.info.id = self._sandbox.sandbox_id

            # Shielded through the handle capture: a cancel mid-create would otherwise
            # leave a running sandbox that `teardown` never learns of.
            await run_shielded(create_and_capture())
            logger.info(
                "e2b: sandbox %s up (image=%s template=%s)",
                self.info.id,
                self.config.image,
                template,
            )
            await self._sandbox.files.make_dir(
                self.config.workdir, user=self.config.user
            )
        except (
            Exception
        ) as e:  # provisioning failure is one rollout's problem, not the eval's
            raise SandboxError(f"e2b sandbox provisioning failed: {e}") from e

    async def _ensure_template(self) -> str:
        from e2b import AsyncTemplate

        cpu = max(1, round(self.config.cpu))
        memory_mb = max(128, round(self.config.memory * 1024))
        digest = hashlib.sha256(
            f"{self.config.image}:{cpu}:{memory_mb}".encode()
        ).hexdigest()
        name = f"vf-{digest[:16]}"
        async with _template_locks.setdefault(name, asyncio.Lock()):
            if not await AsyncTemplate.exists(name):
                logger.warning(
                    "e2b: no template for image %s yet - building %s (first use of an "
                    "image takes a minute or more, later sandboxes start in seconds)",
                    self.config.image,
                    name,
                )
                await AsyncTemplate.build(
                    AsyncTemplate().from_image(self.config.image),
                    name,
                    cpu_count=cpu,
                    memory_mb=memory_mb,
                )
        return name

    async def expose(self, port: int) -> str | None:
        return f"https://{self._sandbox.get_host(port)}"

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        from e2b import CommandExitException

        try:
            result = await self._sandbox.commands.run(
                shlex.join(argv),
                envs=self.process_env(env),
                user=self.config.user,
                cwd=self.config.workdir,
                timeout=0,  # rollout cancellation owns the execution timeout
            )
        except CommandExitException as e:  # a non-zero exit is a result, not a failure
            result = e
        except (
            Exception
        ) as e:  # a sandbox/API failure is one rollout's problem, not the eval's
            raise SandboxError(f"e2b exec failed: {e}") from e
        return ProgramResult(
            exit_code=result.exit_code or 0,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
        )

    async def run_background(
        self, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        # `&` backgrounds inside the sandbox; the command returns immediately and the
        # process lives until the sandbox is killed in stop().
        inner = f"nohup {shlex.join(argv)} > {shlex.quote(log)} 2>&1 &"
        result = await self.run(["sh", "-c", inner], env)
        if result.exit_code != 0:
            raise SandboxError(f"e2b background launch failed: {result.stderr.strip()}")

    def _abs(self, path: str) -> str:
        # The filesystem API resolves relative paths against the user's home, not our workdir.
        if path.startswith("/"):
            return path
        return f"{self.config.workdir.rstrip('/')}/{path}"

    async def _read(self, path: str, max_bytes: int | None = None) -> bytes:
        if max_bytes is not None:
            return await super()._read(path, max_bytes)
        try:
            data = await self._sandbox.files.read(
                self._abs(path), format="bytes", user=self.config.user
            )
        except Exception as e:
            raise SandboxError(f"read {path!r}: {e}") from e
        return bytes(data)

    async def write(self, path: str, data: bytes) -> None:
        try:
            await self._sandbox.files.write(
                self._abs(path), data, user=self.config.user
            )
        except Exception as e:
            raise SandboxError(f"write {path!r}: {e}") from e

    def cleanup(self) -> None:
        # Synchronous atexit backstop (the async API can't run once the loop is gone): kill
        # the sandbox via the sync API. Idempotent — the async `stop` kills it on the normal
        # path, and killing a gone sandbox just returns False.
        sandbox, self._sandbox = self._sandbox, None
        if sandbox is not None:  # keep info.id available after teardown
            from e2b import Sandbox

            with contextlib.suppress(Exception):
                Sandbox.kill(sandbox_id=sandbox.sandbox_id)

    async def teardown(self) -> None:
        # Best-effort, idempotent teardown: kill the sandbox (the costly resource). Runs via
        # `stop`, shielded from cancellation. `_sandbox` — the atexit backstop's key — is
        # consumed only after the kill attempt, so a loop death mid-await still leaves the
        # backstop something to kill.
        sandbox = self._sandbox
        if sandbox is None:
            return
        try:
            await sandbox.kill()
        except Exception as e:  # noqa: BLE001 - provider teardown is best-effort
            logger.warning("e2b: failed to kill sandbox %s: %s", self.info.id, e)
        self._sandbox = None
