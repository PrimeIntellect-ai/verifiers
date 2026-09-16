"""Remote E2B sandbox runtime."""

import asyncio
import contextlib
import hashlib
import ipaddress
import json
import logging
import os
import re
import shlex
from collections.abc import AsyncIterator
from pathlib import PurePosixPath
from typing import Any, ClassVar, Literal
from urllib.parse import urlsplit

from pydantic import model_validator

from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import (
    BaseRuntimeInfo,
    ProgramResult,
    Runtime,
    RuntimeProcess,
)
from verifiers.v1.runtimes.limiters import creation_limiter
from verifiers.v1.utils.aio import run_shielded

logger = logging.getLogger(__name__)

_ALL_TRAFFIC = "0.0.0.0/0"
_DNS_RESOLVER = "8.8.8.8"
_DEFAULT_TEMPLATE = "base"
_DEFAULT_WORKDIR = "/home/user"
_DEFAULT_TIMEOUT = 24 * 60 * 60
_HOST_RE = re.compile(
    r"(?:\*\.)?(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)*"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
)


def _sdk() -> Any:
    try:
        import e2b
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "E2BRuntime requires the E2B SDK; install `verifiers[e2b]`."
        ) from e
    return e2b


def _ensure_auth() -> None:
    if os.getenv("E2B_API_KEY"):
        return
    raise SystemExit("not authenticated with e2b - set $E2B_API_KEY")


def _network_selector(
    rule: str, *, deny: bool = False, framework: bool = False
) -> str | None:
    """Translate one verifiers egress rule to the selector E2B can enforce."""
    value = rule.strip()
    try:
        return str(ipaddress.ip_address(value))
    except ValueError:
        pass
    try:
        return str(ipaddress.ip_network(value, strict=False))
    except ValueError:
        pass

    try:
        parsed = urlsplit(value if "://" in value else f"//{value}")
        port = parsed.port
    except ValueError as e:
        raise ValueError(f"invalid E2B egress rule {rule!r}: {e}") from e
    host = (parsed.hostname or "").lower().rstrip(".")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if framework and (
        host == "localhost" or (address is not None and address.is_loopback)
    ):
        return None
    if parsed.scheme not in ("", "http", "https"):
        raise ValueError(
            f"E2B egress rule {rule!r} is unsupported; use a DNS name, IP "
            "address, or CIDR"
        )
    if not framework and (parsed.scheme or port is not None):
        raise ValueError(
            f"E2B egress rule {rule!r} is unsupported; scheme- or port-specific "
            "rules would be broadened by E2B, so use a bare DNS name"
        )
    if port not in (None, 80, 443):
        raise ValueError(
            f"E2B framework route {rule!r} is unsupported; DNS filtering only "
            "covers ports 80 and 443"
        )
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"E2B egress rule {rule!r} must not contain credentials")
    if address is not None:
        return str(address)
    if not host or not _HOST_RE.fullmatch(host):
        raise ValueError(
            f"invalid E2B egress rule {rule!r}; expected a DNS name, IP address, "
            "CIDR, or HTTP(S) origin"
        )
    if deny:
        raise ValueError(
            f"E2B cannot deny a DNS name ({rule!r}); use an IP address or CIDR, "
            "or switch to an allowlist"
        )
    return host


def _required_selector(rule: str, *, deny: bool = False) -> str:
    selector = _network_selector(rule, deny=deny)
    assert selector is not None
    return selector


def _is_address(selector: str) -> bool:
    try:
        ipaddress.ip_network(selector, strict=False)
    except ValueError:
        return False
    return True


def _expanded(selector: str) -> list[str]:
    # Shared verifiers semantics make *.example.com match the apex too; E2B does not.
    return [selector, selector[2:]] if selector.startswith("*.") else [selector]


class E2BConfig(NetworkPolicyConfig):
    type: Literal["e2b"] = "e2b"
    image: str | None = None
    """Docker image to build into a project template on first use."""
    template: str | None = None
    """Existing E2B template to use when no image is set; defaults to ``base``."""
    workdir: str | None = None
    """Working directory override; None uses the task's workdir, or /home/user."""
    timeout: int = _DEFAULT_TIMEOUT
    """Sandbox lifetime in seconds (24h Pro maximum; Hobby maximum is 1h)."""
    creates_per_sec: float | None = 4.0
    """Pace sandbox creation across every local worker (None/<=0 disables it)."""

    @model_validator(mode="after")
    def _validate_e2b(self) -> "E2BConfig":
        if not 0 < self.timeout <= _DEFAULT_TIMEOUT:
            raise ValueError("E2B timeout must be between 1 and 86400 seconds")
        if not self.network_restricted:
            return self
        if self.allow == ["*"]:
            for rule in self.block:
                _required_selector(rule, deny=True)
        else:
            for rule in self.allow:
                _required_selector(rule)
        return self


class E2BRuntimeInfo(E2BConfig, BaseRuntimeInfo):
    resolved_template: str | None = None


async def _queue_stream(queue: asyncio.Queue[bytes | None]) -> AsyncIterator[bytes]:
    while (chunk := await queue.get()) is not None:
        yield chunk


class E2BProcess(RuntimeProcess):
    def __init__(
        self,
        handle,
        commands,
        stdout: asyncio.Queue[bytes | None],
        stderr: asyncio.Queue[bytes | None],
    ) -> None:
        self._handle = handle
        self._commands = commands
        self._stdout = stdout
        self._stderr = stderr
        self.stdout = _queue_stream(self._stdout)
        self.stderr = _queue_stream(self._stderr)
        self._exit_code: int | None = None
        self._error: Exception | None = None
        self._done = asyncio.create_task(self._watch())

    async def _watch(self) -> None:
        e2b = _sdk()
        try:
            result = await self._handle.wait()
            self._exit_code = result.exit_code
        except e2b.CommandExitException as e:
            if e.error == "signal: terminated":
                self._exit_code = -15
            elif e.error == "signal: killed":
                self._exit_code = -9
            else:
                self._exit_code = e.exit_code
        except Exception as e:  # noqa: BLE001 - surfaced to the caller by wait()
            self._error = e
        finally:
            self._stdout.put_nowait(None)
            self._stderr.put_nowait(None)

    async def write(self, data: bytes) -> None:
        try:
            await self._handle.send_stdin(data)
        except Exception as e:
            raise SandboxError(f"e2b process stdin failed: {e}") from e

    async def wait(self) -> int:
        await asyncio.shield(self._done)
        if self._error is not None:
            raise SandboxError(
                f"e2b process wait failed: {self._error}"
            ) from self._error
        assert self._exit_code is not None
        return self._exit_code

    async def terminate(self) -> None:
        if self._done.done():
            return
        e2b = _sdk()
        try:
            await self._commands.run(f"kill -TERM {self._handle.pid}", timeout=0)
        except e2b.CommandExitException:
            # The process may have exited between the done check and kill(1).
            return
        except Exception as e:
            raise SandboxError(f"e2b process terminate failed: {e}") from e

    async def kill(self) -> None:
        if self._done.done():
            return
        try:
            await self._handle.kill()
        except Exception as e:
            raise SandboxError(f"e2b process kill failed: {e}") from e


class E2BRuntime(Runtime):
    is_local: ClassVar[bool] = False
    info: E2BRuntimeInfo

    def __init__(self, config: E2BConfig, name: str | None = None) -> None:
        _ensure_auth()
        super().__init__(name)
        self.config = config.model_copy(
            update={"workdir": config.workdir or _DEFAULT_WORKDIR}
        )
        self.info = E2BRuntimeInfo(**self.config.model_dump())
        self._sandbox: Any | None = None

    def _require_sandbox(self) -> Any:
        if self._sandbox is None:
            raise SandboxError("e2b sandbox is not running")
        return self._sandbox

    def _abs(self, path: str) -> str:
        if path.startswith("/"):
            return path
        assert self.config.workdir is not None
        return str(PurePosixPath(self.config.workdir) / path)

    def _command(
        self, argv: list[str], env: dict[str, str]
    ) -> tuple[str, dict[str, str]]:
        process_env = self.process_env(env)
        path = process_env.pop("PATH", None)
        command = shlex.join(argv)
        if path is not None:
            # E2B starts a login shell, whose /etc/profile resets an injected PATH.
            command = f"export PATH={shlex.quote(path)}; exec {command}"
        return command, process_env

    async def _start_command(
        self,
        argv: list[str],
        env: dict[str, str],
        *,
        stdin: bool = False,
        on_stdout: Any = None,
        on_stderr: Any = None,
    ) -> Any:
        sandbox = self._require_sandbox()
        command, process_env = self._command(argv, env)
        handle = None

        async def start() -> None:
            nonlocal handle
            handle = await sandbox.commands.run(
                command,
                background=True,
                stdin=stdin,
                cwd=self.config.workdir,
                envs=process_env,
                timeout=0,
                on_stdout=on_stdout,
                on_stderr=on_stderr,
            )

        try:
            await run_shielded(start())
        except BaseException:
            if handle is not None:
                with contextlib.suppress(BaseException):
                    await run_shielded(handle.kill())
            raise
        assert handle is not None
        return handle

    async def _resolve_template(self) -> str:
        if self.config.image is None:
            return self.config.template or _DEFAULT_TEMPLATE
        spec = json.dumps(
            {"version": 1, "image": self.config.image},
            sort_keys=True,
            separators=(",", ":"),
        )
        name = f"vf-{hashlib.sha256(spec.encode()).hexdigest()[:20]}"
        e2b = _sdk()
        if await e2b.AsyncTemplate.alias_exists(name):
            return name
        logger.warning(
            "e2b: image %s has no cached template - building %s (first use may "
            "take several minutes)",
            self.config.image,
            name,
        )
        builder = e2b.AsyncTemplate().from_image(self.config.image)
        try:
            await e2b.AsyncTemplate.build(builder, name)
        except Exception:
            # Another worker may have won the first-use build race.
            if not await e2b.AsyncTemplate.alias_exists(name):
                raise
        return name

    async def start(self) -> None:
        e2b = _sdk()
        try:
            template = await self._resolve_template()
            async with (
                creation_limiter(self.config.creates_per_sec, "e2b-sandbox")
                or contextlib.nullcontext()
            ):
                await run_shielded(self._create_sandbox(e2b, template))
            self.info.resolved_template = template
            await self._require_sandbox().files.make_dir(self.config.workdir)
            logger.info("e2b: sandbox %s up (template=%s)", self.info.id, template)
        except Exception as e:
            raise SandboxError(f"e2b sandbox provisioning failed: {e}") from e

    async def _create_sandbox(self, e2b, template: str) -> None:
        self._sandbox = await e2b.AsyncSandbox.create(
            template,
            timeout=self.config.timeout,
            envs={key: value for key, value in self.env.items() if key != "PATH"},
            metadata={"verifiers-runtime": self.name},
        )
        self.info.id = self._sandbox.sandbox_id

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        e2b = _sdk()
        try:
            handle = await self._start_command(argv, env)
            try:
                result = await handle.wait()
            except e2b.CommandExitException as e:
                result = e
            except BaseException:
                with contextlib.suppress(BaseException):
                    await run_shielded(handle.kill())
                raise
        except Exception as e:
            raise SandboxError(f"e2b exec failed: {e}") from e
        return ProgramResult(
            exit_code=result.exit_code,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
        )

    async def open_process(
        self, argv: list[str], env: dict[str, str]
    ) -> RuntimeProcess:
        stdout: asyncio.Queue[bytes | None] = asyncio.Queue()
        stderr: asyncio.Queue[bytes | None] = asyncio.Queue()
        try:
            sandbox = self._require_sandbox()
            handle = await self._start_command(
                argv,
                env,
                stdin=True,
                on_stdout=lambda chunk: stdout.put_nowait(chunk.encode()),
                on_stderr=lambda chunk: stderr.put_nowait(chunk.encode()),
            )
            return E2BProcess(handle, sandbox.commands, stdout, stderr)
        except Exception as e:
            raise SandboxError(f"e2b live process failed to start: {e}") from e

    async def run_background(
        self, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        inner = f"nohup {shlex.join(argv)} > {shlex.quote(log)} 2>&1 < /dev/null &"
        result = await self.run(["sh", "-c", inner], {"PYTHONUNBUFFERED": "1", **env})
        if result.exit_code != 0:
            raise SandboxError(f"e2b background launch failed: {result.stderr.strip()}")

    async def _read(self, path: str, max_bytes: int | None = None) -> bytes:
        if max_bytes is not None:
            # The base shell path applies head(1) before any bytes cross the boundary.
            return await super()._read(self._abs(path), max_bytes)
        try:
            data = await self._require_sandbox().files.read(
                self._abs(path), format="bytes"
            )
            return bytes(data)
        except Exception as e:
            raise SandboxError(f"read {path!r}: {e}") from e

    async def write(self, path: str, data: bytes) -> None:
        try:
            await self._require_sandbox().files.write(self._abs(path), data)
        except Exception as e:
            raise SandboxError(f"write {path!r}: {e}") from e

    async def prepare_execution(self, routes: list[str] | None) -> None:
        if not self.network_restricted:
            return
        try:
            if routes is None:
                policy = {"allow_internet_access": True}
            else:
                framework: list[str] = []
                for route in routes:
                    selector = _network_selector(route, framework=True)
                    if selector is not None:
                        framework.extend(_expanded(selector))
                if self.config.allow == ["*"]:
                    allow = framework
                    deny = [
                        _required_selector(rule, deny=True)
                        for rule in self.config.block
                    ]
                else:
                    configured = [
                        selector
                        for rule in self.config.allow
                        for selector in _expanded(_required_selector(rule))
                    ]
                    allow = [*framework, *configured]
                    deny = [_ALL_TRAFFIC]
                allow = list(dict.fromkeys(allow))
                if allow and all(_is_address(selector) for selector in allow):
                    allow.append(_DNS_RESOLVER)
                policy = {"allow_out": allow, "deny_out": deny}
            await self._require_sandbox().update_network(policy)
        except Exception as e:
            raise SandboxError(f"e2b egress policy failed: {e}") from e
        logger.info(
            "e2b: egress policy applied on sandbox %s: %s", self.info.id, policy
        )

    async def expose(self, port: int) -> str | None:
        if not 1 <= port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        sandbox = self._sandbox
        if sandbox is None:
            return None
        return f"https://{sandbox.get_host(port)}"

    def cleanup(self) -> None:
        if self.info.id is None:
            return
        with contextlib.suppress(Exception):
            _sdk().Sandbox.kill(self.info.id)

    async def teardown(self) -> None:
        sandbox = self._sandbox
        if sandbox is None:
            return
        try:
            await sandbox.kill()
        except Exception as e:  # noqa: BLE001 - provider teardown is best-effort
            logger.warning("e2b: failed to kill sandbox %s: %s", self.info.id, e)
        finally:
            if self._sandbox is sandbox:
                self._sandbox = None
