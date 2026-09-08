"""Remote E2B sandbox runtime.

`expose` (sandbox port -> public URL) uses E2B's built-in port proxy (`get_host`), so a
host-side harness/framework can reach a tool server hosted in the sandbox. The reverse
direction (a program in the sandbox reaching a host service) is the shared host-side
`Tunnel` (interception.tunnel), not the runtime's concern.
"""

import asyncio
import codecs
import contextlib
import fcntl
import hashlib
import ipaddress
import json
import logging
import os
import shlex
import tempfile
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import PurePosixPath
from typing import ClassVar, Literal
from urllib.parse import urlsplit

from pydantic import Field, model_validator

from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.errors import SandboxError
from verifiers.v1.runtimes.base import (
    SERVICE_PORT,
    BaseRuntimeInfo,
    ProgramResult,
    Runtime,
    RuntimeProcess,
)
from verifiers.v1.runtimes.limiters import creation_limiter
from verifiers.v1.utils.aio import run_shielded

logger = logging.getLogger(__name__)

_TEMPLATE_SCHEMA = 1
_TEMPLATE_LOCK_DIR = os.path.join(tempfile.gettempdir(), "vf-e2b-template-locks")

# `run_program` holds a wait stream open for the whole rollout; on a long one it can drop
# (proxy idle, transient network). Reconnecting resumes waiting on the same pid — never a
# re-run — so a bounded number of attempts, backing off from `_WAIT_RECONNECT_BACKOFF`
# seconds and doubling per attempt, is safe.
_WAIT_RECONNECTS = 4
_WAIT_RECONNECT_BACKOFF = 0.25


def _template_resources(cpu: float, memory: float) -> tuple[int, int]:
    """Validate and convert the resource units shared with TaskData.resources."""
    cpu_value = float(cpu)
    if not cpu_value.is_integer():
        raise ValueError(f"E2B templates require whole CPU cores, got {cpu_value:g}")
    cpu_count = int(cpu_value)
    if cpu_count != 1 and cpu_count % 2:
        raise ValueError(
            f"E2B templates require 1 or an even number of CPU cores, got {cpu_count}"
        )

    memory_mb_value = float(memory) * 1024
    if not memory_mb_value.is_integer() or int(memory_mb_value) % 2:
        raise ValueError(
            "E2B template memory must resolve to an even whole number of MB, "
            f"got {memory:g} GB ({memory_mb_value:g} MB)"
        )
    return cpu_count, int(memory_mb_value)


def _validate_egress_rules(allow: list[str], block: list[str]) -> None:
    """Validate the two selector shapes supported by E2B's network API.

    Allow rules accept hostnames (including a leading ``*.``), IPs, and CIDRs. Deny
    rules only accept IPs and CIDRs; the provider rejects hostnames at request time.
    """
    for rule in block:
        try:
            ipaddress.ip_network(rule, strict=False)
        except ValueError:
            raise ValueError(
                f"E2B block rules must be IP addresses or CIDR blocks, got {rule!r}"
            ) from None

    for rule in allow:
        try:
            ipaddress.ip_network(rule, strict=False)
            continue
        except ValueError:
            pass

        hostname = rule.removeprefix("*.").removesuffix(".")
        labels = hostname.split(".")
        valid_hostname = 0 < len(hostname) <= 253 and all(
            0 < len(label) <= 63
            and label[0] != "-"
            and label[-1] != "-"
            and all(
                char.isascii() and (char.isalnum() or char == "-") for char in label
            )
            for label in labels
        )
        if valid_hostname:
            continue

        raise ValueError(
            "E2B allow rules must be hostnames, wildcard hostnames, IP addresses, "
            f"or CIDR blocks (no schemes, ports, or paths), got {rule!r}"
        )


def _egress_update(config: "E2BConfig", routes: list[str] | None) -> dict:
    """The complete egress policy for `update_network`, which replaces all rules
    atomically. None restores unrestricted egress for another trusted setup phase;
    otherwise `routes` (the interception and MCP endpoints) stay reachable alongside
    the configured allowlist. E2B resolves an `allow_out` entry over a same-host
    `deny_out` one, so an allowlist is expressed as allow + deny-everything."""
    if routes is None:
        return {"allow_internet_access": True}
    hosts = list(
        dict.fromkeys(
            host for route in routes if (host := urlsplit(route).hostname) is not None
        )
    )
    if config.allow == ["*"]:
        return {"allow_out": hosts, "deny_out": list(config.block)}
    entries = list(dict.fromkeys([*hosts, *config.allow]))
    if not entries:
        return {"allow_internet_access": False}
    return {"allow_out": entries, "deny_out": ["0.0.0.0/0"]}


class E2BConfig(NetworkPolicyConfig):
    type: Literal["e2b"] = "e2b"
    image: str = "python:3.11-slim"
    """Public Debian-based Docker image used to build a cached E2B template."""
    workdir: str = "/app"
    # TaskData.resources uses these units; non-default runtime config values take precedence.
    cpu: float = Field(default=2.0, ge=1)
    """CPU cores. E2B templates require 1 or an even whole number."""
    memory: float = Field(default=1.0, ge=0.125)
    """Memory in GB. The converted MB value must be a whole even number."""
    disk: float = Field(default=5.0, gt=0)
    """Advisory disk request in GB. E2B template builds have no disk-size knob, so this
    is accepted (so a task can declare it without a warning) but not enforced."""
    creates_per_sec: float | None = 1.0
    """Pace sandbox creation to this many per second, enforced host-wide across every
    env-server worker process (None/<= 0 disables it). The default fits E2B's base tier;
    Pro allows ~5/s and enterprise plans more — raise this to your plan's rate for
    high-concurrency evals or training."""

    @model_validator(mode="after")
    def _validate_template_resources(self) -> "E2BConfig":
        _template_resources(self.cpu, self.memory)
        if self.network_restricted:
            _validate_egress_rules(
                [rule for rule in self.allow if rule != "*"],
                [rule for rule in self.block if rule != "*"],
            )
        return self


class E2BRuntimeInfo(E2BConfig, BaseRuntimeInfo):
    template: str | None = None


async def _queue_stream(queue: asyncio.Queue[bytes | None]) -> AsyncIterator[bytes]:
    while (chunk := await queue.get()) is not None:
        yield chunk


class E2BProcess(RuntimeProcess):
    """Built in two steps: the queues (and their `on_*` callbacks) must exist before
    `commands.run` is called with them, and the handle only exists after — so construct
    first, then `attach` the started handle.

    Byte fidelity: the SDK hands the callbacks *text* (chunks decoded as UTF-8 with
    ``errors="replace"``), which is lossy for arbitrary bytes — but the decode goes
    through a per-handle incremental decoder that `open_process` swaps for latin-1
    before the program is allowed to write. Latin-1 maps bytes 1:1 onto U+00–U+FF, so
    ``chunk.encode("latin-1")`` recovers the exact bytes. Output that precedes the swap
    (login-shell profile noise) may still be UTF-8-decoded; it all lands before the
    wrapper's start marker and is dropped by the marker strip, so the lossy
    ``errors="replace"`` on the re-encode below can only ever touch discarded bytes."""

    def __init__(self, sandbox, marker: bytes) -> None:
        self._handle = None
        self._sandbox = sandbox
        self._marker = marker
        # Pre-marker buffer per stream; None once the marker has been consumed.
        self._prefix: dict[str, bytearray | None] = {
            "stdout": bytearray(),
            "stderr": bytearray(),
        }
        self._stdout_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._stderr_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.stdout = _queue_stream(self._stdout_queue)
        self.stderr = _queue_stream(self._stderr_queue)
        self._wait_task: asyncio.Task[int] | None = None

    def _payload(self, stream: str, chunk: str) -> bytes:
        """Recover the raw bytes of a chunk and drop everything through the marker."""
        data = chunk.encode("latin-1", errors="replace")
        pending = self._prefix[stream]
        if pending is None:
            return data
        pending.extend(data)
        start = pending.find(self._marker)
        if start < 0:
            return b""
        self._prefix[stream] = None
        return bytes(pending[start + len(self._marker) :])

    def on_stdout(self, chunk: str) -> None:
        if data := self._payload("stdout", chunk):
            self._stdout_queue.put_nowait(data)

    def on_stderr(self, chunk: str) -> None:
        if data := self._payload("stderr", chunk):
            self._stderr_queue.put_nowait(data)

    def attach(self, handle) -> None:
        self._handle = handle
        self._wait_task = asyncio.create_task(self._wait())

    async def _wait(self) -> int:
        # Importable here for sure: a live handle means the SDK import in `start` succeeded.
        from e2b import CommandExitException

        try:
            result = await self._handle.wait()
            return result.exit_code
        except CommandExitException as e:
            return e.exit_code
        except Exception as e:
            raise SandboxError(f"e2b live process failed: {e}") from e
        finally:
            self._stdout_queue.put_nowait(None)
            self._stderr_queue.put_nowait(None)

    async def write(self, data: bytes) -> None:
        try:
            await self._handle.send_stdin(data)
        except Exception as e:
            raise SandboxError(f"e2b live process stdin failed: {e}") from e

    async def wait(self) -> int:
        assert self._wait_task is not None
        return await self._wait_task

    async def terminate(self) -> None:
        if self._wait_task is not None and self._wait_task.done():
            return
        try:
            await self._sandbox.commands.run(
                f"kill -TERM {self._handle.pid}", user="root"
            )
        except Exception as e:
            if self._wait_task is None or not self._wait_task.done():
                raise SandboxError(f"e2b live process signal failed: {e}") from e

    async def kill(self) -> None:
        if self._wait_task is not None and self._wait_task.done():
            return
        try:
            await self._handle.kill()
        except Exception as e:
            if self._wait_task is None or not self._wait_task.done():
                raise SandboxError(f"e2b live process kill failed: {e}") from e


@asynccontextmanager
async def _template_lock(name: str):
    """Serialize a deterministic template build across local worker processes."""
    os.makedirs(_TEMPLATE_LOCK_DIR, exist_ok=True)
    path = os.path.join(_TEMPLATE_LOCK_DIR, f"{name}.lock")
    fd = await asyncio.to_thread(os.open, path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        await asyncio.to_thread(fcntl.flock, fd, fcntl.LOCK_EX)
        yield
    finally:
        await asyncio.to_thread(fcntl.flock, fd, fcntl.LOCK_UN)
        os.close(fd)


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

    def _template_name(self) -> str:
        definition = json.dumps(
            {
                "schema": _TEMPLATE_SCHEMA,
                "image": self.config.image,
                "workdir": self.config.workdir,
                "cpu": self.config.cpu,
                "memory": self.config.memory,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(definition.encode()).hexdigest()[:24]
        return f"vf-{digest}"

    async def _ensure_template(self) -> str:
        from e2b import AsyncTemplate, Template

        # Guaranteed valid by now — `__init__` revalidated the resolved config (task-level
        # resources arrive via model_copy(update=...), which skips validation) when it built
        # the info model. This call just converts to E2B's units.
        cpu_count, memory_mb = _template_resources(self.config.cpu, self.config.memory)
        name = self._template_name()
        async with _template_lock(name):
            if not await AsyncTemplate.exists(name):
                logger.info(
                    "e2b: building template %s from image %s", name, self.config.image
                )
                template = (
                    Template()
                    .from_image(self.config.image)
                    .set_workdir(self.config.workdir)
                )
                await AsyncTemplate.build(
                    template,
                    name,
                    cpu_count=cpu_count,
                    memory_mb=memory_mb,
                )
        return name

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

            async def _create() -> None:
                async with (
                    creation_limiter(self.config.creates_per_sec, "e2b-sandbox")
                    or contextlib.nullcontext()
                ):
                    # Created unrestricted: setup (uv installs, task staging) needs open
                    # egress; `prepare_execution` locks the policy down before the agent.
                    self._sandbox = await AsyncSandbox.create(
                        template,
                        timeout=24 * 60 * 60,  # Maximum lifetime of any sandbox.
                        metadata={"runtime": "verifiers", "name": self.name},
                    )
                    # The atexit backstop kills by id, so record it the moment the sandbox
                    # exists — the cancellation `run_shielded` re-raises can keep the code
                    # after it from ever running.
                    self.info.id = self._sandbox.sandbox_id

            # Finish an accepted create request even if the rollout is cancelled so the
            # surrounding provision_runtime finally can see and kill the sandbox.
            await run_shielded(_create())
            logger.info(
                "e2b: sandbox %s up (image=%s, template=%s)",
                self.info.id,
                self.config.image,
                template,
            )
            # The template bakes the workdir in, but a task-supplied workdir lands on a
            # cached template built for another; create it either way (make_dir is a no-op
            # on an existing directory).
            await self._sandbox.files.make_dir(self.config.workdir)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            raise SandboxError(f"e2b sandbox provisioning failed: {e}") from e

    async def prepare_execution(self, routes: list[str] | None) -> None:
        """Apply the host policy after setup; the update replaces all rules atomically."""
        if not self.network_restricted:
            return
        network = _egress_update(self.config, routes)
        try:
            await self._sandbox.update_network(network)
        except Exception as e:
            raise SandboxError(f"e2b egress policy failed: {e}") from e
        logger.info(
            "e2b: egress policy applied on sandbox %s (%s)", self.info.id, network
        )

    def _command(
        self, argv: list[str], env: dict[str, str]
    ) -> tuple[str, dict[str, str], str]:
        """The wrapped command, its env, and the per-call start marker.

        E2B commands run through a login shell (the SDK hardcodes ``bash -l -c``),
        which brings two hazards the wrapper fences off: profile initialization can
        print to stdout/stderr (polluting results on images whose /etc/profile.d has
        banners), and it re-derives PATH. The marker — echoed on both streams right
        before ``exec`` — lets consumers drop everything the shell produced; the
        caller-supplied PATH is restored after initialization so it has the same
        precedence as on other runtimes."""
        command_env = self.process_env(env)
        marker = f"vf-start-{uuid.uuid4().hex}"
        quoted = shlex.quote(marker)
        command = (
            f"printf '%s' {quoted}; printf '%s' {quoted} >&2; exec {shlex.join(argv)}"
        )
        if "PATH" in command_env:
            command_env["VF_RUNTIME_PATH"] = command_env.pop("PATH")
            command = f'export PATH="$VF_RUNTIME_PATH"; {command}'
        return command, command_env, marker

    @staticmethod
    def _after_marker(text: str, marker: str) -> str:
        """Drop login-shell noise: everything through the start marker."""
        _, sep, tail = text.partition(marker)
        return tail if sep else text

    async def run(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        from e2b import CommandExitException  # importable once `start` succeeded

        command, command_env, marker = self._command(argv, env)
        try:
            result = await self._sandbox.commands.run(
                command,
                envs=command_env,
                cwd=self.config.workdir,
                timeout=0,  # no command deadline; the agent's own timeouts govern
            )
        except CommandExitException as e:  # non-zero exit is a result, not a failure
            result = e
        except Exception as e:
            raise SandboxError(f"e2b exec failed: {e}") from e
        return ProgramResult(
            result.exit_code,
            self._after_marker(result.stdout or "", marker),
            self._after_marker(result.stderr or "", marker),
        )

    async def _finished_status(self, status_path: str) -> str | None:
        """The wrapper's recorded exit status, or None while the program still runs.
        Read rather than test existence: the shell truncates the file before printf
        fills it, so only non-empty content proves the program completed."""
        try:
            status = await self._sandbox.files.read(status_path)
        except Exception:  # noqa: BLE001 - unreadable means "not finished yet"
            return None
        return status.strip() or None

    async def run_program(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        """Run the rollout through a durable E2B process without ever replaying it."""
        prefix = f"/tmp/vf-program-{uuid.uuid4().hex}"
        stdout_path, stderr_path, status_path = (
            f"{prefix}.stdout",
            f"{prefix}.stderr",
            f"{prefix}.status",
        )
        wrapper = (
            'out=$1; err=$2; status=$3; shift 3; "$@" >"$out" 2>"$err"; '
            'rc=$?; printf "%s\\n" "$rc" >"$status"; exit 0'
        )
        command, command_env, _ = self._command(
            [
                "sh",
                "-c",
                wrapper,
                "vf-program",
                stdout_path,
                stderr_path,
                status_path,
                *argv,
            ],
            env,
        )
        try:
            handle = await self._sandbox.commands.run(
                command,
                background=True,
                envs=command_env,
                cwd=self.config.workdir,
                stdin=False,
                timeout=0,
            )
            # The status file, not the wait stream, is the source of truth: on any
            # stream failure — including a reconnect finding the pid already gone
            # because the program finished during the backoff — completion wins.
            status: str | None = None
            for attempt in range(_WAIT_RECONNECTS):
                try:
                    await handle.wait()
                    break
                except Exception as e:
                    status = await self._finished_status(status_path)
                    if status is not None:
                        break
                    if attempt == _WAIT_RECONNECTS - 1:
                        raise SandboxError(
                            f"e2b durable program connection failed: {e}"
                        ) from e
                    await asyncio.sleep(_WAIT_RECONNECT_BACKOFF * 2**attempt)
                    try:
                        handle = await self._sandbox.commands.connect(
                            handle.pid, timeout=0
                        )
                    except Exception:  # noqa: BLE001 - completion beats a dead stream
                        status = await self._finished_status(status_path)
                        if status is None:
                            raise SandboxError(
                                f"e2b durable program connection failed: {e}"
                            ) from e
                        break
            stdout, stderr = await asyncio.gather(
                self._sandbox.files.read(stdout_path),
                self._sandbox.files.read(stderr_path),
            )
            if status is None:
                status = (await self._sandbox.files.read(status_path)).strip()
            return ProgramResult(int(status), stdout, stderr)
        except asyncio.CancelledError:
            raise
        except SandboxError:
            raise
        except Exception as e:
            raise SandboxError(f"e2b durable program failed: {e}") from e
        finally:
            with contextlib.suppress(Exception):
                await asyncio.gather(
                    self._sandbox.files.remove(stdout_path),
                    self._sandbox.files.remove(stderr_path),
                    self._sandbox.files.remove(status_path),
                )

    async def open_process(
        self, argv: list[str], env: dict[str, str]
    ) -> RuntimeProcess:
        command, command_env, marker = self._command(argv, env)
        # Hold the program on a stdin gate so it cannot write before this method swaps
        # the handle's stream decoders to latin-1 (see E2BProcess) — only then is the
        # RuntimeProcess bytes contract (e.g. ACP's length-prefixed frames) honored.
        command = f"read -r vf_go_; {command}"
        process = E2BProcess(self._sandbox, marker.encode())
        try:
            handle = await self._sandbox.commands.run(
                command,
                background=True,
                envs=command_env,
                cwd=self.config.workdir,
                on_stdout=process.on_stdout,
                on_stderr=process.on_stderr,
                stdin=True,
                timeout=0,
            )
        except Exception as e:
            raise SandboxError(f"e2b live process failed to start: {e}") from e
        try:
            for name in ("_stdout_decoder", "_stderr_decoder"):
                if not hasattr(handle, name):
                    raise SandboxError(
                        "e2b SDK command handle no longer exposes its stream "
                        "decoders; cannot deliver byte-accurate process output"
                    )
                setattr(handle, name, codecs.getincrementaldecoder("latin-1")())
            process.attach(handle)
            await handle.send_stdin(b"\n")  # release the gate
        except BaseException:
            with contextlib.suppress(Exception):
                await handle.kill()
            raise
        return process

    async def run_background(
        self, argv: list[str], env: dict[str, str], log: str
    ) -> None:
        inner = f"nohup {shlex.join(argv)} > {shlex.quote(log)} 2>&1 &"
        result = await self.run(["sh", "-c", inner], env)
        if result.exit_code != 0:
            raise SandboxError(f"e2b background launch failed: {result.stderr.strip()}")

    def _abs(self, path: str) -> str:
        if path.startswith("/"):
            return path
        return f"{self.config.workdir.rstrip('/')}/{path}"

    async def _read(self, path: str) -> bytes:
        try:
            return bytes(
                await self._sandbox.files.read(self._abs(path), format="bytes")
            )
        except Exception as e:
            raise SandboxError(f"read {path!r}: {e}") from e

    async def write(self, path: str, data: bytes) -> None:
        target = self._abs(path)
        try:
            await self._sandbox.files.make_dir(str(PurePosixPath(target).parent))
            await self._sandbox.files.write(target, data)
        except Exception as e:
            raise SandboxError(f"write {path!r}: {e}") from e

    async def expose(self, port: int) -> str | None:
        # Publish a server hosted IN the sandbox: E2B proxies every sandbox port at a
        # public HTTPS URL derived locally from the sandbox id — nothing to declare up
        # front or tear down.
        if self._sandbox is None:
            return None
        return f"https://{self._sandbox.get_host(port)}"

    def cleanup(self) -> None:
        # Synchronous atexit backstop (the async API can't run once the loop is gone): kill
        # the sandbox by id via E2B's sync API so the paid resource isn't left to its
        # timeout. Idempotent — the async `stop` handles the normal path, and a second kill
        # is a no-op on E2B's side.
        if self._sandbox is None or self.info.id is None:
            return
        from e2b import Sandbox

        try:
            Sandbox.kill(self.info.id)
        except Exception:  # noqa: BLE001 - a later cleanup call may retry
            return
        self._sandbox = None

    async def teardown(self) -> None:
        # Best-effort, idempotent teardown on the normal path: kill the sandbox (the costly
        # resource) via the async API. Runs via `stop`, shielded from cancellation, so it
        # fires on success, error, and Ctrl-C. `_sandbox` — the atexit backstop's key — is
        # consumed only after the kill attempt: if loop death (second Ctrl-C) truncates the
        # await, the backstop can still kill by id.
        sandbox = self._sandbox
        if sandbox is None:
            return
        try:
            await sandbox.kill()
        except Exception as e:  # noqa: BLE001 - provider teardown is best-effort
            logger.warning("e2b: failed to kill sandbox %s: %s", self.info.id, e)
            return
        self._sandbox = None
