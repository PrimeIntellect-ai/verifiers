"""Durable operations on a live `Runtime`. `Box.run` files a command's output and exit
code on the box before any reply crosses the transport, so a reply the platform lost is
read back and a command the box never ran is sent once more; an upload the platform
dropped is sent again; a large file is read in parts. A sandbox outage (a typed
`SandboxUnavailableError`: no route to the box, the routing catalog down, a paused box
being placed, the API or the file gateway dropping the connection) holds the operation
on the run-wide `Platform` and retries it, never the caller's fault; any other fault is
an `InfraError`. `provision_box` provisions through the same hold, within a bound."""

import asyncio
import contextlib
import logging
import shlex
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from types import SimpleNamespace
from typing import Any, ClassVar

from verifiers.v1.errors import (
    RolloutError,
    SandboxError,
    SandboxNotFoundError,
    SandboxTimeoutError,
    SandboxUnavailableError,
)
from verifiers.v1.runtimes.base import Runtime

logger = logging.getLogger(__name__)

HANG_GRACE = 300
"""Seconds past a command's own timeout before its exec is abandoned as hung on the
platform's side."""
INLINE_OUTPUT = 2_000_000
"""Bytes of output an exec reply carries inline; a larger output is read back from the
box in parts."""
READ_PART = 8_000_000
"""Bytes one transport read carries; a larger file is split on the box and read part by
part."""
POLL_S = 3.0
"""Seconds between two reads of the exit code a command files while its status poll
timed out."""
BACKOFF_START = 1.0
"""Seconds an operation that met the outage waits before its first retry; each further
wait doubles."""
BACKOFF_MAX = 30.0
"""The longest wait between two retries of one held operation."""
NOTICE_INTERVAL = 60.0
"""Seconds between two log lines about one outage."""
OP_TIMEOUT = 900
"""Seconds one read or write of a box may take before it is abandoned as hung."""
PROVISION_WAIT = 480
"""Seconds a box may take to be provisioned before the attempt is a fault."""
OUTAGE_BUDGET = 600
"""Seconds an operation holds and retries on the platform's outage before the error
reaches its caller."""


class InfraError(RolloutError):
    """The machinery's own fault around a box: an operation the durable layer gave up on
    (its outage budget spent, a hang past its bound, a transport that failed twice),
    with the typed fault as its cause; `retryable` False marks a subclass that is a
    verdict about the work, never a transient fault."""

    retryable: ClassVar[bool] = True


def cancel_requested() -> bool:
    """Whether the current task holds a cancellation no handler has answered: the SDK
    stack swallows one (connectrpc turns a CancelledError inside a request into
    ConnectError CANCELED, which arrives as a SandboxError) and the task would otherwise
    retry, reopen or replace a box while the run is ending."""
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


def check_cancelled() -> None:
    """Raise the cancellation the current task holds, if any (`cancel_requested`)."""
    if cancel_requested():
        raise asyncio.CancelledError()


def _setting(value: float | Callable[[], float]) -> float:
    """A setting given as a number or as a callable read when it is needed (a run's
    settings file may change)."""
    if isinstance(value, (int, float)):
        return float(value)
    return float(value())


class Platform:
    """The sandbox platform's availability, run-wide: an operation that met an outage
    (`SandboxUnavailableError`) holds here and retries itself for `budget` seconds (a
    number, or a callable read at each hold), the wait logged at most once per
    `NOTICE_INTERVAL`; `answered` ends the outage."""

    def __init__(
        self,
        budget: float | Callable[[], float] = OUTAGE_BUDGET,
        log: Callable[[str], None] = print,
    ):
        self.budget = budget
        self.log = log
        self.since: float | None = None
        """`time.time()` the current outage began; None while the platform answers."""
        self.waiting = 0
        """Operations held here right now."""
        self._noticed = 0.0

    async def hold(self, hold: dict, error: SandboxError) -> None:
        """Hold one operation once more (`hold` keeps its delay and deadline; `what`
        names it, `box` the box), or raise past the budget."""
        if "deadline" not in hold:
            hold.update(
                delay=BACKOFF_START, deadline=time.monotonic() + _setting(self.budget)
            )
        if time.monotonic() + hold["delay"] > hold["deadline"]:
            raise InfraError(f"{hold['what']} failed: {error}") from error
        if self.since is None:
            self.since = time.time()
            self._noticed = 0.0
        if time.monotonic() - self._noticed >= NOTICE_INTERVAL:
            self._noticed = time.monotonic()
            began = time.strftime("%H:%M:%S", time.gmtime(self.since))
            where = f" on box {hold['box'].id}" if hold.get("box") is not None else ""
            self.log(
                f"[platform] sandboxes unavailable since {began}; {self.waiting + 1} operations waiting, "
                f"this one {hold['what']}{where}: {error}"
            )
        self.waiting += 1
        try:
            await asyncio.sleep(hold["delay"])
        finally:
            self.waiting -= 1
        hold["delay"] = min(hold["delay"] * 2, BACKOFF_MAX)

    def answered(self) -> None:
        """An operation succeeded: the outage, if one was on, is over."""
        if self.since is not None:
            self.log(
                f"[platform] sandboxes available again after {time.time() - self.since:.0f}s"
            )
            self.since = None


PLATFORM = Platform()
"""The default hold: one per process; a run with its own budget or log makes its own and
hands it to its boxes."""


class Box:
    """Durable operations on a runtime: commands under bash in `workdir`, files by path;
    `home` is the box's own directory (its `tmp` holds the filed outputs); `placed`
    remembers what was written where, for callers that place files by content."""

    def __init__(
        self,
        runtime: Runtime,
        workdir: str,
        *,
        home: str = "/tmp/vf-box",
        platform: Platform = PLATFORM,
        op_timeout: float | Callable[[], float] = OP_TIMEOUT,
    ):
        self.runtime = runtime
        self.workdir = workdir
        self.home = home
        self.platform = platform
        self.op_timeout = op_timeout
        self.placed: dict[str, tuple[str, bool]] = {}

    @property
    def id(self) -> str:
        return str(self.runtime.info.id)

    @property
    def tmp(self) -> str:
        return f"{self.home}/tmp"

    @property
    def closing(self) -> bool:
        """Whether the runtime's teardown has begun: no operation is started on such a
        box."""
        return bool(getattr(self.runtime, "stopped", False))

    async def _op(
        self,
        what: str,
        seconds: float | None,
        op: Callable[[], Any],
        *,
        holds: bool = True,
    ) -> Any:
        """One operation; a platform outage holds and retries it while `holds`, any
        other fault is an `InfraError`."""
        hold = {"what": what, "box": self}
        while True:
            check_cancelled()
            if self.closing:
                raise InfraError(f"{what} refused: box {self.id} is closing")
            clock = None
            try:
                async with asyncio.timeout(seconds) as clock:
                    result = await op()
            except TimeoutError as e:
                raise InfraError(
                    f"{what} hung past {seconds}s (platform-side); abandoned"
                ) from e
            except (SandboxError, OSError) as e:
                check_cancelled()
                if clock is not None and clock.expired():
                    raise InfraError(
                        f"{what} hung past {seconds}s (platform-side); abandoned"
                    ) from e
                if not holds or not isinstance(e, SandboxUnavailableError):
                    raise InfraError(f"{what} failed: {e}") from e
                await self.platform.hold(hold, e)
                continue
            self.platform.answered()
            return result

    async def run(
        self, command: str, timeout: int | None = None, cwd: str | None = None
    ) -> tuple[int, str]:
        """Run `command` under bash in `cwd` (the workdir by default); exit code and
        combined output, both filed on the box first so a reply the platform lost is
        read back and a command it never ran is sent again. An exec that failed
        `unavailable` may have run (its status poll dropped) or never reached the box:
        the filed exit code decides; one whose status poll timed out waits for the filed
        exit code up to the command's bound and is never sent again; a command past
        `timeout` is killed and says so."""
        output = f"{self.tmp}/out-{uuid.uuid4().hex}"
        q = shlex.quote
        wrapped = (
            f"mkdir -p {q(self.tmp)}; export TMPDIR={q(self.tmp)}; find {q(self.tmp)} -maxdepth 1 -name 'out-*' -mmin +30 -delete; "
            f"(cd {q(cwd or self.workdir)} && ( {command}\n)) > {q(output)} 2>&1; "
            f"rc=$?; echo $rc > {q(output + '.rc')}; size=$(stat -c %s {q(output)}); printf '%s\\n' \"$size\"; "
            f'if [ "$size" -le {INLINE_OUTPUT} ]; then cat {q(output)}; fi; exit $rc'
        )
        argv = ["bash", "-c", wrapped]
        if timeout is not None:
            argv = ["timeout", "-k", "30", str(timeout), *argv]
        bound = timeout + HANG_GRACE if timeout is not None else None
        until = time.monotonic() + (bound if bound is not None else HANG_GRACE)
        hold, sent = {"what": "exec", "box": self}, 0
        while True:
            sent += 1
            lost = polled = None
            try:
                res = await self._op(
                    "exec", bound, lambda: self.runtime.run(argv, {}), holds=False
                )
            except InfraError as error:
                if isinstance(error.__cause__, SandboxUnavailableError):
                    lost = error.__cause__
                elif isinstance(error.__cause__, SandboxTimeoutError):
                    polled = error.__cause__
                else:
                    raise
                res = SimpleNamespace(exit_code=None, stdout="", stderr="")
            size, separator, out = (res.stdout or "").partition("\n")
            rc = res.exit_code
            if separator and size.isdigit():
                if int(size) > INLINE_OUTPUT:
                    out = (await self.read_big(output, int(size))).decode(
                        "utf-8", errors="replace"
                    )
                break
            if rc == 124:
                out = await self.read(output) or res.stdout or ""
                break
            filed = await self._filed(output, until if polled is not None else None)
            if filed is not None:
                rc, out = filed
                break
            if polled is not None:
                raise InfraError(
                    f"exec: the status poll timed out and no exit code was filed within {bound or HANG_GRACE}s: {polled}"
                ) from polled
            if lost is not None:
                await self.platform.hold(hold, lost)
            elif sent == 2:
                raise InfraError(
                    f"command output transport failed: {res.stdout or ''} {res.stderr or ''}"
                )
            else:
                await asyncio.sleep(3)
        if res.stderr:
            out += ("\n" if out else "") + res.stderr
        if rc == 124:
            out += (
                "\n" if out else ""
            ) + f"<command timed out after {timeout}s and was killed>"
        return rc, out

    async def _filed(self, output: str, until: float | None) -> tuple[int, str] | None:
        """The exit code and output a command filed at `output`, or None while the box
        holds no exit code; with `until` the box is asked again every `POLL_S` seconds
        up to it."""
        while True:
            ran = await self.read(output + ".rc")
            if ran is not None and ran.strip().isdigit():
                return int(ran.strip()), await self.read(output) or ""
            if until is None or time.monotonic() >= until:
                return None
            await asyncio.sleep(POLL_S)

    async def read(self, path: str) -> str | None:
        """The file's text, or None when the box has no such path."""
        try:
            data = await self._op(
                f"read {path}",
                _setting(self.op_timeout),
                lambda: self.runtime.read(path),
            )
        except InfraError as e:
            if isinstance(e.__cause__, SandboxNotFoundError):
                return None
            raise
        return data.decode("utf-8", errors="replace")

    async def read_bytes(self, path: str) -> bytes:
        return await self._op(
            f"read {path}", _setting(self.op_timeout), lambda: self.runtime.read(path)
        )

    async def read_big(self, path: str, size: int) -> bytes:
        """A file of `size` bytes, in parts when it is larger than one transport call
        carries."""
        if size <= READ_PART:
            return await self.read_bytes(path)
        rc, out = await self.run(
            f"rm -f {path}.part.* && split -b {READ_PART} -d -a 4 {path} {path}.part.",
            cwd="/",
        )
        if rc != 0:
            raise InfraError(f"split of {path} failed: {out}")
        data = bytearray()
        try:
            for i in range((size + READ_PART - 1) // READ_PART):
                try:
                    data += await self.read_bytes(f"{path}.part.{i:04d}")
                except InfraError as error:
                    logger.warning(
                        "Reading part %04d of %s failed (%s); retrying once",
                        i,
                        path,
                        error,
                    )
                    await asyncio.sleep(2)
                    data += await self.read_bytes(f"{path}.part.{i:04d}")
        finally:
            try:
                await self.run(
                    f"rm -f {path}.part.*",
                    timeout=int(_setting(self.op_timeout)),
                    cwd="/",
                )
            except InfraError as error:
                logger.warning("Temporary chunk cleanup failed for %s: %s", path, error)
        return bytes(data)

    async def write_bytes(self, path: str, data: bytes) -> None:
        """Write `data` at `path`; an upload the platform drops without a reason is sent
        once more."""
        for attempt in (1, 2):
            try:
                return await self._op(
                    f"write {path}",
                    _setting(self.op_timeout),
                    lambda: self.runtime.write(path, data),
                )
            except InfraError:
                if attempt == 2:
                    raise
                await asyncio.sleep(3)

    async def write(self, path: str, content: str, mode: str = "644") -> None:
        await self.write_bytes(path, content.encode())
        if mode != "644":
            rc, out = await self.run(f"chmod {mode} {shlex.quote(path)}", cwd="/")
            if rc != 0:
                raise InfraError(f"chmod {path} failed: {out}")


def _leave(cm: AbstractAsyncContextManager, entering: asyncio.Task) -> None:
    """Abandon a provisioning still in flight; a box it made anyway is closed after
    it."""

    def ended(task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception() is None:
            asyncio.ensure_future(cm.__aexit__(None, None, None)).add_done_callback(
                _retrieve
            )

    entering.cancel()
    entering.add_done_callback(ended)


def _retrieve(task: asyncio.Task) -> None:
    if not task.cancelled():
        task.exception()


Provisioning = Callable[[], AbstractAsyncContextManager[Runtime]]
"""What provisions one box: a fresh provisioning context each time it is called
(`lambda: agent.provision(task, reuse=key)`, `lambda: provision_runtime(config)`)."""


async def provision_box(
    open: Provisioning,
    *,
    wait: float | Callable[[], float] = PROVISION_WAIT,
    platform: Platform = PLATFORM,
) -> tuple[AbstractAsyncContextManager[Runtime], Runtime]:
    """A box from `open` within `wait` seconds: the provisioning context and the
    runtime. A platform outage, or a create that timed out before any box existed (the
    creation limiter refusing a backlog), holds on `platform` and tries again; past the
    hold's budget, a box that does not come within the bound, or any other sandbox
    fault, is an `InfraError`; a box that comes late is closed."""
    seconds, hold = _setting(wait), {"what": "box provisioning"}
    while True:
        cm = open()
        entering = asyncio.ensure_future(cm.__aenter__())
        try:
            await asyncio.wait_for(asyncio.shield(entering), seconds)
        except BaseException as e:
            if not entering.done():
                _leave(cm, entering)
                check_cancelled()
                raise InfraError(
                    f"box provisioning failed: no box within {seconds:g}s"
                ) from e
        try:
            runtime = entering.result()
        except (SandboxUnavailableError, SandboxTimeoutError) as e:
            check_cancelled()
            await platform.hold(hold, e)
            continue
        except SandboxError as e:
            check_cancelled()
            raise InfraError(f"box provisioning failed: {e}") from e
        platform.answered()
        return cm, runtime


@asynccontextmanager
async def provisioned(
    open: Provisioning,
    *,
    wait: float | Callable[[], float] = PROVISION_WAIT,
    platform: Platform = PLATFORM,
) -> AsyncIterator[Runtime]:
    """`provision_box` as a context: the runtime for the block, its provisioning context
    left as `open` decides (a pool lease parks the box, a plain provisioning closes
    it)."""
    cm, runtime = await provision_box(open, wait=wait, platform=platform)
    async with contextlib.AsyncExitStack() as stack:
        stack.push_async_exit(cm)
        yield runtime


@asynccontextmanager
async def bare_box(
    config,
    *,
    workdir: str,
    home: str = "/tmp/vf-box",
    wait: float | Callable[[], float] = PROVISION_WAIT,
    platform: Platform = PLATFORM,
    op_timeout: float | Callable[[], float] = OP_TIMEOUT,
) -> AsyncIterator[Box]:
    """A box of the runtime `config` for the machinery alone (no agent, no pool): the
    `Box` for the block, closed after."""
    from verifiers.v1.runtimes import provision_runtime

    async with provisioned(
        lambda: provision_runtime(config), wait=wait, platform=platform
    ) as runtime:
        yield Box(runtime, workdir, home=home, platform=platform, op_timeout=op_timeout)


__all__ = [
    "PLATFORM",
    "Box",
    "InfraError",
    "Platform",
    "Provisioning",
    "bare_box",
    "cancel_requested",
    "check_cancelled",
    "provision_box",
    "provisioned",
]
