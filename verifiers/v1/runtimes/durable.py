"""Durable operations on a live `Runtime`. `Box.run` files a command's output and exit
code on the box before any reply crosses the transport, so a reply the platform lost is
read back and a command the box never ran is sent once more (held through a typed
`SandboxUnavailableError` for `outage_s`: an exec is not idempotent, so its retry
belongs here, beside the read-back; a read or write is retried by the runtime itself,
`PrimeConfig.outage_budget_s`); an upload the platform dropped is sent again; a large
file is read in parts. Every command has a bound (`op_timeout` when the caller gives
none). Any fault the layer gives up on is an `InfraError` with the typed fault as its
cause; an operation that hangs is told apart by one probe of the box (`hung`: the box
answers, the command's own hang; else the platform's)."""

import asyncio
import logging
import shlex
import time
import uuid
from collections.abc import Callable, Mapping
from typing import Any

from verifiers.v1.errors import (
    RolloutError,
    SandboxError,
    SandboxGoneError,
    SandboxNotFoundError,
    SandboxTimeoutError,
    SandboxUnavailableError,
)
from verifiers.v1.runtimes.base import ProgramResult, Runtime
from verifiers.v1.utils.aio import check_cancelled

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
"""Seconds a held exec waits before it is sent again; each further wait doubles."""
BACKOFF_MAX = 30.0
"""The longest wait between two sends of one held exec."""
OP_TIMEOUT = 900
"""Seconds one read, write or command of a box may take before it is abandoned as
hung; the bound of a command run without one."""
OUTAGE_BUDGET = 600
"""Seconds an exec whose reply the platform lost is held and sent again before the
error reaches its caller."""
PROBE_S = 30.0
"""Seconds the box gets to answer the probe at a hung operation."""
FILED_READS = 3
"""Times a file an exec just filed is read before a short or missing answer is final."""


class InfraError(RolloutError):
    """The machinery's own fault around a box: an operation the durable layer gave up on
    (its outage budget spent, a hang past its bound, a transport that failed twice),
    with the typed fault as its cause. `hung`: the operation hung while the box answered
    a probe, so the hang is the command's own (an infinite loop, a server left in the
    foreground), not the platform's."""

    def __init__(self, message: str = "", *, hung: bool = False) -> None:
        super().__init__(message)
        self.hung = hung


class Box:
    """Durable operations on a runtime: commands under bash in `workdir`, files by path;
    `home` is the box's own directory (its `tmp` holds the filed outputs)."""

    def __init__(
        self,
        runtime: Runtime,
        workdir: str,
        *,
        home: str = "/tmp/vf-box",
        op_timeout: float = OP_TIMEOUT,
        outage_s: float = OUTAGE_BUDGET,
    ):
        self.runtime = runtime
        self.workdir = workdir
        self.home = home
        self.op_timeout = op_timeout
        self.outage_s = outage_s

    @property
    def id(self) -> str:
        return str(self.runtime.info.id)

    @property
    def tmp(self) -> str:
        return f"{self.home}/tmp"

    async def _op(self, what: str, seconds: float | None, op: Callable[[], Any]) -> Any:
        """One operation within `seconds`; a fault is an `InfraError`, the typed fault
        its cause (a hang past `seconds` probed, `_hung`); none is started on a runtime
        whose teardown began."""
        check_cancelled()
        if self.runtime.stopped:
            raise InfraError(f"{what} refused: box {self.id} is closing")
        try:
            async with asyncio.timeout(seconds) as clock:
                return await op()
        except TimeoutError as e:
            raise await self._hung(f"{what} hung past {seconds}s") from e
        except SandboxError as e:
            check_cancelled()
            if clock.expired():
                raise await self._hung(f"{what} hung past {seconds}s") from e
            raise InfraError(f"{what} failed: {e}") from e

    async def _hung(self, message: str) -> InfraError:
        """The fault of an operation that hung, after one probe of the box within
        `PROBE_S`: a box that answers makes the hang the command's own (`hung`); one
        that is gone, refuses or stays silent makes it the platform's."""
        check_cancelled()
        try:
            async with asyncio.timeout(PROBE_S):
                answers = await self.runtime.alive()
        except (SandboxError, TimeoutError):
            answers = False
        if answers:
            return InfraError(f"{message} while the box answers", hung=True)
        return InfraError(f"{message} and the box does not answer; abandoned")

    async def run(
        self,
        command: str,
        timeout: int | None = None,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> tuple[int, str]:
        """Run `command` under bash in `cwd` (the workdir by default) with `env` on top of
        the runtime's, within `timeout` seconds (`op_timeout` by default); exit code and
        combined output, both filed on the box first so a reply the platform lost is
        read back and a command it never ran is sent again. An exec that failed `unavailable` may have run (its status
        poll dropped) or never reached the box: the filed exit code decides; one whose
        status poll timed out waits for the filed exit code up to the command's bound
        and is never sent again; a command past `timeout` is killed and says so."""
        timeout = timeout if timeout is not None else int(self.op_timeout)
        output = f"{self.tmp}/out-{uuid.uuid4().hex}"
        q = shlex.quote
        wrapped = (
            f"mkdir -p {q(self.tmp)}; export TMPDIR={q(self.tmp)}; find {q(self.tmp)} -maxdepth 1 -name 'out-*' -mmin +30 -delete; "
            f"(cd {q(cwd or self.workdir)} && ( {command}\n)) > {q(output)} 2>&1; "
            f"rc=$?; echo $rc > {q(output + '.rc')}; size=$(stat -c %s {q(output)}); printf '%s\\n' \"$size\"; "
            f'if [ "$size" -le {INLINE_OUTPUT} ]; then cat {q(output)}; fi; exit $rc'
        )
        argv = ["timeout", "-k", "30", str(timeout), "bash", "-c", wrapped]
        bound = timeout + HANG_GRACE
        until = time.monotonic() + bound
        delay, deadline, sent = BACKOFF_START, None, 0
        while True:
            sent += 1
            lost = polled = None
            try:
                res = await self._op(
                    "exec", bound, lambda: self.runtime.run(argv, dict(env or {}))
                )
            except InfraError as error:
                if isinstance(error.__cause__, SandboxUnavailableError):
                    lost = error.__cause__
                elif isinstance(error.__cause__, SandboxTimeoutError):
                    polled = error.__cause__
                else:
                    raise
                res = ProgramResult(exit_code=-1, stdout="", stderr="")
            size, separator, out = res.stdout.partition("\n")
            rc = res.exit_code
            if separator and size.isdigit():
                if int(size) > INLINE_OUTPUT:
                    out = (await self.read_big(output, int(size))).decode(
                        "utf-8", errors="replace"
                    )
                break
            if rc == 124:
                filed = await self.read(output)
                out = filed.decode("utf-8", errors="replace") if filed else res.stdout
                break
            filed = await self._filed(output, until if polled is not None else None)
            if filed is not None:
                rc, out = filed
                break
            if polled is not None:
                raise await self._hung(
                    f"exec: the status poll timed out and no exit code was filed within {bound}s: {polled}"
                ) from polled
            if lost is not None:
                # the box never ran it: held and sent again within the outage budget
                deadline = deadline or time.monotonic() + self.outage_s
                if time.monotonic() + delay > deadline:
                    raise InfraError(f"exec failed: {lost}") from lost
                logger.warning(
                    "exec on box %s held %gs, the platform unavailable: %s",
                    self.id,
                    delay,
                    lost,
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, BACKOFF_MAX)
            elif sent == 2:
                raise InfraError(
                    f"command output transport failed: {res.stdout} {res.stderr}"
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
                out = await self.read(output)
                return int(ran), (out or b"").decode("utf-8", errors="replace")
            if until is None or time.monotonic() >= until:
                return None
            await asyncio.sleep(POLL_S)

    async def read(self, path: str, size: int | None = None) -> bytes | None:
        """The file's bytes, or None when the box answered that it has no such path; a
        box that is gone (a `SandboxGoneError`: its placement lost, terminated) is the
        fault, never None (r8 20:55: a lost box read as "no such path" for a file that
        stood, and the caller polled a box that would never answer). With `size`, the
        bytes an exec just filed (its `stat`'s): a read that answers short or None is
        asked again, `FILED_READS` in all `POLL_S` apart, before None is the answer (r8
        01:51: a bundle read short unbundled as `premature end of pack file`)."""
        for tried in range(1, (FILED_READS if size is not None else 1) + 1):
            try:
                data = await self._op(
                    f"read {path}", self.op_timeout, lambda: self.runtime.read(path)
                )
            except InfraError as e:
                if not isinstance(e.__cause__, SandboxNotFoundError) or isinstance(
                    e.__cause__, SandboxGoneError
                ):
                    raise
                data = None
            if size is None or (data is not None and len(data) == size):
                return data
            if tried < FILED_READS:
                await asyncio.sleep(POLL_S)
        return None

    async def read_big(self, path: str, size: int) -> bytes:
        """A file of `size` bytes, in parts when it is larger than one transport call
        carries; `no such path` only when the box answered so (a box that is gone is
        `read`'s own fault), and a part is never asked twice of a box that is gone."""
        if size <= READ_PART:
            data = await self.read(path)
            if data is None:
                raise InfraError(f"read {path} failed: no such path")
            return data
        rc, out = await self.run(
            f"rm -f {path}.part.* && split -b {READ_PART} -d -a 4 {path} {path}.part.",
            cwd="/",
        )
        if rc != 0:
            raise InfraError(f"split of {path} failed: {out}")
        data = bytearray()
        try:
            for i in range((size + READ_PART - 1) // READ_PART):
                part = f"{path}.part.{i:04d}"
                try:
                    data += await self.read(part) or b""
                except InfraError as error:
                    if isinstance(error.__cause__, SandboxGoneError):
                        raise
                    logger.warning("read of %s failed (%s); retrying once", part, error)
                    await asyncio.sleep(2)
                    data += await self.read(part) or b""
        finally:
            try:
                await self.run(f"rm -f {path}.part.*", cwd="/")
            except InfraError as error:
                logger.warning("Temporary chunk cleanup failed for %s: %s", path, error)
        return bytes(data)

    async def write_bytes(self, path: str, data: bytes) -> None:
        """Write `data` at `path`; an upload the platform drops without a reason is sent
        once more, never to a box that is gone."""
        for attempt in (1, 2):
            try:
                return await self._op(
                    f"write {path}",
                    self.op_timeout,
                    lambda: self.runtime.write(path, data),
                )
            except InfraError as error:
                if attempt == 2 or isinstance(error.__cause__, SandboxNotFoundError):
                    raise
                await asyncio.sleep(3)

    async def write(self, path: str, content: str) -> None:
        await self.write_bytes(path, content.encode())


__all__ = ["Box", "InfraError"]
