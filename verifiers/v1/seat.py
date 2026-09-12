"""One agent's turn executor over the durable layer. A `Seat` is an `Agent` (its
harness, model and runtime policy over a shared interception and box pool): `turn`
leases a box (a parked one under the opening's lease, else fresh), prepares it, runs one
interaction and hands the box to the caller's harvest before it is parked; `session`
keeps one interaction open across turns fed from outside (the transcript is the agent's
memory), reopened by the caller after a `Fault`. `Session.turn(message)` is one user
turn with the one nudge when the box stands unchanged after a reply without a call and
the fault rule: a permanent provider refusal ends the run (`ModelConfigurationError`), a
transient provider or harness death is a `Fault` unless the opening says a died
interaction is the agent's evidence. `LiveTrace` rewrites an open interaction's trace to
a file while it changes."""

import asyncio
import contextlib
import json
import logging
import os
import tempfile
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Self, TypeVar
from urllib.parse import urlsplit, urlunsplit

from verifiers.v1.agent import Agent, Interaction, Segment
from verifiers.v1.errors import HarnessError, RolloutError, SandboxError
from verifiers.v1.runtimes.base import Runtime
from verifiers.v1.runtimes.durable import (
    PLATFORM,
    PROVISION_WAIT,
    Box,
    InfraError,
    Platform,
    check_cancelled,
    provisioned,
)
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)
T = TypeVar("T")


# --- the fault rule


class ModelConfigurationError(RolloutError):
    """A permanent provider refusal (bad credentials, a forbidden model); it ends the
    run."""

    retryable: ClassVar[bool] = False

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class Fault(Exception):
    """One turn's infra fault: `detail` and `fresh` (the box is gone)."""

    def __init__(self, detail: str, *, fresh: bool):
        super().__init__(detail)
        self.detail, self.fresh = detail, fresh


def public_url(url: str) -> str:
    """The endpoint without its user information and query."""
    parts = urlsplit(url)
    if not (parts.scheme and parts.netloc):
        return url
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    if parts.port:
        host += f":{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def describe(error) -> str:
    """`type: message` for a trace error or an exception."""
    kind = getattr(error, "type", None) or type(error).__name__
    return f"{kind}: {getattr(error, 'message', None) or error}"


def provider_status(error) -> int | None:
    """The typed HTTP status of a provider error, None when it carries none."""
    status = getattr(error, "status_code", None)
    return status if isinstance(status, int) else None


def is_provider_error(error) -> bool:
    return (
        error is not None
        and (getattr(error, "type", None) or type(error).__name__) == "ProviderError"
    )


def permanent_provider_error(error) -> bool:
    """Whether the error is a ModelConfigurationError or a provider error with status
    401 or 403."""
    if isinstance(error, ModelConfigurationError):
        return True
    return is_provider_error(error) and provider_status(error) in {401, 403}


def rollout_budget_expired(err) -> bool:
    """Whether a trace error is the harness's agent timeout."""
    return (
        err is not None
        and err.type == "HarnessError"
        and (err.message or "").startswith("agent timeout")
    )


def provider_fault(err) -> bool:
    """Whether a trace error is a transient provider or harness death the conversation
    may reopen on."""
    if err is None:
        return False
    if err.type == "ProviderError":
        return not permanent_provider_error(err)
    return err.type == "HarnessError" and not rollout_budget_expired(err)


def configuration_error(seat: str, config, error) -> ModelConfigurationError:
    """The ModelConfigurationError naming the seat, its model and endpoint, and the
    provider's error."""
    base_url = getattr(getattr(config, "client", None), "base_url", None)
    target = (
        f" ({config.model} at {public_url(str(base_url))})"
        if config is not None and base_url
        else ""
    )
    return ModelConfigurationError(
        f"{seat}{target}: permanent model request failure: {describe(error)}",
        status_code=provider_status(error),
    )


# --- the live trace


def write_record(traces: list[Trace], path: Path) -> None:
    """The last trace's record as JSON at `path`, written whole (a reader never sees a
    part)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(traces[-1].to_record(), ensure_ascii=False))
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


class LiveTrace:
    """An open interaction's in-memory trace, rewritten at `path` every `period` seconds
    while it changes (its mark: nodes, calls, stop condition) by `write(traces, path)`;
    a loop timer, not a sleeping task (nothing to cancel at a hard stop); a write that
    fails is logged, never the turn's fault; zero arms nothing."""

    def __init__(
        self,
        traces: list[Trace],
        path: Path,
        *,
        period: float,
        write: Callable[[list[Trace], Path], None] = write_record,
    ):
        self.traces, self.path, self.period, self._write = (
            traces,
            Path(path),
            period,
            write,
        )
        self._timer: asyncio.TimerHandle | None = None
        self._written: tuple | None = None

    def start(self) -> None:
        if self.period > 0 and self._timer is None:
            self._timer = asyncio.get_running_loop().call_later(self.period, self._tick)

    def _tick(self) -> None:
        self._timer = asyncio.get_running_loop().call_later(self.period, self._tick)
        self.write()

    def write(self) -> None:
        """Write the trace as it stands when it changed since the last write."""
        if not self.traces:
            return
        trace = self.traces[-1]
        mark = (len(trace.nodes), len(trace.calls), trace.stop_condition)
        if mark == self._written:
            return
        try:
            self._write(self.traces, self.path)
        except Exception as exc:  # noqa: BLE001 - a live file is a convenience, never the turn's fault
            logger.warning("live trace %s not written: %s", self.path, exc)
            return
        self._written = mark

    def stop(self) -> None:
        timer, self._timer = self._timer, None
        if timer is not None:
            timer.cancel()

    def remove(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            self.path.unlink()


# --- the seat


@dataclass(frozen=True)
class Opening:
    """What one turn or session asks of a seat: the `task` (its prompts and placement),
    `key` (what the log lines name), `name` (the trace's name), the pool `lease` (None:
    a fresh box, closed after), `prepare(box, fresh)` before the interaction,
    `unchanged(box)` asked once after a reply without a call and, when it says so, the
    `nudge` sent as one more user turn (None: never), `during(box)` run beside the
    interaction and cancelled at its end, `serve(interaction)` a context entered around
    it (the calls it serves), `evidence` when a died interaction is the agent's evidence
    rather than a fault."""

    task: Task
    key: str
    name: str
    lease: str | None = None
    prepare: Callable[[Box, bool], Awaitable[None]] | None = None
    unchanged: Callable[[Box], Awaitable[bool]] | None = None
    nudge: str = ""
    during: Callable[[Box], Awaitable[None]] | None = None
    serve: Callable[[Interaction], AbstractAsyncContextManager[Any]] | None = None
    evidence: bool = False


class Seat:
    """One agent's turn executor. `turn(opening, harvest)` leases the box, prepares it
    (`fresh` when this seat never prepared it), runs one interaction of a prompted task
    and awaits `harvest(session)` on the box before it is parked; `session(opening)` is
    the interaction kept open for the block. An infra fault is a `Fault`; a permanent
    provider refusal a `ModelConfigurationError`; the seat's own defect on one turn is
    that turn's `Fault`, never the run's."""

    def __init__(
        self,
        name: str,
        agent: Agent,
        *,
        log: Callable[[str], None] = print,
        platform: Platform = PLATFORM,
        wait: float | Callable[[], float] = PROVISION_WAIT,
    ):
        self.name, self.agent, self.log, self.platform, self.wait = (
            name,
            agent,
            log,
            platform,
            wait,
        )
        self.prepared: set[str] = set()

    def say(self, key: str, text: str) -> None:
        self.log(f"[{self.name} {key}] {text}")

    async def discard(self, lease: str) -> None:
        """The parked box of `lease` stopped."""
        if self.agent.runtimes is not None:
            await self.agent.runtimes.discard(lease)

    def box_of(self, runtime: Runtime, opening: Opening) -> Box:
        """The leased runtime as a `Box`: its configured workdir (the runtime's own
        directory on the host), the box's home beside it."""
        workdir = getattr(getattr(runtime, "config", None), "workdir", None) or str(
            getattr(runtime, "workdir", "/")
        )
        return Box(runtime, workdir, home=f"{workdir}/.seat", platform=self.platform)

    def live(self, session: "Session") -> LiveTrace | None:
        """The live trace of an open session, None for none."""
        return None

    def record(self, session: "Session") -> None:
        """What a closed session leaves (its trace filed); nothing by default.
        `session.traces` is empty when the interaction never opened."""

    async def turn(
        self, opening: Opening, harvest: Callable[["Session"], Awaitable[T]]
    ) -> T:
        """One turn: the box leased (a parked one, else fresh and `prepare`d), one
        interaction on it, then `harvest(session)` on the box before it is parked; what
        `harvest` returned."""
        try:
            async with self._leased(opening, reuse=opening.lease) as box:
                fresh = box.id not in self.prepared
                self.say(opening.key, f"box {box.id} {'up' if fresh else 'reused'}")
                await self._prepare(opening, box, fresh)
                self.prepared.add(box.id)
                try:
                    async with Session(self, opening, box) as session:
                        await session.turn()
                    # the session closed: its trace is filed and `during` has ended before the harvest reads the box
                    return await harvest(session)
                finally:
                    if opening.lease is None:
                        self.prepared.discard(box.id)
        except InfraError as error:
            raise Fault(str(error), fresh=True) from error
        except (Fault, ModelConfigurationError, asyncio.CancelledError):
            raise
        except Exception as error:  # one turn's defect is its fault, never the run's
            logger.exception("turn %s %s", self.name, opening.key)
            raise Fault(f"{type(error).__name__}: {error}", fresh=True) from error

    @asynccontextmanager
    async def session(self, opening: Opening) -> AsyncIterator["Session"]:
        """A persistent exchange: a fresh box prepared, the interaction open for the
        block; every `session.turn(message)` is one more user turn on it."""
        try:
            async with self._leased(opening, reuse=None) as box:
                self.say(opening.key, f"box {box.id} up for a session")
                await self._prepare(opening, box, True)
                async with Session(self, opening, box) as opened:
                    yield opened
        except InfraError as error:
            raise Fault(str(error), fresh=True) from error
        except (Fault, ModelConfigurationError, asyncio.CancelledError):
            raise
        except Exception as error:  # one session's defect is its fault, never the run's
            logger.exception("session %s %s", self.name, opening.key)
            raise Fault(f"{type(error).__name__}: {error}", fresh=True) from error

    @asynccontextmanager
    async def _leased(
        self, opening: Opening, *, reuse: str | None
    ) -> AsyncIterator[Box]:
        async with provisioned(
            lambda: self.agent.provision(opening.task, reuse=reuse),
            wait=self.wait,
            platform=self.platform,
        ) as runtime:
            yield self.box_of(runtime, opening)

    async def _prepare(self, opening: Opening, box: Box, fresh: bool) -> None:
        check_cancelled()
        if opening.prepare is not None:
            await opening.prepare(box, fresh)


class Session:
    """One open interaction on a box (`serve` entered around it, `during` beside it, the
    live trace); `turn(message)` sends one user message (the opening one is the task's
    prompt when it has one) and returns its `Segment`; `died` is how the interaction
    ended when it did not end by the model's reply; the seat records the trace when the
    session closes."""

    def __init__(self, seat: Seat, opening: Opening, box: Box):
        self.seat, self.opening, self.box = seat, opening, box
        self.traces: list[Trace] = []
        self.stack = contextlib.AsyncExitStack()
        self.it: Interaction | None = None
        self.live: LiveTrace | None = None
        self.beside: asyncio.Task | None = None
        self.turns = 0
        self.started = time.time()
        self.segment: Segment | None = None

    @property
    def key(self) -> str:
        return self.opening.key

    @property
    def name(self) -> str:
        return self.opening.name

    @property
    def trace(self) -> Trace | None:
        """The interaction's trace: live while it is open, the one it minted when it
        failed to open, None before."""
        if self.it is not None:
            return self.it.trace
        return self.traces[-1] if self.traces else None

    @property
    def messages(self) -> list:
        trace = self.trace
        return [node.message for node in trace.nodes] if trace is not None else []

    @property
    def died(self) -> str | None:
        """How the interaction ended when an error did: `type: message` of the trace's
        last error, else None."""
        trace = self.trace
        if trace is None or trace.last_error is None:
            return None
        return describe(trace.last_error)

    async def __aenter__(self) -> Self:
        opening, box = self.opening, self.box
        try:
            self.it = await self.stack.enter_async_context(
                self.seat.agent.interaction(
                    opening.task, runtime=box.runtime, on_trace=self.traces.append
                )
            )
            check_cancelled()
            if opening.serve is not None:
                await self.stack.enter_async_context(opening.serve(self.it))
        except HarnessError as e:
            await self.__aexit__(None, None, None)
            check_cancelled()
            raise Fault(f"interaction could not open: {e}", fresh=False) from e
        except SandboxError as e:
            await self.__aexit__(None, None, None)
            check_cancelled()
            raise Fault(
                f"interaction could not open on the box: {e}", fresh=True
            ) from e
        if opening.during is not None:
            self.beside = asyncio.ensure_future(opening.during(box))
        self.live = self.seat.live(self)
        if self.live is not None:
            self.live.start()
        return self

    async def turn(self, message: str | None = None) -> Segment:
        """One user turn (the opening one of a prompted task takes none: the prompt
        speaks), then the nudge once when the box stands unchanged after a reply without
        a call; a death is the interaction's evidence when the opening says so, else a
        provider or harness death is a `Fault` and any other death is left to the caller
        (`died`)."""
        opening, box, it = self.opening, self.box, self.it
        assert it is not None, "the session is not open"
        self.started = time.time()
        self.turns += 1
        prompted = self.turns == 1 and opening.task.data.prompt is not None
        seg = await (it.turn() if prompted else it.turn(message))
        if (
            opening.unchanged is not None
            and it.trace.last_error is None
            and not seg.terminated
            and await opening.unchanged(box)
        ):
            seg = await it.turn(opening.nudge)
        check_cancelled()
        error = it.trace.last_error
        if error is not None and permanent_provider_error(error):
            raise configuration_error(self.seat.name, self.seat.agent.config, error)
        if error is not None and not opening.evidence and provider_fault(error):
            raise Fault(describe(error), fresh=False)
        self.segment = seg
        return seg

    async def __aexit__(self, *exc) -> None:
        if self.beside is not None:
            self.beside.cancel()
            await asyncio.gather(self.beside, return_exceptions=True)
            self.beside = None
        if self.live is not None:
            self.live.stop()
        try:
            await self.stack.aclose()
        finally:
            self.seat.record(self)
            if self.live is not None:
                self.live.remove()


__all__ = [
    "Fault",
    "LiveTrace",
    "ModelConfigurationError",
    "Opening",
    "Seat",
    "Session",
    "configuration_error",
    "describe",
    "is_provider_error",
    "permanent_provider_error",
    "provider_fault",
    "provider_status",
    "public_url",
    "rollout_budget_expired",
    "write_record",
]
