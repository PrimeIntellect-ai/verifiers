"""Explicit executable agents.

An `Agent` is the reusable execution primitive under evals, topologies, and plain agent
programs: harness + model context + runtime policy, with one arrow, `run(task) -> Trace`.
Topologies add graph recording and deferred rewards around that arrow; the agent itself
only knows how to execute one task.

A bare agent is fully standalone — each `run` provisions a runtime from the policy and
brings up its own per-rollout interception server, right for scripts and small programs:

    agent = vf.Agent(DirectHarness(DirectHarnessConfig()), ctx)
    trace = await agent.run(MyTask(vf.TaskData(idx=0, prompt="...")))

Serving resources have exactly one owner: `RunServices` (interception pools). For pooled
operation (N concurrent runs sharing interception servers, like an eval), enter a scope
and inject it — `async with RunServices() as services: Agent(..., services=services)` —
which is precisely what `TopologyRunner.serving` does for every agent it binds. The agent
never owns services; it borrows them. Taskset-scoped shared tool servers likewise arrive
pre-served (`shared_tools=`, see `serve_shared`), from whoever owns the taskset.

The framework's config-driven packaging of agents — CLI/toml-addressable, persisted as
agent graphs, eval- and trainer-integrated — is the topology (`verifiers.v1.topology`);
reach for a bare `Agent` when you're scripting.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Sequence
from typing import TYPE_CHECKING

from verifiers.v1.clients import ModelContext
from verifiers.v1.env import (
    TimeoutConfig,
    resolve_runtime_config,
    resolve_stage_timeouts,
    validate_task_pairing,
)
from verifiers.v1.harness import Harness
from verifiers.v1.interception import InterceptionPool, RolloutLimits
from verifiers.v1.mcp import SharedToolServer
from verifiers.v1.retries import RolloutRetryConfig, run_with_retry
from verifiers.v1.rollout import Rollout
from verifiers.v1.runtimes import Runtime, RuntimeConfig, make_runtime
from verifiers.v1.services import RunServices
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace
from verifiers.v1.types import Messages, UserMessage

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


Parent = Trace | str
"""An upstream lineage link: the parent trace itself, or just its id."""


def _parent_ids(parents: Sequence[Parent]) -> list[str]:
    return [parent.id if isinstance(parent, Trace) else parent for parent in parents]


_END = object()
"""Inbox sentinel: the session was ended; the episode's next turn is refused."""
_DEAD = object()
"""Reply sentinel: the episode finished (completed, errored, or budget-tripped) —
poisons a pending or future `turn()` into `SessionEnded` instead of a hang."""


class SessionEnded(RuntimeError):
    """Raised by `Session.turn` when the episode can no longer take a turn — it ended,
    errored, or a budget tripped. Carries the trace: episode failure is data, and `go`
    decides what a dead seat means (forfeit it, end the game, keep playing without it)."""

    def __init__(self, trace: Trace | None) -> None:
        condition = trace.stop_condition if trace is not None else None
        super().__init__(f"session ended ({condition or 'episode never started'})")
        self.trace = trace


class Session:
    """A live, suspended episode `go` converses with — the counterpart seat of one
    agent's rollout, held open by `Agent.interact`.

    Three members, deliberately: `turn(message)` sends the episode its next user turn
    and returns the model's reply; `end()` finishes the episode early (idempotent —
    scope exit calls it for you); `.trace` is the live trace (read mid-game state,
    stamp `info`). Everything else — who talks to whom, in what order, with what view
    of the interaction — is plain imperative code in `go`.

    Mechanically this is a user simulator without the server: the interception layer
    suspends the episode between model turns awaiting `_respond`, exactly as it awaits
    a `User`'s respond tool; here the "user" is `go` itself, over a queue handshake.
    The handshake is also the safety contract: `turn()` never blocks on an episode that
    can no longer answer — it raises `SessionEnded` — and a second in-flight `turn()`
    on one session raises immediately (one driver per seat)."""

    def __init__(self) -> None:
        self._rollout: Rollout | None = None
        self._episode: asyncio.Task | None = None
        self._inbox: asyncio.Queue = asyncio.Queue()
        self._replies: asyncio.Queue = asyncio.Queue()
        self._opened = False
        self._pending = False

    @property
    def trace(self) -> Trace:
        """The episode's live trace — readable mid-game (`trace.state`, `num_turns`)
        and the place `go` stamps outcome facts (`trace.info[...]`) for the topology's
        declared rewards to read."""
        if self._rollout is None or self._rollout.trace is None:
            raise RuntimeError("session's episode has not started yet")
        return self._rollout.trace

    async def _respond(self, last_assistant: str) -> Messages:
        """The user-seat contract (see `RolloutSession.user`), driven by the interception
        loop between model turns: deliver the model's turn to `go`, suspend until `go`
        supplies the next user turn. The very first call (`respond(\"\")`, the no-prompt
        opening) delivers nothing — there is no reply yet."""
        if self._opened:
            await self._replies.put(last_assistant)
        self._opened = True
        incoming = await self._inbox.get()
        if incoming is _END:
            return []  # stop is already set; `refused()` ends the exchange cleanly
        return incoming

    async def turn(self, message: str | Messages) -> str:
        """Send the episode its next user turn(s) and return the model's reply text.
        Raises `SessionEnded` (never hangs) when the episode can't answer, and refuses
        a second concurrent `turn()` on this session — including after a `turn()` was
        *cancelled* mid-flight (its message is already with the model, so the
        conversation is desynced; `end()` the seat instead of driving on)."""
        if self._pending:
            raise RuntimeError(
                "a turn is already pending (or was cancelled mid-flight) on this "
                "session — one driver per seat; end() a desynced seat"
            )
        if self._episode is not None and self._episode.done():
            raise SessionEnded(self._rollout.trace if self._rollout else None)
        turns: Messages = (
            [UserMessage(content=message)]
            if isinstance(message, str)
            else list(message)
        )
        self._pending = True
        try:
            await self._inbox.put(turns)
            reply = await self._replies.get()
        except asyncio.CancelledError:
            # The message is in flight with no reader for its reply: leave `_pending`
            # set so a later turn() can't consume the stale reply one turn late.
            raise
        self._pending = False
        if reply is _DEAD:
            raise SessionEnded(self._rollout.trace if self._rollout else None)
        return reply

    async def end(self, condition: str = "interaction_complete") -> None:
        """Finish the episode early and cleanly (forfeits, eliminations). Idempotent;
        scope exit calls it, so plain `go` code never has to."""
        trace = self._rollout.trace if self._rollout is not None else None
        if trace is not None and trace.stop_condition is None:
            trace.stop(condition)
        await self._inbox.put(_END)

    def _poison(self, _task: asyncio.Task) -> None:
        """Episode-completion callback: wake any pending (or future) `turn()` into
        `SessionEnded` rather than leaving it awaiting a reply that will never come."""
        self._replies.put_nowait(_DEAD)


class _AgentAttempt:
    """One retryable attempt (see `retries.Runnable`): each `run()` builds and runs a
    fresh `Rollout` when the agent owns runtime provisioning; borrowed-runtime attempts
    deliberately reuse the borrowed box."""

    def __init__(
        self,
        agent: "Agent",
        task: Task,
        *,
        runtime_config: RuntimeConfig,
        runtime: Runtime | None,
        interception: InterceptionPool | None,
        parents: Sequence[Parent],
    ) -> None:
        self.agent = agent
        self.task = task
        self.runtime_config = runtime_config
        self.runtime = runtime
        self.interception = interception
        self.parents = parents

    async def run(self) -> Trace:
        agent = self.agent
        timeouts = resolve_stage_timeouts(agent.timeout, self.task, self.runtime_config)
        rollout = Rollout(
            task=self.task,
            harness=agent.harness,
            ctx=agent.ctx,
            runtime_config=self.runtime_config,
            setup_timeout=timeouts.setup,
            harness_timeout=timeouts.rollout,
            finalize_timeout=timeouts.finalize,
            scoring_timeout=timeouts.scoring,
            limits=agent.limits,
            shared_tools=agent.shared_tools,
            interception=self.interception,
            runtime=self.runtime,
        )
        trace = await rollout.run()
        agent.stamp(
            trace,
            parents=self.parents,
            runtime=rollout.runtime,
            borrowed=self.runtime is not None,
        )
        return trace


class Agent:
    """A harness + model context + runtime policy, runnable on any compatible task.

    `run(task)` provisions a fresh runtime from the policy and tears it down. To share a
    world explicitly, use `provision()` and pass the yielded runtime back to one or more
    `run(..., runtime=box)` calls; borrowed-runtime rollouts never start or stop the box.

    `services` is the (optional) serving scope the agent borrows pooled interception
    from — see the module docstring; without one, every run brings up its own
    per-rollout interception server. `shared_tools` are the taskset-scoped shared
    servers (already served, see `serve_shared`) this agent's rollouts reuse."""

    def __init__(
        self,
        harness: Harness,
        ctx: ModelContext,
        runtime: RuntimeConfig | None = None,
        *,
        name: str | None = None,
        trainable: bool = True,
        limits: RolloutLimits | None = None,
        timeout: TimeoutConfig | None = None,
        services: RunServices | None = None,
        shared_tools: dict[str, SharedToolServer] | None = None,
    ) -> None:
        self.harness = harness
        self.ctx = ctx
        self.runtime_config = runtime if runtime is not None else harness.config.runtime
        self.name = name
        self.trainable = trainable
        self.limits = limits or RolloutLimits()
        self.timeout = timeout or TimeoutConfig()
        self.shared_tools = shared_tools or {}
        self._services = services
        self._warned_resources: set[tuple[str, str]] = set()
        self._validated: set[tuple[type[Task], str, str]] = set()

    def runtime_for(self, task: Task, runtime: Runtime | None = None) -> RuntimeConfig:
        """The runtime placement this run will actually use."""
        if runtime is not None:
            return runtime.config
        return resolve_runtime_config(self.runtime_config, task, self._warned_resources)

    def _validate_pairing(self, task: Task, runtime_config: RuntimeConfig) -> None:
        key = (type(task), runtime_config.type, runtime_config.__class__.__name__)
        if key not in self._validated:
            validate_task_pairing(
                self.harness, type(task), shared_tools=tuple(self.shared_tools)
            )
            self._validated.add(key)

    async def run(
        self,
        task: Task,
        *,
        parents: Sequence[Parent] = (),
        runtime: Runtime | None = None,
        retry: RolloutRetryConfig | None = None,
    ) -> Trace:
        """Run this agent on `task` once and return its trace, stamped with lineage
        (`parents`) and this agent's provenance. `runtime` places the run into a live
        borrowed box instead of provisioning one; `retry` applies the whole-rollout
        retry policy."""
        runtime_config = self.runtime_for(task, runtime)
        self._validate_pairing(task, runtime_config)
        services = self._services
        attempt = _AgentAttempt(
            self,
            task,
            runtime_config=runtime_config,
            runtime=runtime,
            interception=await services.pool_for(runtime_config)
            if services is not None
            else None,
            parents=parents,
        )
        if retry is not None:
            return await run_with_retry(attempt, retry)
        return await attempt.run()

    def stamp(
        self,
        trace: Trace,
        *,
        parents: Sequence[Parent],
        runtime: Runtime | None,
        borrowed: bool,
    ) -> None:
        """Record agent provenance on a finished trace."""
        if self.name is not None:
            trace.agent = self.name
        trace.parents = _parent_ids(parents)
        trace.trainable = self.trainable
        trace.info["agent"] = {
            "name": self.name,
            "harness": self.harness.config.id,
            "model": self.ctx.model,
            "runtime": {
                "type": runtime.config.type if runtime is not None else None,
                "descriptor": runtime.descriptor if runtime is not None else None,
                "borrowed": borrowed,
            },
        }

    @contextlib.asynccontextmanager
    async def provision(self, task: Task | None = None) -> AsyncIterator[Runtime]:
        """Provision a runtime from this agent's policy and tear it down on exit."""
        if task is None:
            config = self.runtime_config
        else:
            config = self.runtime_for(task)
            self._validate_pairing(task, config)
        runtime = make_runtime(config)
        try:
            await runtime.start()
            yield runtime
        finally:
            await runtime.stop()

    @contextlib.asynccontextmanager
    async def interact(
        self,
        task: Task,
        *,
        parents: Sequence[Parent] = (),
        runtime: Runtime | None = None,
    ) -> AsyncIterator[Session]:
        """Hold a live episode open and yield the `Session` to converse with it — the
        primitive under back-and-forth multi-agent (each agent is the other's user seat,
        `go` is the router). The episode runs in the background, suspending between
        turns; scope exit ends it cleanly (stop → finalize → task scoring) and the
        completed trace is stamped like any `run()`.

        Session episodes are opened by the first `turn()` — build the task with
        `prompt=None` and put the framing in `system_prompt`. No `retry=`: one side of a
        half-played interaction can't be transparently re-run; a dead seat surfaces as
        `SessionEnded` and `go` decides what it means. Note the episode's budgets
        (`max_turns`, timeouts) span the whole interaction, including time suspended
        while other seats move — size them for the game, not a solo run."""
        if task.data.prompt is not None:
            raise ValueError(
                "interact() episodes are opened by the first turn(): build the task "
                "with prompt=None and put the framing in system_prompt"
            )
        if type(task).user is not None:
            raise ValueError(
                f"task {type(task).__name__} declares its own user simulator "
                "(Task.user), and a session is a second claimant for the same user "
                "seat — run it with run(), or drop the simulator"
            )
        if not type(self.harness).SUPPORTS_USER_SIM:
            raise ValueError(
                f"harness {self.harness.config.id!r} cannot take injected user turns; "
                "sessions need a user-capable harness (e.g. direct, null, default)"
            )
        runtime_config = self.runtime_for(task, runtime)
        self._validate_pairing(task, runtime_config)
        services = self._services
        session = Session()
        timeouts = resolve_stage_timeouts(self.timeout, task, runtime_config)
        rollout = Rollout(
            task=task,
            harness=self.harness,
            ctx=self.ctx,
            runtime_config=runtime_config,
            setup_timeout=timeouts.setup,
            harness_timeout=timeouts.rollout,
            finalize_timeout=timeouts.finalize,
            scoring_timeout=timeouts.scoring,
            limits=self.limits,
            shared_tools=self.shared_tools,
            interception=await services.pool_for(runtime_config)
            if services is not None
            else None,
            runtime=runtime,
            user=session._respond,
        )
        session._rollout = rollout
        episode = asyncio.create_task(rollout.run())
        session._episode = episode
        episode.add_done_callback(session._poison)
        try:
            yield session
        finally:
            await session.end()
            try:
                trace = await episode  # never raises: a rollout captures its failures
            except asyncio.CancelledError:
                episode.cancel()
                raise
            self.stamp(
                trace,
                parents=parents,
                runtime=rollout.runtime,
                borrowed=runtime is not None,
            )
