"""The Agent: a reusable (harness x model x runtime) value with one executable arrow.

An `Agent` bundles WHO does the work — the harness (the program), the model context
(model + client + sampling, the same `ModelContext` every rollout consumes), and
a runtime policy (where a run's box comes from by default). `agent.run(task)` executes one
rollout and returns its `Trace`. Everything else is a parameter, not a concept:

  - placement: `runtime=` borrows a live box (creator owns teardown) instead of
    provisioning a fresh one — put a judge into a solver's sandbox, or two agents into
    one world. `agent.provision(task)` hands you a box to place runs into.
  - judgement: the task carries it. A `Task` subclass's hooks (`setup`/`finalize`)
    and signals (`@reward`/`@metric`) run as in any eval; a plain base `Task` has
    no-op hooks and no signals, so the run is unscored — a pure `Task -> Trace` arrow.
  - the user: `user=` supplies the other half of the conversation — any async
    `str -> Messages` callable (see `session.Respond`), injected as user turns by the
    interception. A scripted user is a plain closure; a *modeled* user is another
    agent, driven live through `agent.chat(task)` — the caller side of the same
    channel, one `turn()` per user message.
  - chaining: plain functions. Mint the next task's `TaskData` from earlier traces
    (stamp `sources`/`relation` for lineage) and hand it to the next agent.

Interception follows the runtime story: a live `Interception` can be injected at
construction (borrowed — whoever entered it owns its lifecycle; that's how several
agents share one pool of servers and tunnels). Without one, entering the agent
(`async with`) owns an elastic pool sized to its runtime policy, so N concurrent runs
share interception servers like an eval does. Un-entered, each run brings up its own
per-rollout interception server — fine for scripts and small programs.

The execution machinery is the standard rollout engine (`run_rollout`: staged lifecycle,
typed error attribution, token-true trace capture). The Agent only decides what goes into
the rollout.
"""

import asyncio
import logging
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

from verifiers.v1.clients import (
    ModelContext,
)
from verifiers.v1.env import (
    TimeoutConfig,
    cap_remote_harness_timeout,
    resolve_runtime_config,
    validate_pairing,
)
from verifiers.v1.harness import Harness
from verifiers.v1.interception import ElasticInterceptionPool, Interception
from verifiers.v1.mcp import SharedToolServer
from verifiers.v1.rollout import run_rollout
from verifiers.v1.runtimes import (
    Runtime,
    RuntimeConfig,
    SubprocessConfig,
    make_runtime,
    runtime_is_local,
)
from verifiers.v1.session import Respond, RolloutLimits
from verifiers.v1.task import Task
from verifiers.v1.trace import Trace
from verifiers.v1.types import Messages, UserMessage

logger = logging.getLogger(__name__)


def _check_borrowed_placement(task: Task, runtime: Runtime) -> None:
    """A borrowed box is never re-provisioned, so a task's placement fields can't be
    honored. Parity with the provisioning path where it refuses: a task `image` on a
    subprocess box raises (`resolve_runtime_config` raises the same on a subprocess
    policy — a lifetime/wiring bug in the borrowing program, so it goes to the caller,
    not the trace). A container box whose image differs only warns: placing a run into
    an existing world is the point of borrowing (e.g. a judge in a solver's box)."""
    if task.data.image is None:
        return
    if isinstance(runtime.config, SubprocessConfig):
        raise ValueError(
            f"task {task.data.idx!r} requires image {task.data.image!r}, but the "
            "borrowed runtime is subprocess-backed (no container); borrow a container "
            "box (e.g. agent.provision(task)) or drop the task's image"
        )
    box_image = getattr(runtime.config, "image", None)
    if box_image != task.data.image:
        logger.warning(
            "task %r requires image %r, but borrowed box %r runs %r; a borrowed box "
            "is never re-provisioned, so the run proceeds in the box's world",
            task.data.idx,
            task.data.image,
            runtime.name,
            box_image,
        )


@dataclass(frozen=True)
class Reply:
    """One assistant turn, as `ChatSession.turn` returns it. `stopped` marks the
    exchange over — the run ended (a limit, a `@stop`, or the harness finishing)
    instead of producing another turn; a stopped `Reply` carries no text (the last
    real turn was already delivered), and the session's `trace` holds the full
    exchange."""

    text: str
    stopped: bool = False


class ChatSession:
    """An agent's run, held open turn-by-turn: the caller IS the run's user.

    `agent.chat(task)` starts the run in the background wired to this session;
    `await session.turn("...")` sends one user message and returns the assistant's
    `Reply`. The first `turn()` opens the conversation (chat tasks carry no prompt).
    One consumer at a time — `turn()` is a strict request/response alternation, not a
    mailbox. `session.trace` is live from the moment the run mints it: watch tokens
    and turns mid-exchange, read rewards after close. Leaving the `chat()` context
    closes the session: the run is told the user is gone (the exchange stops as
    `user_closed`), then awaited to completion — hooks and scoring included."""

    def __init__(self) -> None:
        self._to_caller: asyncio.Queue[Reply] = asyncio.Queue()
        self._from_caller: asyncio.Queue[Messages] = asyncio.Queue()
        self._opened = False
        self._ended = False
        self._closed = False
        self._run: asyncio.Task[Trace] | None = None
        self.trace: Trace | None = None

    async def _respond(self, content: str) -> Messages:
        """The run-side half (the `Respond` handed to `run(user=...)`): deliver the
        assistant's turn to the caller, then wait for the caller's next message. The
        first call is the opening ping (chat tasks have no prompt), which has no
        assistant turn to deliver — it only waits for the first `turn()`."""
        if self._opened:
            self._to_caller.put_nowait(Reply(text=content))
        self._opened = True
        return await self._from_caller.get()

    def _watch(self, trace: Trace) -> None:
        self.trace = trace

    def _on_run_done(self, _task: asyncio.Task) -> None:
        # Whatever ended the run (limit, @stop, harness exit, error), a caller mid-
        # `turn()` must wake up: hand it the stopped marker.
        self._ended = True
        self._to_caller.put_nowait(Reply(text="", stopped=True))

    async def turn(self, message: str) -> Reply:
        """Send one user message; return the assistant's `Reply`. A `stopped` reply
        means the run ended instead of answering (the message went unconsumed)."""
        if self._closed:
            raise RuntimeError("this chat is closed")
        if self._ended:
            raise RuntimeError(
                "the exchange is over (the run ended); read session.trace"
            )
        self._from_caller.put_nowait([UserMessage(content=message)])
        return await self._to_caller.get()

    async def close(self) -> Trace:
        """End the exchange and run the rollout to completion (idempotent): the run's
        pending user wait resolves empty (`user_closed`), then the finished — scored —
        trace returns (also on `session.trace`)."""
        if self._run is None:
            raise RuntimeError("session was never started; use agent.chat(task)")
        if not self._closed:
            self._closed = True
            # Resolve the run's pending (or next) user wait: no messages = user closed.
            self._from_caller.put_nowait([])
        return await self._run


class Agent:
    """A harness + model context + runtime policy, runnable on any task.

    `harness` is a concrete `Harness` object (v1 construction is explicit), e.g.
    `DefaultHarness(DefaultHarnessConfig())`; harnesses are stateless, so one instance
    can back any number of agents. `load_harness(config)` resolves hub/local ids.

    `ctx` is the `ModelContext` (model + client + sampling) — an agent IS a model in
    a harness, bound at construction. The client is
    yours to build (`resolve_client(EvalClientConfig())`) and to share: agents on the
    same endpoint should share one `Client` (one connection pool); prime-rl hands every
    agent its renderer client the same way.

    `runtime` here is a *policy* (a `RuntimeConfig`): each `run` provisions a fresh box
    from it, resolved per task (image / workdir / resources); it defaults to the harness
    config's own `runtime`. To place a run into an existing box instead, pass a live
    `Runtime` to `run(runtime=...)` — borrowed boxes are never started or torn down by
    the run; their creator owns their lifecycle.

    `interception` is the same story for the model boundary: a live, already-entered
    `Interception` to borrow — whoever entered it owns its lifecycle, this agent only
    acquires slots. Pass one pool to several agents so they share servers (and tunnels,
    behind remote runtimes). Without it, an entered agent owns an elastic pool; an
    un-entered agent's runs each bring up their own per-rollout server.

    `shared_tools` completes the borrowing set: live `SharedToolServer`s (taskset-scoped
    MCP, served once by their owner — an eval's `serving()`, or a program via
    `serve_shared`) that every run of this agent reuses. Borrowed like the others: never
    started or torn down here."""

    def __init__(
        self,
        harness: Harness,
        ctx: ModelContext,
        runtime: RuntimeConfig | None = None,
        *,
        interception: Interception | None = None,
        shared_tools: Mapping[str, SharedToolServer] | None = None,
        limits: RolloutLimits | None = None,
        timeout: TimeoutConfig | None = None,
    ) -> None:
        self.harness = harness
        self.ctx = ctx
        self.runtime_config: RuntimeConfig = (
            runtime if runtime is not None else harness.config.runtime
        )
        self.interception = interception
        self.shared_tools = dict(shared_tools) if shared_tools else {}
        self.limits = RolloutLimits() if limits is None else limits
        self.timeout = TimeoutConfig() if timeout is None else timeout
        self._entered = False
        self._pool: ElasticInterceptionPool | None = None
        self._warned_resources: set[tuple[str, str]] = set()

    async def __aenter__(self) -> "Agent":
        if self._entered:
            raise RuntimeError("Agent is already entered; enter it once and share it")
        self._entered = True
        if self.interception is None:
            # Sized to the runtime policy: a remote policy needs the tunnel. Runs the
            # pool can't serve fall back per run (`_interception_for`).
            self._pool = ElasticInterceptionPool(
                requires_tunnel=not runtime_is_local(self.runtime_config)
            )
            await self._pool.__aenter__()
        return self

    async def __aexit__(self, *exc) -> None:
        self._entered = False
        pool, self._pool = self._pool, None
        if pool is not None:
            await pool.__aexit__(*exc)

    def _interception_for(self, run_is_local: bool, task: Task) -> Interception | None:
        """Which interception this run rides. An injected one always — its owner sized
        its reach over its consumers, like an eval injecting into every rollout. The
        owned pool only when provably reachable from all of this run's consumers: always
        when it tunnels (a tunnel URL works from anywhere), else for a local run whose
        task brings no tool/user servers and this agent no shared ones (any such server
        may sit in its own remote runtime and must reach `/state`). Otherwise `None` —
        the rollout brings up a per-run server sized to the task."""
        if self.interception is not None:
            return self.interception
        if self._pool is None:
            return None
        if self._pool.requires_tunnel or (
            run_is_local and not self.shared_tools and not type(task).tools
        ):
            return self._pool
        return None

    def _check_user_support(self) -> None:
        if not self.harness.SUPPORTS_USER_SIM:
            raise ValueError(
                f"Harness {self.harness.config.id!r} cannot host a user: its program "
                "doesn't consume injected user turns (SUPPORTS_USER_SIM). Use a "
                "harness that does (e.g. default, null, or the in-process direct "
                "harness)."
            )

    async def run(
        self,
        task: Task,
        *,
        runtime: Runtime | None = None,
        user: Respond | None = None,
        on_trace: Callable[[Trace], None] | None = None,
    ) -> Trace:
        """Run this agent on `task` once and return the trace.

        The task carries its own judgement (its hooks + `@reward`/`@metric` run as in
        any eval); a plain base `Task` makes the run unscored. `runtime` places the run
        into a live box (borrowed — not started or torn down here) instead of
        provisioning a fresh one from the agent's runtime policy. `user` supplies the
        other half of the conversation (any async `str -> Messages`; see
        `session.Respond`): the interception injects its replies as user turns after
        each tool-less model turn, and it ends the exchange by returning no messages —
        for a prompt-less task it also opens the conversation. To BE the user
        yourself, live, use `chat()` instead. `on_trace` observes the run's trace the
        moment it's minted (before any I/O) — how a caller watches the run live (the
        eval dashboard reads stage, tokens, and turns off it)."""
        if user is not None:
            self._check_user_support()
        if runtime is not None:
            _check_borrowed_placement(task, runtime)
            runtime_config = runtime.config
            run_is_local = runtime.is_local
        else:
            runtime_config = resolve_runtime_config(
                self.runtime_config, task, self._warned_resources
            )
            run_is_local = runtime_is_local(runtime_config)
        validate_pairing(
            self.harness, type(task), runtime_config, shared_tools=self.shared_tools
        )
        # Timeout precedence as in an eval's env-rollouts, with the agent standing in
        # for cli/toml: agent-level wins, else the task's, else no limit.
        setup_timeout = (
            self.timeout.setup
            if self.timeout.setup is not None
            else task.data.timeout.setup
        )
        harness_timeout = (
            self.timeout.rollout
            if self.timeout.rollout is not None
            else task.data.timeout.harness
        )
        harness_timeout = cap_remote_harness_timeout(
            harness_timeout, runtime_config, task
        )
        finalize_timeout = (
            self.timeout.finalize
            if self.timeout.finalize is not None
            else task.data.timeout.finalize
        )
        scoring_timeout = (
            self.timeout.scoring
            if self.timeout.scoring is not None
            else task.data.timeout.scoring
        )
        trace = await run_rollout(
            task=task,
            harness=self.harness,
            ctx=self.ctx,
            runtime_config=runtime_config,
            setup_timeout=setup_timeout,
            harness_timeout=harness_timeout,
            finalize_timeout=finalize_timeout,
            scoring_timeout=scoring_timeout,
            limits=self.limits,
            shared_tools=self.shared_tools,
            interception=self._interception_for(run_is_local, task),
            runtime=runtime,
            user=user,
            on_trace=on_trace,
        )
        # Who produced this trace — so a program's traces stay attributable after the
        # Agent objects are gone. Resolved per run: a borrowed box wins over the policy.
        trace.info["agent"] = {
            "harness": self.harness.config.id,
            "model": self.ctx.model,
            "runtime": {
                "type": runtime_config.type,
                "descriptor": trace.runtime.id if trace.runtime is not None else None,
                "borrowed": runtime is not None,
            },
        }
        return trace

    @asynccontextmanager
    async def chat(
        self, task: Task, *, runtime: Runtime | None = None
    ) -> AsyncIterator[ChatSession]:
        """Converse with this agent turn-by-turn: a full `run` of `task` where YOU are
        the user. Yields a `ChatSession` — `await session.turn("...")` sends one user
        message and returns the assistant's `Reply`; the first `turn()` opens the
        conversation, so the task must carry no prompt (put fixed context in
        `task.data.system_prompt`; a prompted task takes `run(user=...)` instead).
        Everything is a real rollout — the trace (live on `session.trace`), limits,
        `@stop`s, and scoring all apply; leaving the context ends the exchange
        (`user_closed`) and finishes the run, hooks and scoring included. Wire two
        sessions' `turn`s together and two agents converse; hand `session.turn`-shaped
        callables around and 'the user is just another agent'."""
        self._check_user_support()
        if task.data.prompt is not None:
            raise ValueError(
                "chat() opens the conversation itself (the first turn() is the "
                "opening user message), so the task must have no prompt; put fixed "
                "context in task.system_prompt, or run a prompted task with "
                "run(user=...)"
            )
        session = ChatSession()
        session._run = asyncio.create_task(
            self.run(
                task, runtime=runtime, user=session._respond, on_trace=session._watch
            )
        )
        session._run.add_done_callback(session._on_run_done)
        try:
            yield session
        finally:
            await session.close()

    @asynccontextmanager
    async def provision(self, task: Task | None = None) -> AsyncIterator[Runtime]:
        """Provision (and on exit tear down) a box from this agent's runtime policy —
        resolved for `task` when given (image / workdir / resources). Place runs into it
        via `run(..., runtime=box)`: the program that provisions a box owns it, so several
        runs (by this or other agents) can share one world."""
        config = (
            resolve_runtime_config(self.runtime_config, task, self._warned_resources)
            if task is not None
            else self.runtime_config
        )
        runtime = make_runtime(config)
        try:
            # start inside the try: a failed start may already hold a remote sandbox,
            # so it must reach `stop()` (safe on a partially-started runtime) like in
            # `run_rollout`.
            await runtime.start()
            yield runtime
        finally:
            await runtime.stop()
