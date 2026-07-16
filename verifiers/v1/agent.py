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
  - the exchange: `agent.chat(task)` holds the rollout open turn-by-turn — the
    caller IS the run's user, one `turn()` per harness segment (the program yields,
    the caller answers, the next segment resumes with the answer). Who computes the
    turns is control flow, not a framework concept: an env's rollout loop, another
    agent's session, a game engine, a scripted closure, a human.
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
from verifiers.v1.rollout import RolloutRun, _as_messages, run_rollout
from verifiers.v1.runtimes import (
    Runtime,
    RuntimeConfig,
    SubprocessConfig,
    make_runtime,
    runtime_is_local,
)
from verifiers.v1.session import RolloutLimits
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
    """An agent's rollout, held open turn-by-turn: the caller IS the run's user.

    `agent.chat(task)` opens the rollout wired to this session; `await
    session.turn("...")` sends one user turn and runs ONE harness segment — the
    program, resumed onto the conversation, until it yields — returning the
    assistant's `Reply`. A prompt-less (or masked) task is opened by the first
    `turn(message)`; a prompted task speaks first — a bare `turn()` takes its
    opening reply. One consumer at a time — `turn()` is a strict
    request/response alternation, not a mailbox. `session.trace` is live from the
    moment the session exists: watch tokens and turns mid-exchange, read rewards
    after close. Leaving the `chat()` context closes the session (the exchange
    stops as `user_closed`) and finishes the rollout — hooks and scoring
    included."""

    def __init__(self, run: "RolloutRun") -> None:
        self._run = run
        self._over = False  # a stopped reply was already delivered
        self._started = False  # a segment has run (the exchange is under way)

    @property
    def trace(self) -> Trace:
        return self._run.trace

    async def turn(self, message: str | Messages | None = None) -> Reply:
        """Send one user turn (a string, or full `Messages` for multimodal /
        multi-message turns); run one segment; return the assistant's `Reply`. A
        prompted task speaks FIRST: take its opening reply with a bare `turn()`
        before answering. A `stopped` reply means the run ended instead of
        answering (the message went unconsumed)."""
        if self._run.closed:
            raise RuntimeError("this chat is closed")
        if self._over:
            raise RuntimeError(
                "the exchange is over (the run ended); read session.trace"
            )
        prompted = not self._started and self.trace.task.data.prompt is not None
        if message is None and not prompted:
            raise ValueError(
                "nothing to run a turn on: a bare turn() takes a prompted task's "
                "opening reply; this exchange takes its next user message"
            )
        if message is not None and prompted:
            raise ValueError(
                "the task's prompt opens this exchange: take its first reply with "
                "a bare turn() before answering (or mask the prompt with "
                "chat(mask_prompt=True) to open the conversation yourself)"
            )
        messages: Messages | None = None
        if isinstance(message, str):
            messages = [UserMessage(content=message)]
        elif message is not None:
            messages = _as_messages(message)
        self._started = True
        turns_before = self.trace.num_turns
        await self._run.step(messages)
        if self.trace.num_turns > turns_before:
            # The segment answered — even if a limit or @stop then ended the
            # exchange, that surfaces as the NEXT turn's stopped reply.
            return Reply(text=self.trace.last_reply)
        self._over = True
        return Reply(text="", stopped=True)

    async def close(self) -> Trace:
        """End the exchange and finish the rollout (idempotent): scoring and hooks
        run, then the finished trace returns (also on `session.trace`)."""
        if self._run.ok:
            self.trace.stop("user_closed")
        return await self._run.close()


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
        task brings no tool servers and this agent no shared ones (any such server
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
        # Multi-turn capability is a derived fact, not a flag: an exchange advances
        # by resuming the harness onto the conversation, so the harness needs either
        # the default relaunch (a Messages prompt) or its own native continuation.
        harness = self.harness
        if (
            type(harness).resume is Harness.resume
            and not harness.SUPPORTS_MESSAGE_PROMPT
        ):
            raise ValueError(
                f"Harness {harness.config.id!r} cannot host a user: resuming an "
                "exchange takes a Messages prompt (SUPPORTS_MESSAGE_PROMPT) for the "
                "default relaunch-on-the-conversation, or a native resume() "
                "override. Use a harness that has one (e.g. default, null, or the "
                "in-process direct harness)."
            )

    async def run(
        self,
        task: Task,
        *,
        runtime: Runtime | None = None,
        on_trace: Callable[[Trace], None] | None = None,
    ) -> Trace:
        """Run this agent on `task` once and return the trace: one segment — the
        program runs on the task's prompt until it exits.

        The task carries its own judgement (its hooks + `@reward`/`@metric` run as in
        any eval); a plain base `Task` makes the run unscored. `runtime` places the run
        into a live box (borrowed — not started or torn down here) instead of
        provisioning a fresh one from the agent's runtime policy. A multi-turn
        exchange is `chat()`: whoever calls `turn()` is the run's user — an env's
        control flow, a simulator agent's replies, a game engine, or you. `on_trace`
        observes the run's trace the moment it's minted (before any I/O) — how a
        caller watches the run live (the eval dashboard reads stage, tokens, and
        turns off it)."""
        params = self._rollout_params(task, runtime)
        trace = await run_rollout(task=task, on_trace=on_trace, **params)
        self._stamp_agent(trace, params["runtime_config"], borrowed=runtime is not None)
        return trace

    def _rollout_params(self, task: Task, runtime: Runtime | None) -> dict:
        """Resolve one run's execution parameters — runtime config, pairing checks,
        stage timeouts, interception — shared by `run` and `chat`."""
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
        harness_timeout = (
            self.timeout.rollout
            if self.timeout.rollout is not None
            else task.data.timeout.harness
        )
        return dict(
            harness=self.harness,
            ctx=self.ctx,
            runtime_config=runtime_config,
            setup_timeout=(
                self.timeout.setup
                if self.timeout.setup is not None
                else task.data.timeout.setup
            ),
            harness_timeout=cap_remote_harness_timeout(
                harness_timeout, runtime_config, task
            ),
            finalize_timeout=(
                self.timeout.finalize
                if self.timeout.finalize is not None
                else task.data.timeout.finalize
            ),
            scoring_timeout=(
                self.timeout.scoring
                if self.timeout.scoring is not None
                else task.data.timeout.scoring
            ),
            limits=self.limits,
            shared_tools=self.shared_tools,
            interception=self._interception_for(run_is_local, task),
            runtime=runtime,
        )

    def _stamp_agent(
        self, trace: Trace, runtime_config: RuntimeConfig, *, borrowed: bool
    ) -> None:
        # Who produced this trace — so a program's traces stay attributable after the
        # Agent objects are gone. Resolved per run: a borrowed box wins over the policy.
        trace.info["agent"] = {
            "harness": self.harness.config.id,
            "model": self.ctx.model,
            "runtime": {
                "type": runtime_config.type,
                "descriptor": trace.runtime.id if trace.runtime is not None else None,
                "borrowed": borrowed,
            },
        }

    @asynccontextmanager
    async def chat(
        self,
        task: Task,
        *,
        runtime: Runtime | None = None,
        mask_prompt: bool = False,
        on_trace: Callable[[Trace], None] | None = None,
    ) -> AsyncIterator[ChatSession]:
        """Converse with this agent turn-by-turn: a full rollout of `task` where the
        CALLER is the run's user — the one exchange surface. Yields a `ChatSession`;
        `await session.turn("...")` sends one user turn and runs one harness segment,
        returning the assistant's `Reply`. Who computes the turns is control flow,
        not framework machinery: an env's rollout loop, another agent's session, a
        game engine, a scripted closure, a human.

        The task's shape says who speaks first: a prompt-less task is opened by the
        first `turn(message)`; a prompted task speaks first — take its opening reply
        with a bare `turn()` before answering. `mask_prompt` says a prompted task's
        prompt belongs to the USER side (a scenario the caller pursues, not the
        assistant's seed): the wire hides it — the caller opens — while the task
        object keeps the full row, so its hooks, rewards, and judges score the real
        question (the user-sim env's contract).

        Everything is a real rollout — the trace (live on `session.trace`), limits,
        `@stop`s, and scoring all apply; leaving the context ends the exchange
        (`user_closed`) and finishes the rollout, hooks and scoring included."""
        self._check_user_support()
        if mask_prompt and task.data.prompt is None:
            raise ValueError(
                "mask_prompt hides a prompt the task doesn't have; a prompt-less "
                "task is already opened by the first turn()"
            )
        params = self._rollout_params(task, runtime)
        run = RolloutRun(
            task=task,
            wire_data=(
                task.data.model_copy(update={"prompt": None}) if mask_prompt else None
            ),
            has_user=True,
            on_trace=on_trace,
            **params,
        )
        session = ChatSession(run)
        await run.open()
        try:
            yield session
        finally:
            trace = await session.close()
            self._stamp_agent(
                trace, params["runtime_config"], borrowed=runtime is not None
            )

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
