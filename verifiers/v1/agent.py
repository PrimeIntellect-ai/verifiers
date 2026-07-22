"""The Agent: a reusable (harness x model x runtime) value with one executable
arrow — `agent.run(task) -> Trace`; `runtime=` borrows a live box,
`provision(task)` hands you one. The exchange is `agent.chat(task)`: the rollout
held open turn-by-turn, the caller as the run's user — one `turn()` per harness
segment; who computes the turns is control flow, not a framework concept.
Inject a live `Interception` to share servers across agents (a pool belongs to
what spans agents, never to one agent); an entered agent (`async with`) owns one
server; un-entered, each run brings its own."""

import asyncio
import logging
from collections.abc import Callable, Iterator, Mapping
from contextlib import asynccontextmanager, nullcontext
from dataclasses import dataclass
from typing import AsyncIterator

from pydantic import SerializeAsAny, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.clients import (
    Client,
    ClientConfig,
    EvalClientConfig,
    ModelContext,
    resolve_client,
)
from verifiers.v1.harness import Harness, HarnessConfig
from verifiers.v1.interception import Interception, InterceptionServer
from verifiers.v1.mcp import SharedToolServer
from verifiers.v1.retries import RetryConfig, backoff, trace_should_retry
from verifiers.v1.rollout import RolloutRun, _as_messages
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
from verifiers.v1.types import Messages, Sampling, SamplingConfig, UserMessage
from verifiers.v1.utils.compile import (
    cap_remote_harness_timeout,
    resolve_runtime_config,
    validate_pairing,
)

logger = logging.getLogger(__name__)


class TimeoutConfig(BaseConfig):
    """Per-agent wall-clock timeouts per rollout stage, in seconds (None = no
    limit); each stage falls back to the task's own `TaskTimeout` when unset."""

    setup: float | None = None  # one shared budget: task setup + provisioning
    rollout: float | None = None
    finalize: float | None = None
    scoring: float | None = None


class AgentConfig(BaseConfig):
    """One env agent: who plays it, and its per-run caps. It pins only what
    makes it a different actor; everything unpinned falls back — the model context
    to the run's own, the harness to the taskset's default."""

    harness: SerializeAsAny[HarnessConfig] | None = None
    """The agent's program + runtime policy (None = the taskset's default harness)."""
    model: str | None = None
    """Model id (None = the run's model, i.e. the policy under evaluation/training)."""
    client: ClientConfig | None = None
    """Endpoint override (None = the run's client)."""
    sampling: SamplingConfig | None = None
    """Sampling override (None = the run's sampling)."""
    timeout: TimeoutConfig = TimeoutConfig()
    retries: RetryConfig = RetryConfig()
    """Whole-run retries: rerun this agent's rollout while its trace ends with a
    retryable error (never into a borrowed box)."""
    max_turns: int | None = None
    """Max model turns per run (None = no limit). Framework-enforced (the
    interception server refuses turns past it), so it applies to any harness."""
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_total_tokens: int | None = None
    """Token caps per run (None = no limit); framework-enforced between turns."""

    @model_validator(mode="before")
    @classmethod
    def _resolve_harness(cls, data):
        """Narrow a pinned `harness` to its concrete config type by `id` (absent
        stays None = the taskset's default). The lazy import keeps class-body
        `AgentConfig()` defaults constructible while this module initializes."""
        if isinstance(data, dict) and data.get("harness") is not None:
            from verifiers.v1.loaders import harness_config_type, narrow_plugin_field

            narrow_plugin_field(data, "harness", harness_config_type, "bash")
        return data


def _check_borrowed_placement(task: Task, runtime: Runtime) -> None:
    """A borrowed box is never re-provisioned, so a task's placement fields can't
    be honored. A task `image` on a subprocess box raises (a wiring bug — it goes
    to the caller, not the trace); a container box whose image differs only warns,
    since placing a run into an existing world is the point of borrowing."""
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
        self._lock = asyncio.Lock()

    @property
    def trace(self) -> Trace:
        return self._run.trace

    async def turn(self, message: str | Messages | None = None) -> Reply:
        """Send one user turn (a string, or full `Messages` for multimodal /
        multi-message turns); run one segment; return the assistant's `Reply`. A
        prompted task speaks FIRST: take its opening reply with a bare `turn()`
        before answering. A `stopped` reply means the run ended instead of
        answering (the message went unconsumed)."""
        async with self._lock:
            return await self._turn(message)

    async def _turn(self, message: str | Messages | None) -> Reply:
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
        async with self._lock:
            if not self._run.closed and self._run.ok:
                self.trace.stop("user_closed")
            return await self._run.close()


class Agent:
    """A configured harness + model + runtime policy, runnable on any task.

    Built from an `AgentConfig` alone; `client=`/`interception=` inject live
    resources to borrow — agents on one endpoint should share one `Client`, and a
    live `Interception`'s owner keeps its lifecycle. The harness config's
    `runtime` is a *policy*: each `run` provisions a fresh box from it, resolved
    per task; `run(runtime=...)` places the run into an existing box instead
    (borrowed boxes are never started or torn down by the run)."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        client: Client | None = None,
        interception: Interception | None = None,
    ) -> None:
        from verifiers.v1.loaders import harness_config_type, load_harness

        if config.model is None:
            raise ValueError(
                "AgentConfig.model is unset; an Agent needs a pinned model "
                "(inside an env the run's own model fills it in)"
            )
        harness_config = config.harness
        if harness_config is None:
            harness_config = harness_config_type("bash")(id="bash")
        self.config = config
        self.harness = load_harness(harness_config)
        self._owns_client = client is None
        if self._owns_client:
            client = resolve_client(config.client or EvalClientConfig())
        self.ctx = ModelContext(
            model=config.model,
            client=client,
            sampling=config.sampling if config.sampling is not None else Sampling(),
        )
        self._closed = False
        self.runtime_config: RuntimeConfig = self.harness.config.runtime
        self.interception = interception
        self.limits = RolloutLimits(
            max_turns=config.max_turns,
            max_input_tokens=config.max_input_tokens,
            max_output_tokens=config.max_output_tokens,
            max_total_tokens=config.max_total_tokens,
        )
        self.timeout = config.timeout
        # Env-owned standing, not config: `Env.setup` marks fixed agents
        # untrainable and traces are stamped from here; inert outside an env.
        self.trainable: bool = True
        self._entered = False
        self._server: InterceptionServer | None = None
        self._warned_resources: set[tuple[str, str]] = set()

    async def __aenter__(self) -> "Agent":
        if self._entered:
            raise RuntimeError("Agent is already entered; enter it once and share it")
        if self._closed:
            raise RuntimeError("Agent is closed; create a new agent")
        self._entered = True
        if self.interception is None:
            # Sized to the runtime policy (remote needs the tunnel); runs the
            # server can't serve fall back per run.
            self._server = InterceptionServer(
                requires_tunnel=not runtime_is_local(self.runtime_config)
            )
            try:
                await self._server.__aenter__()
            except BaseException:
                # A failed __aenter__ gets no __aexit__ from `async with`: unwind
                # here, or the agent stays "already entered" forever.
                self._entered, self._server = False, None
                if self._owns_client:
                    self._closed = True
                    await self.ctx.client.close()
                raise
        return self

    async def __aexit__(self, *exc) -> None:
        self._entered = False
        server, self._server = self._server, None
        try:
            if server is not None:
                await server.__aexit__(*exc)
        finally:
            if self._owns_client:
                self._closed = True
                await self.ctx.client.close()

    def _interception_for(
        self, run_is_local: bool, task: Task, shared_tools: Mapping
    ) -> Interception | None:
        """Which interception this run rides: an injected one always (its owner
        sized its reach); the owned server only when provably reachable from all
        the run's consumers — when it tunnels, else for a local run with no tool
        servers in play (such servers may sit in a remote runtime and must reach
        `/state`). Otherwise `None`: a per-run server sized to the task."""
        if self.interception is not None:
            return self.interception
        if self._server is None:
            return None
        if self._server.tunnel is not None or (
            run_is_local and not shared_tools and not type(task).tools
        ):
            return self._server
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
                "override. Use a harness that has one (e.g. bash or null)."
            )

    async def run(
        self,
        task: Task,
        *,
        runtime: Runtime | None = None,
        tools: Mapping[str, SharedToolServer] | None = None,
        on_trace: Callable[[Trace], None] | None = None,
    ) -> Trace:
        """Run this agent on `task` once and return the trace: one segment — the
        program runs on the task's prompt until it exits (a multi-turn exchange
        is `chat()`). `runtime` places it into a live borrowed box instead of
        provisioning one; `tools` are live servers borrowed from their
        owner, counted in the pairing check; `on_trace` observes the trace the
        moment it's minted, before any I/O. Retries whole while the trace ends
        with a retryable error (`config.retries`) — never into a borrowed box;
        the final trace keeps earlier attempts' errors."""
        if self._closed:
            raise RuntimeError("Agent is closed; create a new agent")
        retry = self.config.retries
        history: list = []
        for attempt in range(retry.max_retries + 1):
            trace = await self._run_once(task, runtime, tools, on_trace)
            if attempt == retry.max_retries or not trace_should_retry(trace, retry):
                break
            if runtime is not None:
                logger.warning(
                    "not retrying the rollout on a borrowed box (its state is no "
                    "longer the task's start state); the error stands"
                )
                break
            history.extend(trace.errors)
            delay = backoff(attempt)
            logger.warning(
                "retrying agent rollout (retry %d/%d) in %.1fs after error: %s",
                attempt + 1,
                retry.max_retries,
                delay,
                trace.error.type if trace.error else "?",
            )
            await asyncio.sleep(delay)
        if history:
            # The full history rides the final trace either way; success is the
            # `ok` stamp, never errors-emptiness.
            trace.errors = history + trace.errors
        return trace

    async def _run_once(
        self,
        task: Task,
        runtime: Runtime | None,
        shared_tools: Mapping[str, SharedToolServer] | None,
        on_trace: Callable[[Trace], None] | None,
    ) -> Trace:
        params = self._rollout_params(task, runtime, dict(shared_tools or {}))
        run = RolloutRun(task=task, on_trace=on_trace, **params)
        try:
            if await run.open():
                await run.step()
                if run.ok:
                    run.trace.stop("agent_completed")
            trace = await run.close()
        except BaseException:
            # A cancellation mid-run (or a lifetime bug raised to the caller) means
            # close() never runs — free the run's servers and owned runtime first.
            await run.abort()
            raise
        if trace.runtime is not None:
            trace.runtime.borrowed = runtime is not None
        return trace

    @asynccontextmanager
    async def chat(
        self,
        task: Task,
        *,
        runtime: Runtime | None = None,
        tools: Mapping[str, SharedToolServer] | None = None,
        mask_prompt: bool = False,
        on_trace: Callable[[Trace], None] | None = None,
    ) -> AsyncIterator[ChatSession]:
        """Converse with this agent turn-by-turn: a full rollout of `task` where
        the CALLER is the run's user — the one exchange surface. Yields a
        `ChatSession`; `await session.turn("...")` sends one user turn and runs
        one harness segment, returning the assistant's `Reply`. Who computes the
        turns is control flow, not framework machinery: an env's rollout loop,
        another agent's session, a game engine, a scripted closure, a human.

        The task's shape says who speaks first: a prompt-less task is opened by
        the first `turn(message)`; a prompted task speaks first — take its opening
        reply with a bare `turn()` before answering. `mask_prompt` says a prompted
        task's prompt belongs to the USER side (a scenario the caller pursues, not
        the assistant's seed): the wire hides it — the caller opens — while the
        task object keeps the full row, so its hooks, rewards, and judges score
        the real question (the user-sim env's contract).

        `runtime` and `tools` borrow live resources from their owners, just as
        they do for `run()`; an env supplies its taskset's shared tools
        automatically for tasks loaded from that taskset.

        Everything is a real rollout — the trace (live on `session.trace`),
        limits, `@stop`s, and scoring all apply; leaving the context ends the
        exchange (`user_closed`) and finishes the rollout, hooks and scoring
        included. An exchange is caller-driven, so `config.retries` does not
        apply here."""
        if self._closed:
            raise RuntimeError("Agent is closed; create a new agent")
        self._check_user_support()
        if mask_prompt and task.data.prompt is None:
            raise ValueError(
                "mask_prompt hides a prompt the task doesn't have; a prompt-less "
                "task is already opened by the first turn()"
            )
        params = self._rollout_params(task, runtime, dict(tools or {}))
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
        except Exception as e:
            run.fail(e)
            raise
        except BaseException:
            await run.abort()
            raise
        finally:
            trace = run.trace if run.closed else await session.close()
            if trace.runtime is not None:
                trace.runtime.borrowed = runtime is not None

    def _rollout_params(
        self, task: Task, runtime: Runtime | None, shared_tools: dict
    ) -> dict:
        """Resolve one run's runtime config, pairing checks, timeouts,
        interception — shared by `run` and `chat`."""
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
            self.harness, type(task), runtime_config, shared_tools=shared_tools
        )
        # Timeout precedence: agent-level wins, else the task's, else no limit.
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
            shared_tools=shared_tools,
            interception=self._interception_for(run_is_local, task, shared_tools),
            runtime=runtime,
        )

    @asynccontextmanager
    async def provision(self, task: Task | None = None) -> AsyncIterator[Runtime]:
        """Provision (and on exit tear down) a box from this agent's runtime
        policy, resolved for `task` when given; share it via `run(..., runtime=box)`."""
        config = (
            resolve_runtime_config(self.runtime_config, task, self._warned_resources)
            if task is not None
            else self.runtime_config
        )
        runtime = make_runtime(config)
        try:
            # start() inside the try: a failed start may already hold a remote
            # sandbox, so it must reach stop() (safe on a partially-started runtime).
            await runtime.start()
            yield runtime
        finally:
            await runtime.stop()


class _EpisodeAgent(Agent):
    """One role's `Agent` for one episode, built fresh each time (a cheap
    bundle of references — expensive resources are env-owned and borrowed, so no
    state spans concurrent episodes): traces get their agent standing the moment
    they're created, finished ones land in `completed` (the episode's traces),
    each run takes the eval's gate. The taskset's shared tool servers ride only
    its own tasks — on an env-minted task they'd wrongly put MCP in play
    (`tools=` overrides)."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        client: Client,
        interception: Interception | None,
        name: str,
        shared_tools: Mapping[str, SharedToolServer],
        task_cls: type[Task],
        gate: asyncio.Semaphore | None,
        completed: list[Trace],
        on_trace: Callable[[Trace], None] | None,
        on_discard: Callable[[Trace], None] | None,
        warned_resources: set,
    ) -> None:
        super().__init__(config, client=client, interception=interception)
        # Resource warnings dedupe env-wide, not per episode.
        self._warned_resources = warned_resources
        self._name = name
        self._shared_tools = shared_tools
        self._task_cls = task_cls
        self._gate = gate
        self._completed = completed
        self._on_trace = on_trace
        self._on_discard = on_discard

    def _shared_for(self, task: Task) -> Mapping[str, SharedToolServer]:
        return self._shared_tools if isinstance(task, self._task_cls) else {}

    def _watch(
        self, on_trace: Callable[[Trace], None] | None
    ) -> Callable[[Trace], None]:
        last: Trace | None = None

        def watch(trace: Trace) -> None:
            nonlocal last
            if trace.agent is not None:
                trace.agent.name = self._name
                trace.agent.trainable = self.trainable
            # A per-agent retry mints a replacement: the abandoned attempt's trace
            # must leave live views (only the final one joins the episode).
            if last is not None and self._on_discard is not None:
                self._on_discard(last)
            last = trace
            if self._on_trace is not None:
                self._on_trace(trace)
            if on_trace is not None:
                on_trace(trace)

        return watch

    async def run(
        self,
        task: Task,
        *,
        runtime: Runtime | None = None,
        tools: Mapping[str, SharedToolServer] | None = None,
        on_trace: Callable[[Trace], None] | None = None,
    ) -> Trace:
        async with self._gate or nullcontext():
            trace = await super().run(
                task,
                runtime=runtime,
                tools=tools if tools is not None else self._shared_for(task),
                on_trace=self._watch(on_trace),
            )
        self._completed.append(trace)
        return trace

    @asynccontextmanager
    async def chat(
        self,
        task: Task,
        *,
        runtime: Runtime | None = None,
        tools: Mapping[str, SharedToolServer] | None = None,
        mask_prompt: bool = False,
        on_trace: Callable[[Trace], None] | None = None,
    ) -> AsyncIterator[ChatSession]:
        """The agent's `chat`, with every trace stamped with its agent standing
        at mint and captured in `completed` at close — a chat driven from
        `Env.run` stays crash-safe. No gate: a session is held open
        across the exchange (two of them interleave in one episode), so it must
        not occupy an eval slot the way a one-shot `run` does."""
        async with super().chat(
            task,
            runtime=runtime,
            tools=tools if tools is not None else self._shared_for(task),
            mask_prompt=mask_prompt,
            on_trace=self._watch(on_trace),
        ) as session:
            try:
                yield session
            finally:
                self._completed.append(session.trace)


def make_agent(
    config: AgentConfig,
    *,
    client: Client | None = None,
    interception: Interception | None = None,
) -> Agent:
    """The agent for a config; `client`/`interception` inject live resources to
    borrow, everything else comes from the config."""
    return Agent(config, client=client, interception=interception)


MakeAgent = Callable[[str, AgentConfig], Agent]
"""An agent factory keyed by name — what `Agents` calls per scraped config field."""


def agent_config_fields(config) -> dict[str, AgentConfig]:
    """The top-level `AgentConfig` fields declared on a config, in declaration
    order — the env's agents, keyed by field name (the only naming site)."""
    return {name: value for name, value in config if isinstance(value, AgentConfig)}


class Agents:
    """A config's agents, addressed by attribute: every top-level `AgentConfig`
    field becomes an `Agent` under the field's name (`agents.solver`)."""

    def __init__(self, config, make: MakeAgent | None = None) -> None:
        if make is None:
            make = lambda _, spec: make_agent(spec)  # noqa: E731
        self._agents: dict[str, Agent] = {
            name: make(name, value)
            for name, value in agent_config_fields(config).items()
        }

    def __getattr__(self, name: str) -> Agent:
        # self.__dict__ directly: attribute lookup re-entering __getattr__ before
        # __init__ ran (copy/unpickle) must raise, not recurse.
        agents = self.__dict__.get("_agents")
        if agents is None or name not in agents:
            raise AttributeError(
                f"no agent {name!r}; this config declares "
                f"{sorted(agents) if agents else []}"
            )
        return agents[name]

    def __iter__(self) -> Iterator[Agent]:
        return iter(self._agents.values())

    def __len__(self) -> int:
        return len(self._agents)
