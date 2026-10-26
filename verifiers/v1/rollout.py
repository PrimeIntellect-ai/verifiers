"""Lifecycle of one agent rollout."""

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass
from urllib.parse import urlsplit

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.agent import AgentConfig
from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.configs.scoring import resolve_scoring
from verifiers.v1.errors import (
    HarnessError,
    RolloutError,
    SandboxError,
    TaskError,
    ToolsetError,
    boundary,
)
from verifiers.v1.grading import (
    bind_scoring_env,
    grade_task,
    scoring_network,
    scoring_transaction,
    verifier_runtime,
)
from verifiers.v1.harness import Harness, HarnessSession
from verifiers.v1.interception import Interception, serve_interception
from verifiers.v1.mcp import SharedToolServer, serve_tools
from verifiers.v1.runtimes import (
    Runtime,
    RuntimeConfig,
    make_backend,
)
from verifiers.v1.runtimes.private_network import tool_vms
from verifiers.v1.session import RolloutLimits, RolloutSession, hook_boundary
from verifiers.v1.state import state_cls
from verifiers.v1.task import Task
from verifiers.v1.trace import AgentInfo, Trace, TraceTask
from verifiers.v1.types import Messages, Request, Response, SystemMessage, UserMessage
from verifiers.v1.utils.artifacts import collect
from verifiers.v1.utils.decorators import discover_decorated, invoke

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RolloutTimeouts:
    """Per-stage rollout timeouts, each bounding one rollout stage."""

    setup: float | None = None
    """Timeout (in seconds) for task/harness setup through session preparation."""
    agent: float | None = None
    """Timeout (in seconds) for the agent's solve attempt."""
    finalize: float | None = None
    """Timeout (in seconds) for the task + harness finalize hooks."""
    scoring: float | None = None
    """Combined budget (in seconds) for harness metrics and task scoring, including
    isolated verifier setup and retries. Finalization has its own budget."""


_HOME_PROBE = (
    "home=$(awk -F: -v u=\"$1\" '$1 == u { print $6 }' /etc/passwd); "
    'echo "$home"; test -n "$home" && test -d "$home"'
)


async def resolve_agent_runtime(runtime: Runtime, user: str | None) -> Runtime:
    """The runtime the harness runs through: `runtime` itself, or a view running as `user`
    with `HOME` set to its home. `user` must exist with a home directory it can write
    (task authors ship it in the image), and the view must really run as it, so a
    runtime ignoring the user never runs the agent as root."""
    if user is None:
        return runtime
    lookup = await runtime.run(["sh", "-c", _HOME_PROBE, "sh", user], {})
    home = lookup.stdout.strip()
    if not home:
        raise TaskError(f"agent_user {user!r} does not exist in the image")
    if lookup.exit_code != 0:
        raise TaskError(f"agent_user {user!r} has no home directory ({home})")
    view = runtime.with_user(user, home=home)
    probe = await view.run(["sh", "-c", 'id -un; test -w "$HOME" && echo writable'], {})
    effective, *rest = probe.stdout.split() or [""]
    if effective != user:
        raise SandboxError(
            f"{type(runtime.backend).__name__} ran as {effective!r} instead of agent_user "
            f"{user!r}; refusing to run the agent"
        )
    if rest != ["writable"]:
        raise TaskError(f"agent_user {user!r} cannot write its home directory ({home})")
    return view


class Rollout:
    """Manages one rollout's lifecycle (open, step, close)."""

    def __init__(
        self,
        *,
        task: Task,
        agent_config: AgentConfig,
        harness: Harness,
        ctx: ModelContext,
        runtime_config: RuntimeConfig,
        has_user: bool = False,
        timeouts: RolloutTimeouts,
        limits: RolloutLimits,
        shared_tools: dict[str, SharedToolServer] | None = None,
        interception: Interception | None = None,
        runtime: Runtime | None = None,
        on_trace: Callable[[Trace], None] | None = None,
        collect_artifacts: bool = False,
        on_score: Callable[[Trace, Runtime], Awaitable[None]] | None = None,
    ) -> None:
        self._on_score = on_score
        self.task = task
        self.harness = harness
        self.ctx = ctx
        self.runtime_config = runtime_config
        self._scoring_policy = resolve_scoring(task.scoring, agent_config.scoring)
        self._grading_config = None
        self._scoring_task = task
        self._grader = None
        self.deployment = None
        self._deployment_config = agent_config.deployment
        self._has_user = has_user
        self._timeouts = timeouts
        self._agent_time_remaining = self._timeouts.agent
        self._shared_tools = shared_tools or {}
        self._interception = interception
        self.runtime = runtime
        self._borrowed_runtime = runtime
        self._collect_artifacts = collect_artifacts
        self.trace: Trace = Trace(
            task=TraceTask(
                type=type(task).__name__,
                data=task.data,
                key=task.key,
                hash=task.hash,
            ),
            state=state_cls(type(task))(),
            # The seat's resolved config, role overrides included — the agent
            # this trace can be reproduced with.
            agent=AgentInfo(config=agent_config),
        )
        self.trace._replay_view = harness.replay_view
        if on_trace is not None:
            on_trace(self.trace)
        interceptors = [
            (hook_boundary(fn, allow_trace=False), fn)
            for fn in discover_decorated(task, "intercept")
        ]
        stops = [(hook_boundary(fn, allow_trace=True), fn) for fn in task.hooks("stop")]
        self._skip_scoring_stops = {
            fn.__name__ for _, fn in stops if getattr(fn, "stop_skip_scoring", False)
        }
        execution_policy = runtime_config
        self._session = RolloutSession(
            ctx=ctx,
            trace=self.trace,
            network_policy=(
                execution_policy
                if isinstance(execution_policy, NetworkPolicyConfig)
                else NetworkPolicyConfig()
            ),
            trace_stops=[fn for boundary, fn in stops if boundary is Trace],
            limits=limits,
            request_interceptors=[
                fn for boundary, fn in interceptors if boundary is Request
            ],
            response_interceptors=[
                fn for boundary, fn in interceptors if boundary is Response
            ],
            request_stops=[fn for boundary, fn in stops if boundary is Request],
            response_stops=[fn for boundary, fn in stops if boundary is Response],
            gates_tools=harness.SUPPORTS_TOOL_INTERCEPTION,
        )
        self._stack = AsyncExitStack()
        self._failed = False
        self._failure: Exception | None = None
        self._opened = False
        self._deployment_started = False
        self._closed = False
        self._harness_cleaned = False
        self._endpoint: str | None = None
        self._urls: dict[str, str] = {}
        self._harness_session: HarnessSession | None = None
        self._agent_runtime: Runtime | None = None
        self.deadline_at: float | None = None
        """The active harness segment's absolute deadline (event-loop clock), or
        None between segments / when unbounded. An interaction spends one cumulative
        `timeouts.agent` budget only while its own segments run, so time awaiting
        the caller (including another interleaved agent) cannot starve it."""

    @property
    def ok(self) -> bool:
        """Whether the exchange can continue: nothing failed, nothing stopped it."""
        return not self._failed and self.trace.stop_condition is None

    @property
    def closed(self) -> bool:
        """Whether `close()` (or `abort()`) already ran — no further segments."""
        return self._closed

    @property
    def failure(self) -> Exception | None:
        """The original exception most recently captured onto the trace."""
        return self._failure

    def fail(self, error: Exception) -> None:
        """Record `error` as this rollout's outcome (captured onto the trace, the
        remaining stages skipped) — the run's owner reporting a failure the run
        itself couldn't see, e.g. its user raising between segments."""
        if (
            self._borrowed_runtime is not None
            and self._borrowed_runtime.backend.stopped
        ):
            # The owner tore the borrowed box down mid-run — a lifetime bug in the
            # borrowing program: raise to the caller instead of capturing a
            # misattributed error onto the trace.
            raise ValueError(
                f"borrowed runtime {self._borrowed_runtime.name!r} was torn down by its owner "
                "mid-run; keep the provisioning context open until every run "
                "placed into the box has completed"
            ) from error
        if not isinstance(error, RolloutError):
            logger.exception("unexpected error in rollout %s", self.trace.id)
        self._failed = True
        self._failure = error
        self.trace.record_error(error)

    def timeout(self, stage: str) -> None:
        """Record `stage`'s expired deadline as this rollout's outcome: a stop, not
        a failure — no further segments run, but finalize and scoring still do."""
        logger.info("rollout %s: %s timeout", self.trace.id, stage)
        self.trace.record_timeout(stage)

    async def open(self) -> bool:
        """Boot the rollout's world up to the point where segments can run: start
        (or borrow) the runtime, run task + harness setup, bring up the
        interception slot and tool servers. Returns whether the exchange can
        proceed; a setup failure is captured onto the trace."""
        self._opened = True
        self.trace.timing.boot.start = time.time()
        self.trace.notify()
        if self._borrowed_runtime is None:
            self.runtime = make_backend(self.runtime_config, name=self.trace.id).bind()
        elif (
            self._borrowed_runtime is not None
            and self._borrowed_runtime.backend.stopped
        ):
            # A lifetime bug in the borrowing program: raise to the caller instead
            # of capturing onto the trace.
            raise ValueError(
                f"borrowed runtime {self.runtime.name!r} was already torn "
                "down by its owner; keep the provisioning context open for every run "
                "placed into the box"
            )
        runtime = self.runtime
        assert self.trace.agent is not None  # minted with the trace
        self.trace.agent.runtime = runtime.info.model_copy()
        logger.info(
            "rollout start: id=%s task=%s harness=%s runtime=%s",
            self.trace.id,
            self.task.data.idx,
            self.harness.config.name,
            self.runtime_config.type,
        )
        loop = asyncio.get_running_loop()
        setup_timeout: asyncio.Timeout | None = None
        setup_deadline = (
            None if self._timeouts.setup is None else loop.time() + self._timeouts.setup
        )
        try:
            # Provisioning, hooks, and service setup share one budget.
            async with asyncio.timeout_at(setup_deadline) as setup_timeout:
                runtime_env = dict(self.task.runtime_env())
                if self._borrowed_runtime is None:
                    runtime.backend.env = runtime_env
                    runtime = self.runtime = runtime.with_env(runtime_env)
                else:
                    runtime = runtime.with_env({**runtime.env, **runtime_env})
                    self.runtime = runtime
                if self.task.data.prompt is None and not self._has_user:
                    raise TaskError(
                        "task has no prompt and no user to open the conversation; set "
                        "task.prompt, or drive the run through agent.interaction() and open "
                        "it with the first turn(message)"
                    )
                async with boundary(ToolsetError, "building tool servers"):
                    toolsets = self.task.toolsets(self.task.config)
                spec = self.task.deployment_spec()
                self.deployment = runtime.backend.deploy(
                    spec=spec,
                    # Tool servers on Prime VMs of their own boot with the workspace.
                    tool_vms=tool_vms(
                        toolsets,
                        runtime.backend.config,
                        composed=spec is not None,
                        borrowed=self._borrowed_runtime is not None,
                    ),
                    env=runtime.env,
                    config=self._deployment_config,
                    borrowed_runtime=self._borrowed_runtime,
                )
                try:
                    await self.deployment.start()
                    self._deployment_started = True
                    runtime = self.runtime = self.deployment.workspace
                finally:
                    # Providers may assign an ID before startup fails; retain it for diagnostics.
                    self.trace.agent.runtime = runtime.info.model_copy()
                if not self.task.scoring_deferred:
                    isolated = self._scoring_policy.mode == "isolated"
                    self._scoring_task = self.task.grading_task(
                        runtime, scoring=self._scoring_policy
                    )
                    if isolated:
                        self._grader = self._scoring_task
                        self._grading_config = verifier_runtime(
                            self._grader,
                            runtime,
                            # Solver task requirements must not override the verifier's.
                            self.trace.agent.config.runtime,
                            self._scoring_policy,
                        )
                    elif self._scoring_policy.runtime is not None:
                        raise TaskError(
                            "verifier runtime overrides require isolated verification"
                        )
                await runtime.backend.prepare_setup()
                now = time.time()
                self.trace.timing.boot.end = now
                self.trace.timing.setup.start = now
                self.trace.notify()
                async with (
                    boundary(TaskError, "task setup"),
                ):
                    await invoke(
                        self.task.setup,
                        {
                            "trace": self.trace,
                            "runtime": runtime,
                            "deployment": self.deployment,
                        },
                    )
                async with (
                    boundary(HarnessError, "harness setup"),
                ):
                    self._agent_runtime = await resolve_agent_runtime(
                        runtime, self.task.data.agent_user
                    )
                    await self.harness.setup(self._agent_runtime)
                # `base_url` is the interception server's reachable URL for this rollout.
                # The harness reaches the model at `{base_url}/v1`; tool servers reach this
                # rollout's `/state` + `/task` at `base_url` — it's universally reachable
                # (the interception is exposed whenever any consumer is remote).
                (
                    base_url,
                    model_secret,
                    state_secret,
                ) = await self._stack.enter_async_context(
                    serve_interception(
                        self._interception,
                        runtime,
                        self._session,
                        toolsets,
                        self._shared_tools,
                    )
                )
                self._endpoint = runtime.host_url(f"{base_url.rstrip('/')}/v1")
                self._secret = model_secret
                services = (
                    {
                        name: self.deployment.target(name)
                        for name in self.deployment.target_backends
                    }
                    if self.deployment.spec is not None
                    else {}
                )
                self._urls = await self._stack.enter_async_context(
                    serve_tools(
                        toolsets,
                        self._agent_runtime,
                        shared=self._shared_tools,
                        state_secret=state_secret,
                        state_route=self.trace.id,
                        state_base=base_url,
                        services=services,
                        hosts=self.deployment.hosts,
                        # Tool servers on VMs of their own, by server name; an explicit
                        # `service` names only a Compose service.
                        placements={
                            name: self.deployment.target(name)
                            for name in self.deployment.tool_vms
                        },
                    )
                )
                # Setup and service provisioning are complete. Apply the runtime's
                # execution policy while preserving the framework routes the agent uses.
                # Service hosts resolve inside the deployment (Compose names, LAN
                # names), not as egress.
                internal = {*services, *self.deployment.hosts.values()}
                await self.deployment.prepare_execution(
                    [
                        url
                        for url in [self._endpoint, *self._urls.values()]
                        if urlsplit(url).hostname not in internal
                    ],
                    state_route=f"{base_url.rstrip('/')}/state",
                )
                async with (
                    boundary(HarnessError, "opening harness session"),
                ):
                    harness_data = self.trace.task.data
                    policy = self._session.network_policy
                    if (
                        self.trace.agent.config.network_notice
                        and policy.network_restricted
                    ):
                        if not policy.allow:
                            network_prompt = "External network access is disabled."
                        elif policy.block:
                            network_prompt = (
                                "External network access is allowed except for these blocked "
                                f"destinations: {', '.join(policy.block)}. "
                                "Do not circumvent this selection of domains."
                            )
                        else:
                            network_prompt = (
                                "External network access is limited to these destinations: "
                                f"{', '.join(policy.allow)}. "
                                "Do not circumvent this selection of domains."
                            )
                        network_prompt = "\n\n".join(
                            text
                            for text in (harness_data.system_prompt, network_prompt)
                            if text
                        )
                        # Keep existing transcript system messages intact on resume;
                        # user-only lists must stay user-only for ACP harnesses.
                        if (
                            harness_data.system_prompt is None
                            and isinstance(harness_data.prompt, list)
                            and any(m.role == "system" for m in harness_data.prompt)
                        ):
                            harness_data = harness_data.model_copy(
                                update={
                                    "prompt": [
                                        SystemMessage(content=network_prompt),
                                        *harness_data.prompt,
                                    ]
                                }
                            )
                        else:
                            harness_data = harness_data.model_copy(
                                update={"system_prompt": network_prompt}
                            )
                    if (
                        self._session.request_interceptors
                        and harness_data.prompt is not None
                    ):
                        prompt = harness_data.prompt
                        system_prompt = harness_data.system_prompt
                        if isinstance(prompt, str):
                            system_prompt, prompt = self.harness.resolve_prompt(
                                harness_data
                            )
                        messages = (
                            [UserMessage(content=prompt)]
                            if isinstance(prompt, str)
                            else list(prompt)
                        )
                        has_system = system_prompt is not None
                        if has_system:
                            messages.insert(0, SystemMessage(content=system_prompt))
                        prepared, rewrites = await self._session.prepare_users(
                            Request(messages=messages)
                        )
                        messages = prepared.messages
                        self.trace.request_rewrites.extend(rewrites)
                        if has_system:
                            messages = messages[1:]
                        rewritten_prompt = (
                            messages[0].content
                            if isinstance(prompt, str)
                            and len(messages) == 1
                            and isinstance(messages[0], UserMessage)
                            and isinstance(messages[0].content, str)
                            else messages
                        )
                        harness_data = harness_data.model_copy(
                            update={
                                "prompt": rewritten_prompt,
                                "system_prompt": system_prompt,
                            }
                        )
                    if not self._session.stopped:
                        session_kwargs = (
                            {
                                "tool_interception_url": runtime.host_url(
                                    f"{base_url.rstrip('/')}/tool"
                                )
                            }
                            if self.harness.SUPPORTS_TOOL_INTERCEPTION
                            and (
                                self._session.request_interceptors
                                or self._session.request_stops
                            )
                            else {}
                        )
                        self._harness_session = await self.harness.session(
                            self.ctx,
                            self.trace,
                            self._agent_runtime,
                            self._endpoint,
                            self._secret,
                            self._urls,
                            harness_data,
                            **session_kwargs,
                        )
        except TimeoutError as e:
            # Only our expired setup budget owns the timeout; a provider can
            # raise its own TimeoutError before the overall deadline.
            if setup_timeout is not None and setup_timeout.expired():
                self.timeout("setup")
                if not self._deployment_started:
                    self.fail(
                        SandboxError("deployment provisioning exceeded setup timeout")
                    )
            else:
                self.fail(e)
            return False
        except Exception as e:  # noqa: BLE001 - setup boundary records every rollout failure
            self.fail(e)
            return False
        except BaseException:
            # A cancellation mid-setup kills the driver's await with it, so no
            # caller reaches close() — free the started runtime and entered
            # servers here rather than relying on the driver's own guard.
            await self.abort()
            raise
        now = time.time()
        self.trace.timing.setup.end = now
        self.trace.timing.agent.start = now
        self.trace.notify()
        return not self._session.stopped

    async def steer(self, message: str, *, message_id: str | None = None) -> dict:
        if self._closed or not self.ok:
            raise RuntimeError("this rollout is closed or stopped")
        if self._harness_session is None:
            return {"outcome": "promptRequired", "reason": "noRunningTurn"}
        return await self._harness_session.steer(message, message_id=message_id)

    async def step(
        self,
        messages: Messages | None = None,
        *,
        on_input: Callable[[], None] | None = None,
    ) -> bool:
        """Run ONE segment: the harness program to its exit. With `messages`, the
        segment resumes the exchange with the user's turn(s) (`Harness.resume` —
        for an exchange the user opens, this is also the first segment, on an
        empty conversation); without, it launches on the task's own prompt.
        Returns whether the exchange can continue — a refused turn (limit, @stop),
        a failure (an expired agent timeout included), or a segment that made no
        progress all end it."""
        if not self._opened or self._closed or not self.ok:
            return False
        trace = self.trace
        turns_before = trace.num_turns
        root_reply_before = trace.root_reply
        trace.root_reply = None
        loop = asyncio.get_running_loop()
        segment_start = loop.time()
        self.deadline_at = (
            None
            if self._agent_time_remaining is None
            else segment_start + max(0.0, self._agent_time_remaining)
        )
        # Prefer an intercepted model/tool error to the harness exit it caused.
        deadline = asyncio.timeout_at(self.deadline_at)
        try:
            async with deadline:
                assert self._harness_session is not None
                if messages is not None and self._session.request_interceptors:
                    prepared, rewrites = await self._session.prepare_users(
                        Request(messages=messages)
                    )
                    messages = prepared.messages
                    self.trace.request_rewrites.extend(rewrites)
                    if self._session.stopped:
                        return False
                self._session.on_turn_input = on_input
                assert self.deployment is not None
                await self._harness_session.turn(messages)
        except TimeoutError as e:
            # An expired rollout deadline is the agent breaking its time budget: a
            # timeout stop. A TimeoutError from the harness's own I/O with no
            # expired deadline stays the raw failure.
            if self.deadline_at is not None and (loop.time() >= self.deadline_at):
                self.timeout("agent")
            else:
                self.fail(e)
            return False
        except Exception as e:  # noqa: BLE001 - harness boundary records every rollout failure
            if self._session.stopped:
                return False
            # A runtime can report the cancellation the deadline caused as its own
            # error (Prime's SDK raises one for a cancelled request); the expired
            # deadline is still the outcome.
            if deadline.expired():
                self.timeout("agent")
                return False
            real = self._session.error
            if real is not None and isinstance(e, RolloutError):
                real.__cause__ = e
                self.fail(real)
            else:
                self.fail(e)
            return False
        finally:
            self._session.on_turn_input = None
            if trace.num_turns == turns_before:
                trace.root_reply = root_reply_before
            if self._agent_time_remaining is not None:
                self._agent_time_remaining = max(
                    0.0, self._agent_time_remaining - (loop.time() - segment_start)
                )
            self.deadline_at = None
        # A harness that completes cleanly after a failed model call handled it (e.g. it
        # ends its run on context overflow); the failure stays recorded on the call. A
        # harness that dies on it surfaces the stashed error through the except above.
        self._session.error = None
        # A segment that committed nothing can't be waiting on the user; treating
        # it as continuable would consult the user against a conversation that
        # never moved, forever.
        return self.ok and trace.num_turns > turns_before

    async def _cleanup_harness(self, runtime: Runtime) -> None:
        if self._harness_cleaned:
            return
        try:
            await self.harness.cleanup(self.trace, runtime)
        except Exception:
            logger.warning(
                "harness cleanup failed (rollout %s)", self.trace.id, exc_info=True
            )
        self._harness_cleaned = True

    async def abort(self) -> None:
        """Free everything this run holds — the entered servers and an owned
        runtime — without finalizing or scoring: the escape path when an exception
        (a cancellation mid-setup, a lifetime bug raised to the caller) means the
        driver will never reach `close()`. Safe after a partial `close()`."""
        self._closed = True
        if self._harness_session is not None:
            with contextlib.suppress(Exception):
                await self._harness_session.close()
        with contextlib.suppress(Exception):
            await self._stack.aclose()
        if self.runtime is not None:
            with contextlib.suppress(Exception):
                await self._cleanup_harness(self.runtime)
        if self.deployment is not None:
            with contextlib.suppress(Exception):
                await self.deployment.close()
        elif (
            self._borrowed_runtime is None
            and self.runtime is not None
            and not self.runtime.backend.stopped
        ):
            with contextlib.suppress(Exception):
                await self.runtime.backend.stop()

    async def _score_harness(self, runtime: Runtime, deadline: float | None) -> None:
        """Collect best-effort metrics without replacing a task failure."""
        try:
            async with asyncio.timeout_at(deadline):
                await self.harness.score(self.trace, self._agent_runtime or runtime)
        except Exception as error:
            self.trace.info["harness_scoring_error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            logger.warning("harness scoring failed", exc_info=True)

    async def _close_harness(self) -> None:
        """Close the harness session, then the tool servers and interception. Idempotent.

        Generation has already completed, so a teardown failure is logged rather than
        raised: it must not discard an otherwise scoreable (or already scored)
        trajectory."""
        if self._harness_session is not None:
            try:
                await self._harness_session.close()
            except Exception:
                logger.warning(
                    "harness session close failed (rollout %s)",
                    self.trace.id,
                    exc_info=True,
                )
        try:
            await self._stack.aclose()
        except Exception:
            logger.warning(
                "tool server or interception teardown failed (rollout %s)",
                self.trace.id,
                exc_info=True,
            )

    def _agent_finished(self) -> bool:
        """Whether the agent ended its last segment on its own, or waits between the
        segments of an exchange: nothing it runs is still acting on the box."""
        return not self._failed and self.trace.stop_condition in (
            None,
            "agent_completed",
        )

    def _scoring_skip_reason(self) -> str | None:
        """Why the resolved scoring policy, or the `skip_scoring` stop that ended this trace,
        skips task scoring for it, if either does."""
        if self._scoring_policy.skip_on_truncated and self.trace.is_truncated:
            return "truncated"
        if self._scoring_policy.skip_on_timeout and self.trace.is_timeout:
            return "timeout"
        if self.trace.stop_condition in self._skip_scoring_stops:
            return self.trace.stop_condition
        return None

    def _metrics_deadline(self) -> float:
        """Best-effort metrics get at most 30 seconds, and a tenth of a finite scoring budget."""
        budget = (
            min(30.0, self._timeouts.scoring / 10)
            if self._timeouts.scoring is not None
            else 30.0
        )
        return asyncio.get_running_loop().time() + budget

    async def close(self) -> Trace:
        """Finish the rollout: task `finalize` and per-rollout scoring (skipped when the
        run already failed, but a stopped run is complete and scores its partial
        trajectory), the harness session, tool servers and interception, then runtime
        teardown.

        An agent that finished its turn keeps its harness session open while its
        environment is collected and, when grading runs in the agent's own box, graded:
        harnesses stop the processes their commands started when the session closes,
        and a service the task asked the agent to leave running is part of what gets
        graded. A run stopped mid-turn (a timeout or a stop condition) closes the
        session first, so an agent that may still be acting cannot change the box while
        it is graded. Isolated grading runs in its own box after the session closes.
        Harness metrics run after the close, which hands over the session's final state.
        Idempotent; always returns the trace."""
        if self._closed:
            return self.trace
        self._closed = True
        trace = self.trace
        skip_reason = self._scoring_skip_reason()
        runtime = self.runtime
        try:
            if trace.timing.agent.start and not trace.timing.agent.end:
                trace.timing.agent.end = time.time()
            if not self._agent_finished():
                await self._close_harness()
            trace.notify()
            if not self._failed and self._opened:
                assert runtime is not None
                assert self.deployment is not None
                grader = self._grader
                isolated = self._scoring_policy.mode == "isolated"
                loop = asyncio.get_running_loop()
                trace.timing.finalize.start = time.time()
                finalize_deadline = (
                    loop.time() + self._timeouts.finalize
                    if self._timeouts.finalize is not None
                    else None
                )
                scoring_elapsed = 0.0
                metrics_done = False
                finalize_phase = "task finalization"
                try:
                    async with (
                        asyncio.timeout_at(finalize_deadline),
                        boundary(TaskError, "task finalize"),
                    ):
                        await invoke(
                            self.task.finalize,
                            {
                                "trace": trace,
                                "runtime": runtime,
                                "deployment": self.deployment,
                            },
                        )
                        if (
                            self._collect_artifacts or grader is not None
                        ) and not trace.state.artifacts:
                            finalize_phase = "submission collection"
                            trace.state.artifacts = await collect(
                                runtime,
                                self.task.data.artifacts,
                                max_bytes=self.task.data.artifact_max_bytes,
                            )
                    if grader is not None:
                        # The submission is collected: close the solver's session, then
                        # capture its metrics before the workspace stops and the verifier
                        # starts. Neither counts against finalization; the metrics count
                        # against scoring, as they did when they ran first.
                        interlude_started = loop.time()
                        await self._close_harness()
                        metrics_started = loop.time()
                        await self._score_harness(runtime, self._metrics_deadline())
                        metrics_done = True
                        scoring_elapsed = loop.time() - metrics_started
                        if finalize_deadline is not None:
                            finalize_deadline += loop.time() - interlude_started
                        finalize_phase = "service finalization"
                        async with (
                            asyncio.timeout_at(finalize_deadline),
                            boundary(TaskError, "task finalize"),
                        ):
                            await self._cleanup_harness(runtime)
                            await self.deployment.stop_execution()
                            await self.task.finalize_services(
                                trace,
                                {
                                    name: self.deployment.target(name)
                                    for name in self.deployment.target_backends
                                },
                            )
                except TimeoutError:
                    self.timeout("finalize")
                    if isolated:
                        await self._close_harness()
                        if not metrics_done:
                            await self._score_harness(runtime, self._metrics_deadline())
                        raise TaskError(
                            f"{finalize_phase} exceeded its finalize deadline"
                        ) from None
                now = time.time()
                trace.timing.finalize.end = now
                trace.timing.scoring.start = now
                trace.notify()
                scoring_deadline = (
                    loop.time() + max(0.0, self._timeouts.scoring - scoring_elapsed)
                    if self._timeouts.scoring is not None
                    else None
                )
                try:
                    if scoring_deadline is not None and loop.time() >= scoring_deadline:
                        raise TimeoutError("scoring budget exhausted")
                    async with (
                        asyncio.timeout_at(
                            None if grader is not None else scoring_deadline
                        ),
                        boundary(TaskError, "scoring"),
                    ):
                        # Cross-trace judgement runs later, after the runtime is gone.
                        if skip_reason := skip_reason or self._scoring_skip_reason():
                            logger.info(
                                "rollout %s: scoring skipped (%s)",
                                trace.id,
                                skip_reason,
                            )
                            trace.info["scoring_skipped"] = skip_reason
                        elif grader is not None:
                            try:
                                await grade_task(
                                    grader,
                                    trace,
                                    self._grading_config,
                                    self._timeouts,
                                    self._scoring_policy,
                                    self.deployment,
                                    on_score=self._on_score,
                                    scoring_elapsed=scoring_elapsed,
                                )
                            except TimeoutError:
                                self.timeout("scoring")
                                if self._on_score is not None:
                                    self.fail(
                                        TaskError(
                                            "verification exceeded its scoring deadline"
                                        )
                                    )
                        else:
                            with (
                                scoring_transaction(trace)
                                if self._on_score is not None
                                else contextlib.nullcontext()
                            ):
                                scoring_runtime = runtime
                                if not self._scoring_task.scoring_deferred:
                                    scoring_runtime = bind_scoring_env(
                                        runtime, self._scoring_policy
                                    )
                                    await invoke(
                                        self._scoring_task.stage_verifier,
                                        {
                                            "trace": trace,
                                            "runtime": scoring_runtime,
                                        },
                                    )
                                    if self._scoring_policy.network is not None:
                                        await scoring_runtime.set_network(
                                            scoring_network(
                                                self.trace.agent.config.runtime,
                                                self._scoring_policy.network,
                                            )
                                        )
                                await self._scoring_task.score(
                                    trace, scoring_runtime, self.deployment
                                )
                                if self._on_score is not None:
                                    await self._on_score(trace, scoring_runtime)
                except TimeoutError:
                    self.timeout("scoring")
                    if self._on_score is not None:
                        self.fail(
                            TaskError("verification exceeded its scoring deadline")
                        )
                finally:
                    if grader is None:
                        # Graded in the agent's own box: now the session can close, and
                        # the metrics read what it handed over and left behind.
                        await self._close_harness()
                        await self._score_harness(runtime, self._metrics_deadline())
                trace.timing.scoring.end = time.time()
                trace.notify()
        except Exception as e:  # noqa: BLE001 - finalize boundary records every rollout failure
            self.fail(e)
        finally:
            if self._harness_session is not None:
                with contextlib.suppress(Exception):
                    await self._harness_session.close()
            with contextlib.suppress(Exception):
                await self._stack.aclose()
            trace.is_completed = True
            trace.ok = not self._failed
            now = time.time()
            for span in (
                trace.timing.boot,
                trace.timing.setup,
                trace.timing.agent,
                trace.timing.finalize,
                trace.timing.scoring,
            ):
                if span.start and not span.end:
                    span.end = now
            trace.split_agent_time()
            if runtime is not None:
                await self._cleanup_harness(runtime)
            # Tear down here — the env's `score()` (later) needs only the traces,
            # not a live runtime. A borrowed runtime is its creator's to tear down,
            # not this rollout's.
            try:
                if self.deployment is not None:
                    await self.deployment.close()
                elif (
                    self._borrowed_runtime is None
                    and runtime is not None
                    and not runtime.backend.stopped
                ):
                    # Setup can fail before a deployment takes ownership.
                    await runtime.backend.stop()
            except Exception:
                logger.warning("runtime teardown failed", exc_info=True)
        logger.info(
            "rollout done: id=%s task=%s reward=%.3f turns=%d stop=%s%s",
            trace.id,
            self.task.data.idx,
            trace.reward,
            trace.num_turns,
            trace.last_error.type if trace.last_error else trace.stop_condition,
            f" dropped={trace.dropped!r}" if trace.is_dropped else "",
        )
        return trace
