"""Lifecycle of one agent rollout."""

import asyncio
import contextlib
import copy
import logging
import time
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass

from verifiers.v1 import steering
from verifiers.v1.box import box_of
from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.agent import AgentConfig
from verifiers.v1.configs.runtime import NetworkPolicyConfig
from verifiers.v1.configs.scoring import resolve_scoring
from verifiers.v1.errors import (
    HarnessError,
    InterceptionError,
    ProviderError,
    RolloutError,
    SandboxError,
    TaskError,
    ToolsetError,
    boundary,
)
from verifiers.v1.grading import (
    bind_scoring_env,
    grade_task,
    scoring_transaction,
    verifier_runtime,
)
from verifiers.v1.harness import Harness, HarnessSession
from verifiers.v1.interception import Interception, serve_interception
from verifiers.v1.mcp import SharedToolServer, serve_tools
from verifiers.v1.runtimes import (
    ModalConfig,
    Runtime,
    RuntimeConfig,
    make_backend,
)
from verifiers.v1.session import RolloutLimits, RolloutSession, hook_boundary
from verifiers.v1.state import state_cls
from verifiers.v1.task import Task
from verifiers.v1.trace import AgentInfo, Checkpoint, SteerRecord, Trace, TraceTask
from verifiers.v1.types import Messages, Request, Response, SystemMessage, UserMessage
from verifiers.v1.utils.artifacts import collect
from verifiers.v1.utils.decorators import discover_decorated, invoke

logger = logging.getLogger(__name__)

OnSave = Callable[[Trace, Checkpoint], None]
"""Observes each per-call save (`AgentConfig.save = "per_call"`): the live trace and the
checkpoint just recorded on it. The caller persists what it needs."""


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


def messaging_toolset():
    """The `message_agent` toolset (`AgentConfig.message_agents`)."""
    from verifiers.v1.mcp import ToolsetConfig
    from verifiers.v1.messaging import AgentMessaging

    return AgentMessaging(ToolsetConfig())


SAVE_CHECKPOINT_ATTEMPTS = 3
"""Checkpoints tried for one per-call save: the platform fails one now and then."""
SAVE_RETRY_DELAY = 1.0
"""Seconds before the second checkpoint try; the third waits twice as long."""


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
        on_save: OnSave | None = None,
        resumes: Trace | None = None,
        peers: dict[str, str] | None = None,
        restored: Checkpoint | None = None,
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
        self._box = box_of(runtime)
        """The owner of a borrowed box `Agent.provision` handed out: it checkpoints the
        box for this run and restores it for a retry."""
        self.box_generation = self._box.generation if self._box is not None else 0
        """Which of the owner's boxes this run started on (`Box.replace`)."""
        self._collect_artifacts = collect_artifacts
        self.trace: Trace = Trace(
            task=TraceTask.of(task),
            state=state_cls(type(task))(),
            # The seat's resolved config, role overrides included — the agent
            # this trace can be reproduced with.
            agent=AgentInfo(config=agent_config),
        )
        self._continues = resumes is not None
        """This run continues a trace: on a box started from a checkpoint, it restores
        the task (`Task.restore`) instead of setting it up."""
        if resumes is not None:
            # A continuation (`resume=`, an interaction retry) links the trace it
            # continues and carries its info, as one session's would.
            self.trace.resumed_from = resumes.id
            self.trace.info = copy.deepcopy(resumes.info)
        if runtime is None and (start := getattr(runtime_config, "checkpoint", None)):
            # An owned box starts from its config's checkpoint.
            restored = Checkpoint(id=start, turns=0, time=time.time())
        if restored is not None:
            if resumes is not None:
                # The continued trace's earlier checkpoints inside the opening
                # conversation, so a replay of this trace restores the box wherever a
                # replay of that one could. First: they are never its latest.
                self.trace.checkpoints.extend(resumes.carried(restored.id))
            # The box starts from this checkpoint, matching the opening conversation.
            # A borrowed box's (`restored`) keeps the time it was taken at, so it
            # orders among the box's other checkpoints as what it is.
            self.trace.checkpoints.append(
                restored.model_copy(
                    update={"turns": 0, "restored": True, "position": None}
                )
            )
        if on_trace is not None:
            on_trace(self.trace)
        interceptors = [
            (hook_boundary(fn, allow_trace=False), fn)
            for fn in discover_decorated(task, "intercept")
        ]
        stops = [(hook_boundary(fn, allow_trace=True), fn) for fn in task.hooks("stop")]
        execution_policy = runtime_config
        self._session = RolloutSession(
            ctx=ctx,
            trace=self.trace,
            network_policy=(
                NetworkPolicyConfig(allow=[])
                if isinstance(execution_policy, ModalConfig)
                and not execution_policy.network_access
                else execution_policy
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
        self._messaging = agent_config.message_agents and peers is not None
        if self._messaging:
            self._session.peers = peers
        self._on_save = on_save
        if runtime is None:
            saves = agent_config.save == "per_call"
        else:
            # A borrowed box saves through its owner (`Box`), when the run or the
            # provisioning agent saves per call; a box with no owner is never saved.
            box = self._box
            saves = (
                box is not None
                and runtime.backend.supports_checkpoint
                and (agent_config.save == "per_call" or box.save)
            )
        if saves:
            # Last, so it saves the response the harness will actually receive.
            self._session.response_interceptors.append(self.save_call)
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
        self._in_step = False
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
    def session(self) -> RolloutSession:
        """This rollout's interception session (pending steers, interceptors)."""
        return self._session

    @property
    def owns_runtime(self) -> bool:
        """Whether this rollout provisions (and tears down) its box, rather than
        borrowing a live one."""
        return self._borrowed_runtime is None

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

    async def recover(self) -> bool:
        """Clear a model-side failure (a provider or interception error) so the next
        segment can run on the same, still-live box: the error stays on the trace,
        the run is no longer failed. False, and nothing changes, for any other
        failure or a box that is no longer alive."""
        failure = self._failure
        if (
            not self._failed
            or not isinstance(failure, (ProviderError, InterceptionError))
            or self._closed
            or self.runtime is None
        ):
            return False
        try:
            if not await self.runtime.alive():
                return False
        except Exception:  # noqa: BLE001 - an unreachable box is not live
            return False
        self._failed = False
        self._failure = None
        self.trace.ok = True
        self.trace.is_completed = False
        self.trace.stop_condition = None
        return True

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
        steering.register(self)
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
                self.deployment = runtime.backend.deploy(
                    spec=self.task.deployment_spec(),
                    env=runtime.env,
                    config=self._deployment_config,
                    borrowed_runtime=self._borrowed_runtime,
                )
                try:
                    await self.deployment.start()
                    self._deployment_started = True
                    runtime = self.runtime = self.deployment.workspace
                    # the box the task's own hooks see (as its setup does)
                    self._session.runtime = runtime
                finally:
                    # Providers may assign an ID before startup fails; retain it for diagnostics.
                    self.trace.agent.runtime = runtime.info.model_copy()
                interceptors = self._session.response_interceptors
                if (
                    self.save_call in interceptors
                    and not runtime.backend.supports_checkpoint
                ):
                    # A composed workspace is a service container on the provider's
                    # box, which has no checkpoints of its own: nothing to save.
                    logger.warning(
                        "rollout %s: per-call saves are off: the %s workspace has no "
                        "filesystem checkpoints",
                        self.trace.id,
                        type(runtime.backend).__name__,
                    )
                    interceptors.remove(self.save_call)
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
                            self.deployment.verifier_config(),
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
                # A box started from a checkpoint already holds setup's files, but
                # only for the run continuing the trace that took it: a fresh run on
                # such a box (a task of another seat, a judge) sets its own task up.
                restored = self._continues and runtime.backend.restored
                setup = self.task.restore if restored else self.task.setup
                async with (
                    boundary(TaskError, "task restore" if restored else "task setup"),
                ):
                    await invoke(
                        setup,
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
                async with boundary(ToolsetError, "building tool servers"):
                    toolsets = self.task.toolsets(self.task.config)
                    if self._messaging:
                        toolsets = [*toolsets, messaging_toolset()]
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
                self._urls = await self._stack.enter_async_context(
                    serve_tools(
                        toolsets,
                        self._agent_runtime,
                        shared=self._shared_tools,
                        state_secret=state_secret,
                        state_route=self.trace.id,
                        state_base=base_url,
                    )
                )
                # Setup and service provisioning are complete. Apply the runtime's
                # execution policy while preserving the framework routes the agent uses.
                await self.deployment.prepare_execution(
                    [self._endpoint, *self._urls.values()]
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

    async def checkpoint(self, turns: int | None = None) -> Checkpoint:
        """Checkpoint the box this rollout owns and record it on the trace, matching
        the conversation at `turns` (default: the turns completed now; the rule is on
        `Checkpoint`). The record is written once the checkpoint is restorable: a
        failed or cancelled checkpoint raises and leaves no record behind."""
        runtime = self.runtime
        if not self.owns_runtime and self._box is None:
            raise ValueError(
                "a borrowed box is its owner's to checkpoint, and this one has no "
                "owner to do it (provision it with agent.provision)"
            )
        if runtime is None or not runtime.backend.supports_checkpoint:
            backend = None if runtime is None else runtime.backend
            raise ValueError(f"{type(backend).__name__} has no filesystem checkpoints")
        if turns is None:
            turns = self.trace.num_turns
        if self._box is not None and not self.owns_runtime:
            checkpoint_id = (await self._box.checkpoint()).id
        else:
            checkpoint_id = await runtime.backend.checkpoint()
        return self.trace.record_checkpoint(checkpoint_id, turns)

    async def save_call(self, response: Response) -> None:
        """Save the exchange as of this model response, before the harness receives it:
        checkpoint the box (an owned one directly, a borrowed one through its `Box`
        owner) and record it on the trace; `on_save` is called once the call has
        committed to the trace. None of the response's tool calls has run, so the box
        matches the conversation up to the request, plus the response when it is a
        final reply (`Checkpoint` has the rule). Installed only on a box whose runtime
        restores checkpoints (`Agent` checks the config before provisioning). A
        checkpoint failing every try is logged and skipped, so the previous one
        stands."""
        trace = self.trace
        # The response is not committed yet: a final reply leaves the box as it is,
        # so the checkpoint matches the conversation with it.
        turns = trace.num_turns + (0 if response.message.tool_calls else 1)
        for attempt in range(1, SAVE_CHECKPOINT_ATTEMPTS + 1):
            try:
                checkpoint = await self.checkpoint(turns)
                break
            except Exception as error:  # noqa: BLE001 - a missed save only makes a later continuation start further back
                if attempt < SAVE_CHECKPOINT_ATTEMPTS:
                    await asyncio.sleep(SAVE_RETRY_DELAY * attempt)
                    continue
                logger.warning(
                    "rollout %s: checkpoint for a per-call save failed %d times, "
                    "keeping the previous one: %s: %s",
                    trace.id,
                    attempt,
                    type(error).__name__,
                    error,
                )
                return
        if self._on_save is None:
            return
        on_save = self._on_save

        def report() -> None:
            # After the call commits: the trace now holds the request and reply that
            # this checkpoint's conversation cut reaches, so a saved trace resumes exactly.
            try:
                on_save(trace, checkpoint)
            except Exception:
                logger.warning("rollout %s: on_save failed", trace.id, exc_info=True)

        self._session.on_commit = report

    async def steer(
        self,
        message: str,
        *,
        message_id: str | None = None,
        sender: str | None = None,
        queue_when_idle: bool = False,
    ) -> dict:
        """Deliver a user message to the running agent: through the harness's native
        mid-turn steering when it has it, else into the rollout's inbox, which the
        harness's program reads before its next model call (`Harness.SUPPORTS_INBOX`).
        A harness with neither raises: it cannot be steered. A rollout with a user (an
        interaction) between turns answers `promptRequired` so its owner can wake it,
        unless `queue_when_idle` holds the message for the next turn's first model
        call. Every accepted message is recorded on `trace.steers`; a `message_id`
        already there is the same steer again (its text must match)."""
        if self._closed or not self.ok:
            raise RuntimeError("this rollout is closed or stopped")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("steering requires a nonempty user message")
        if message_id is not None:
            for record in self.trace.steers:
                if record.id == message_id:
                    if record.text != message:
                        raise ValueError(
                            "steering message ID reused with different content"
                        )
                    return {"outcome": "injected", "delivery": record.delivery}
        if self._harness_session is not None:
            try:
                receipt = await self._harness_session.steer(
                    message, message_id=message_id
                )
            except NotImplementedError:
                receipt = None
            if receipt is not None and (
                receipt.get("outcome") != "promptRequired"
                or (self._has_user and not queue_when_idle)
                or not self.harness.SUPPORTS_INBOX
            ):
                if receipt.get("outcome") == "injected" and not (
                    message_id is not None
                    and any(record.id == message_id for record in self.trace.steers)
                ):
                    # A concurrent send of the same id, which the harness
                    # deduplicated, is one steer.
                    self.trace.steers.append(
                        SteerRecord(
                            id=message_id,
                            text=message,
                            sender=sender,
                            time=time.time(),
                            delivery="native",
                            call=len(self.trace.calls),
                        )
                    )
                    self.trace.notify()
                return receipt
        if self._has_user and not self._in_step and not queue_when_idle:
            return {"outcome": "promptRequired", "reason": "noRunningTurn"}
        if not self.harness.SUPPORTS_INBOX:
            raise RuntimeError(
                f"harness {self.harness.config.id!r} cannot be steered: it has "
                "no native mid-turn steering and no inbox"
            )
        return self._session.queue_steer(message, message_id=message_id, sender=sender)

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
        try:
            self._in_step = True
            async with asyncio.timeout_at(self.deadline_at):
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
            real = self._session.error
            if real is not None and isinstance(e, RolloutError):
                real.__cause__ = e
                self.fail(real)
            else:
                self.fail(e)
            return False
        finally:
            self._in_step = False
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
        steering.unregister(self)
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

    async def close(self) -> Trace:
        """Finish the rollout: tool servers and interception down, task `finalize`
        and per-rollout scoring (skipped when the run already failed — but a stopped
        run is complete and scores its partial trajectory), then runtime teardown.
        Idempotent; always returns the trace."""
        if self._closed:
            return self.trace
        self._closed = True
        steering.unregister(self)
        trace = self.trace
        runtime = self.runtime
        try:
            if self._harness_session is not None:
                try:
                    await self._harness_session.close()
                except Exception:
                    # Generation already completed. A transport teardown failure
                    # must not discard its otherwise scoreable trajectory.
                    logger.warning(
                        "harness session close failed (rollout %s)",
                        trace.id,
                        exc_info=True,
                    )
            try:
                await self._stack.aclose()
            finally:
                if trace.timing.agent.start and not trace.timing.agent.end:
                    trace.timing.agent.end = time.time()
                trace.notify()
            if not self._failed and self._opened:
                assert runtime is not None
                assert self.deployment is not None
                grader = self._grader
                isolated = self._scoring_policy.mode == "isolated"
                trace.timing.finalize.start = time.time()
                scoring_elapsed = 0.0
                early_metrics = (
                    isolated and grader is not None
                ) or self._on_score is not None
                if early_metrics:
                    scoring_started = asyncio.get_running_loop().time()
                    # Metrics are best-effort: reserve at least 90% of a finite
                    # scoring budget for verification, with a 30-second ceiling.
                    # Capture solver metrics before stopping its workspace or letting
                    # an agentic judge change files and execution state in place.
                    try:
                        async with (
                            asyncio.timeout(
                                min(30.0, self._timeouts.scoring / 10)
                                if self._timeouts.scoring is not None
                                else 30.0
                            ),
                            boundary(HarnessError, "harness scoring"),
                        ):
                            await self.harness.score(
                                trace,
                                self._agent_runtime or runtime,
                            )
                    except Exception as error:
                        # Metrics are best-effort; their failure must not prevent
                        # collecting the submission and running its fresh verifier.
                        trace.info["harness_scoring_error"] = {
                            "type": type(error).__name__,
                            "message": str(error),
                        }
                        logger.warning("harness scoring failed", exc_info=True)
                    scoring_elapsed = (
                        asyncio.get_running_loop().time() - scoring_started
                    )
                finalize_phase = "task finalization"
                try:
                    async with (
                        asyncio.timeout(self._timeouts.finalize),
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
                            finalize_phase = "service finalization"
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
                        if not early_metrics:
                            # Deferred scoring has not captured metrics yet.
                            deadline = (
                                asyncio.get_running_loop().time()
                                + self._timeouts.scoring
                                if self._timeouts.scoring is not None
                                else None
                            )
                            await self._score_harness(runtime, deadline)
                        raise TaskError(
                            f"{finalize_phase} exceeded its finalize deadline"
                        ) from None
                now = time.time()
                trace.timing.finalize.end = now
                trace.timing.scoring.start = now
                trace.notify()
                scoring_deadline = (
                    asyncio.get_running_loop().time()
                    + max(0.0, self._timeouts.scoring - scoring_elapsed)
                    if self._timeouts.scoring is not None
                    else None
                )
                if not early_metrics:
                    # Collect metrics before a shared verifier can exhaust the budget.
                    metrics_budget = (
                        min(30.0, self._timeouts.scoring / 10)
                        if self._timeouts.scoring is not None
                        else 30.0
                    )
                    await self._score_harness(
                        runtime, asyncio.get_running_loop().time() + metrics_budget
                    )
                try:
                    if (
                        scoring_deadline is not None
                        and asyncio.get_running_loop().time() >= scoring_deadline
                    ):
                        raise TimeoutError("scoring budget exhausted")
                    async with (
                        asyncio.timeout_at(
                            None if grader is not None else scoring_deadline
                        ),
                        boundary(TaskError, "scoring"),
                    ):
                        # Cross-trace judgement runs later, after the runtime is gone.
                        if grader is not None:
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
            # The final `ok` and closed spans are set after the last notify above:
            # stream them, or a watcher (a durable-call lookup) sees the trace ended
            # with `ok=False`.
            trace.notify()
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
            "rollout done: id=%s task=%s reward=%.3f turns=%d stop=%s",
            trace.id,
            self.task.data.idx,
            trace.reward,
            trace.num_turns,
            trace.last_error.type if trace.last_error else trace.stop_condition,
        )
        return trace
