"""Agent runs: a harness + model, executed as a first-class unit.

An `AgentSpec` names WHO an agent is — the harness that drives it, the model it calls
(a name resolved against the env's model table, "policy" = the rollout's own model),
where it runs (`placement`), and its budget. The policy rollout is the framework's
built-in agent run; a `JudgeSpec` declares an additional run executed in the rollout's
SCORING stage to grade the finished trace and produce a typed verdict.

`run_agent` is the standalone executor — it owns placement, the interception session
(so an agent run's turns, usage, and budget enforcement reuse the same machinery as the
policy's), input-file materialization, and output collection. `Rollout` calls it for
judges today; future callers (group-scope scorers, RUN-stage teammates) reuse it
unchanged. `run_judges` is the SCORE-stage client: it synthesizes each judge's task,
materializes the rollout's records into the judge's runtime, runs the agent, validates
the verdict against the spec's schema, and records the run (with provenance) onto the
rollout's trace.

A judge's I/O contract is files, so any harness — including off-the-shelf coding
agents — can judge without knowing the framework: inputs are materialized under a
per-run `/tmp/vf-agent/<id>/` directory (never the rollout's workdir, which is the
world being judged), and the verdict is a JSON file the framework validates. A missing
or invalid verdict raises `JudgeError` and fails the rollout — it is never a 0 reward,
which would silently poison group baselines.
"""

import asyncio
import dataclasses
import json
import logging
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any, Literal

from pydantic import BaseModel, Field, SerializeAsAny, ValidationError, model_validator
from pydantic_config import BaseConfig

from verifiers.v1.clients import RolloutContext
from verifiers.v1.errors import JudgeError, RolloutError
from verifiers.v1.harness import HarnessConfig
from verifiers.v1.interception import (
    InterceptionPool,
    InterceptionServer,
    RolloutLimits,
    RolloutSession,
)
from verifiers.v1.runtimes import (
    HOST,
    Runtime,
    RuntimeConfig,
    make_runtime,
    reachable_url,
)
from verifiers.v1.task import WireTask
from verifiers.v1.trace import AgentRun, Trace
from verifiers.v1.types import SamplingConfig

logger = logging.getLogger(__name__)


class AgentBudget(BaseConfig):
    """Framework-enforced caps on one agent run (None = no cap), enforced between turns
    by the run's interception session — the same `RolloutLimits` mechanism that bounds
    the policy rollout, so a runaway judge is halted mid-run, not discovered on the bill."""

    max_turns: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_total_tokens: int | None = None

    def limits(self) -> RolloutLimits:
        return RolloutLimits(
            max_turns=self.max_turns,
            max_input_tokens=self.max_input_tokens,
            max_output_tokens=self.max_output_tokens,
            max_total_tokens=self.max_total_tokens,
        )


class AgentSpec(BaseConfig):
    """A runnable agent: harness + model + placement + budget.

    `model` is a logical name: "policy" is the rollout's own model context; any other
    name resolves against the env's model table (`EnvConfig.models`) — endpoints live
    in run config, never in taskset code. `placement` is where the agent's harness
    runs: `"rollout"` provisions it into the rollout's live runtime (the world the
    policy actually mutated — only available to SCORE-stage runs, after the policy
    finished, so the policy can never observe or tamper with it), or a `RuntimeConfig`
    for its own fresh runtime (a clean room for verification, or a plain subprocess
    for trace-only work)."""

    harness: SerializeAsAny[HarnessConfig] = Field(
        default_factory=lambda: HarnessConfig(id="default")
    )
    """The harness driving this agent. Its `runtime` field is ignored — placement is
    `placement`'s job (the harness config type is shared with the policy harness,
    where `runtime` does place it)."""
    model: str = "policy"
    """Logical model name: "policy", or a key into the env's model table."""
    sampling: SamplingConfig | None = None
    """Sampling override for this run (e.g. temperature 0 for a judge). None inherits
    the resolved model's sampling (the table entry's, or the rollout's for "policy")."""
    placement: Literal["rollout"] | RuntimeConfig = "rollout"
    """Where the harness runs: the rollout's live runtime, or its own fresh one."""
    budget: AgentBudget = AgentBudget()
    trainable: bool | None = None
    """Whether this run's tokens are training data. None (default) = the framework's
    rule: False for judge runs. Only a run sampling from the live policy can ever be
    trainable; setting True on any other model is rejected at run time."""

    @model_validator(mode="before")
    @classmethod
    def _resolve_harness(cls, data):
        """Narrow `harness` to the concrete config type its id resolves to (defaulting
        to the built-in `default` harness) — same resolution as `EnvConfig`, so a spec's
        harness gets its plugin-specific fields (e.g. the default harness's `edit`)."""
        from verifiers.v1.loaders import harness_config_type, narrow_plugin_field

        if isinstance(data, dict):
            narrow_plugin_field(data, "harness", harness_config_type, "default")
        return data


class JudgeSpec(AgentSpec):
    """An agentic judge: an agent run executed in the SCORING stage to grade the
    finished rollout. The framework materializes the rollout's records into the
    judge's runtime, appends the I/O contract (file paths + verdict schema) to
    `prompt`, and validates the verdict it writes back.

    `judges(task)` receives the task, so `prompt` is written already rendered — no
    template language, just an f-string over the task's fields."""

    name: str
    """Unique name within the rollout; the verdict is keyed on it (`verdicts[name]`)
    and the run is recorded as `trace.agents[i].name`."""
    prompt: str
    """The judge's instructions — the grading criteria, fully rendered. The framework
    appends the I/O contract (where the records are, where to write the verdict)."""
    verdict: type[BaseModel]
    """The verdict's schema. The judge must write JSON matching it; the parsed model
    instance is what `@reward` functions receive as `verdicts[name]`."""


# The I/O contract appended to every judge prompt. Files, not framework APIs, so any
# harness can judge; the schema is inlined so the judge needs no side channel.
_JUDGE_CONTRACT = """\

---
You are grading a completed agent rollout against the instructions above. \
The rollout's records are files you can read:
- {dir}/task.json — the task the agent was given
- {dir}/transcript.md — the agent's conversation
- {dir}/trace.json — the full structured trace (rewards so far, metrics, per-turn detail)

When you have reached a verdict, write it to {dir}/verdict.json as a single JSON object \
matching this JSON schema (the file must contain only the JSON object — no prose, no \
markdown fences):

{schema}
"""


def _text(content: Any) -> str:
    """A message's plain text: the string itself, or its text parts joined."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            part.text for part in content if getattr(part, "text", None) is not None
        )
    return ""


def render_transcript(trace: Trace) -> str:
    """The rollout's main conversation (its last branch — the final full context) as
    markdown, one `## role` section per message, tool calls summarized inline. The
    judge-readable view; `trace.json` carries the lossless record."""
    branches = trace.branches
    messages = branches[-1].messages if branches else []
    sections: list[str] = []
    for message in messages:
        lines = [f"## {message.role}"]
        text = _text(getattr(message, "content", None))
        if text:
            lines.append(text)
        for call in getattr(message, "tool_calls", None) or []:
            name = getattr(call, "name", None) or getattr(
                getattr(call, "function", None), "name", "?"
            )
            args = getattr(call, "arguments", None) or getattr(
                getattr(call, "function", None), "arguments", ""
            )
            lines.append(f"[tool call: {name}({args})]")
        sections.append("\n".join(lines))
    return "\n\n".join(sections) + "\n"


def resolve_model(
    spec: AgentSpec,
    ctx: RolloutContext,
    models: Mapping[str, RolloutContext] | None,
) -> RolloutContext:
    """The model context `spec` runs against: the rollout's own for "policy", else the
    named table entry; `spec.sampling` overrides either's sampling when set."""
    if spec.model == "policy":
        resolved = ctx
    else:
        resolved = (models or {}).get(spec.model)
        if resolved is None:
            known = ", ".join(sorted(models)) if models else "none configured"
            raise JudgeError(
                f"agent spec references unknown model {spec.model!r} "
                f"(model table entries: {known}); add it under the env's `models` config"
            )
    if spec.sampling is not None:
        resolved = dataclasses.replace(resolved, sampling=spec.sampling)
    return resolved


@asynccontextmanager
async def _serve_agent_interception(
    pool: InterceptionPool | None, runtime: Runtime, session: RolloutSession
):
    """Yield `(endpoint, secret)` for an agent run — a slot on the shared pool when its
    URL is reachable from the agent's runtime (always, for a tunneled pool; for a
    localhost pool only from a local runtime), else a per-run server bridged to the
    runtime. Mirrors `Rollout._serve_interception`, minus the tool/state channels an
    agent run doesn't use."""
    if pool is not None and (not pool.is_local or runtime.is_local):
        async with pool.acquire(session) as (endpoint, secret, _port, _base):
            yield endpoint, secret
    else:
        async with InterceptionServer() as server:
            secret = server.register(session)
            async with reachable_url(HOST, server.port, consumer=runtime) as url:
                yield f"{url}/v1", secret


async def run_agent(
    spec: AgentSpec,
    trace: Trace,
    *,
    ctx: RolloutContext,
    models: Mapping[str, RolloutContext] | None = None,
    rollout_runtime: Runtime | None = None,
    interception: InterceptionPool | None = None,
    files: Mapping[str, bytes] | None = None,
    collect: Sequence[str] = (),
) -> dict[str, bytes | None]:
    """Execute one agent run: resolve its model and placement, materialize `files`
    into its runtime, drive its harness through its own interception session (turns
    recorded onto `trace`, `spec.budget` enforced between turns), then read back the
    `collect` paths (None per missing file) before an owned runtime is torn down.

    The standalone executor: `trace` is caller-made (so a failed run's partial trace
    is still recordable), and nothing here is judge-specific — judges, group-scope
    scorers, and future RUN-stage agents are all callers of this."""
    from verifiers.v1.loaders import (
        load_harness,
    )  # lazy: loaders pulls in plugin machinery

    mctx = resolve_model(spec, ctx, models)
    if spec.trainable and spec.model != "policy":
        raise JudgeError(
            f"agent spec sets trainable=True but samples from {spec.model!r}, not the "
            "policy — only live-policy tokens can be training data"
        )
    harness = load_harness(spec.harness)
    if spec.placement == "rollout":
        if rollout_runtime is None:
            raise JudgeError(
                "agent placement 'rollout' needs the rollout's live runtime; "
                "none was passed (group-scope callers must use a fresh placement)"
            )
        runtime = rollout_runtime
        owned = False
    else:
        runtime = make_runtime(spec.placement, name=trace.id)
        owned = True
    session = RolloutSession(ctx=mctx, trace=trace, limits=spec.budget.limits())
    collected: dict[str, bytes | None] = {}
    trace.timing.setup.start = time.time()
    try:
        if owned:
            await runtime.start()
        for path, data in (files or {}).items():
            await runtime.write(path, data)
        async with _serve_agent_interception(interception, runtime, session) as (
            endpoint,
            secret,
        ):
            await harness.setup(runtime)
            now = time.time()
            trace.timing.setup.end = now
            trace.timing.generation.start = now
            try:
                await harness.run(mctx, trace, runtime, endpoint, secret, {})
            except RolloutError as e:
                # A model call that failed behind the harness is the real cause —
                # prefer it over the harness's own exit (see `RolloutSession.error`).
                if session.error is not None:
                    raise session.error from e
                raise
            else:
                if session.error is not None:
                    raise session.error
        trace.timing.generation.end = time.time()
        for path in collect:
            try:
                collected[path] = await runtime.read(path)
            except Exception:  # missing file — the caller decides if that's fatal
                collected[path] = None
    finally:
        trace.is_completed = True
        if owned:
            try:
                await runtime.stop()
            except Exception:
                logger.warning(
                    "agent runtime teardown failed (run %s)", trace.id, exc_info=True
                )
    return collected


async def run_judges(
    specs: Sequence[JudgeSpec],
    trace: Trace,
    runtime: Runtime,
    *,
    ctx: RolloutContext,
    models: Mapping[str, RolloutContext] | None = None,
    interception: InterceptionPool | None = None,
) -> dict[str, Any]:
    """Run every judge over the finished rollout, concurrently, and return
    `{name: parsed verdict}` for the taskset's `@reward`s. Each run — succeeded or
    failed — is recorded onto `trace.agents` with its provenance, and its model usage
    folded into `trace.extra_usage`, before any error propagates (a failed judge fails
    the rollout, but its partial trace is still there to debug)."""
    if not specs:
        return {}
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise JudgeError(f"duplicate judge names: {names}")

    async def _run_one(spec: JudgeSpec) -> tuple[str, Any]:
        run_dir = f"/tmp/vf-agent/{uuid.uuid4().hex}"
        verdict_path = f"{run_dir}/verdict.json"
        contract = _JUDGE_CONTRACT.format(
            dir=run_dir,
            schema=json.dumps(spec.verdict.model_json_schema(), indent=2),
        )
        judge_task = WireTask(idx=trace.task.idx, prompt=spec.prompt + contract)
        judge_trace: Trace = Trace(task=judge_task)
        verdict: BaseModel | None = None
        try:
            collected = await run_agent(
                spec,
                judge_trace,
                ctx=ctx,
                models=models,
                rollout_runtime=runtime,
                interception=interception,
                files={
                    f"{run_dir}/task.json": trace.task.model_dump_json(
                        indent=2
                    ).encode(),
                    f"{run_dir}/transcript.md": render_transcript(trace).encode(),
                    f"{run_dir}/trace.json": json.dumps(trace.to_record()).encode(),
                },
                collect=[verdict_path],
            )
            raw = collected.get(verdict_path)
            if raw is None:
                raise JudgeError(
                    f"judge {spec.name!r} finished without writing a verdict to "
                    f"{verdict_path}"
                )
            try:
                verdict = spec.verdict.model_validate_json(raw)
            except ValidationError as e:
                raise JudgeError(
                    f"judge {spec.name!r} wrote an invalid verdict: {e}"
                ) from e
            return spec.name, verdict
        finally:
            # Record the run — provenance, conversation, spend — success or not.
            # Built after the run so the record sees the final judge trace (validation
            # into the wire type snapshots it).
            if judge_trace.usage is not None:
                trace.extra_usage.append(judge_trace.usage)
            trace.agents.append(
                AgentRun(
                    name=spec.name,
                    role="judge",
                    model=spec.model,
                    # Never trainable unless explicitly opted in on the policy model.
                    trainable=bool(spec.trainable) and spec.model == "policy",
                    trace=judge_trace,
                    verdict=None
                    if verdict is None
                    else verdict.model_dump(mode="json"),
                )
            )

    results = await asyncio.gather(*(_run_one(spec) for spec in specs))
    return dict(results)
