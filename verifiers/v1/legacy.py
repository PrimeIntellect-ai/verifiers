"""Legacy (v0) environment bridge — the v0 -> v1 adapter, isolated here.

A classic verifiers environment (``verifiers.load_environment(id, **args)`` returning a
``SingleTurnEnv`` / ``MultiTurnEnv`` / ``ToolEnv`` / ... that produces a ``RolloutOutput``)
is served over the **same** v1 ZMQ protocol as a native v1 env, returning v1
``Trace``s. The orchestrator drives it with the same ``EnvClient`` and can't tell v0 from v1.

``LegacyEnvServer`` loads the v0 env once, indexes its dataset by ``task_idx``, runs the v0
rollout through a token-recording renderer client (so training keeps per-turn ids + logprobs),
and maps the v0 ``RolloutOutput`` into a v1 ``Trace`` via ``rollout_output_to_trace``.

This is the only place that imports the v0 ``verifiers`` API; all imports of it are lazy so
v1 stays importable without the v0 package present.
"""

import contextlib
import logging
from pathlib import Path
from typing import Any

import zmq
import zmq.asyncio

from verifiers.v1.clients.config import ClientConfig, TrainClientConfig
from verifiers.v1.serve.server import EnvServer
from verifiers.v1.serve.types import (
    RunGroupRequest,
    RunGroupResponse,
    RunRolloutRequest,
    RunRolloutResponse,
)
from verifiers.v1.task import WireTask
from verifiers.v1 import graph
from verifiers.v1.trace import Error, TimeSpan, Timing, Trace
from verifiers.v1.types import (
    AssistantMessage,
    Response,
    SamplingConfig,
    SystemMessage,
    ToolCall,
    ToolMessage,
    TurnTokens,
    Usage,
    UserMessage,
    content_to_parts,
)

logger = logging.getLogger(__name__)

_FINISH_REASONS = frozenset({"stop", "length", "tool_calls"})


# --- v0 RolloutOutput -> v1 Trace mapping -----------------------------------


def _as_dict(obj: Any) -> Any:
    """v0 rollout objects are pydantic models in-process (messages, ``Response``); coerce
    to plain dicts so the mapping reads them whether they arrive as objects or dicts."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return obj


def _text(content: Any) -> str:
    """Flatten a message ``content`` (str, or a list of content parts) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return "" if content is None else str(content)


def _tool_calls(raw: Any) -> list[ToolCall] | None:
    if not raw:
        return None
    calls: list[ToolCall] = []
    for tc in raw:
        tc = _as_dict(tc)
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function", tc)  # OpenAI shape: {id, function: {name, arguments}}
        calls.append(
            ToolCall(
                id=tc.get("id") or "",
                name=fn.get("name") or "",
                arguments=fn.get("arguments") or "",
            )
        )
    return calls or None


def _to_v1_messages(msgs: Any) -> list:
    out: list = []
    for m in msgs or []:
        m = _as_dict(m)
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "system":
            out.append(SystemMessage(content=content_to_parts(m.get("content"))))
        elif role == "user":
            out.append(UserMessage(content=content_to_parts(m.get("content"))))
        elif role == "assistant":
            out.append(
                AssistantMessage(
                    content=m.get("content"),
                    reasoning_content=m.get("reasoning_content"),
                    tool_calls=_tool_calls(m.get("tool_calls")),
                )
            )
        elif role == "tool":
            out.append(
                ToolMessage(
                    tool_call_id=m.get("tool_call_id") or "",
                    content=content_to_parts(m.get("content")),
                )
            )
    return out


def _to_v1_response(raw: Any, model: str, tokens: TurnTokens | None = None) -> Response:
    raw = _as_dict(raw)
    raw = raw if isinstance(raw, dict) else {}
    msg = _as_dict(raw.get("message"))
    msg = msg if isinstance(msg, dict) else {}
    usage_raw = raw.get("usage")
    usage = None
    if isinstance(usage_raw, dict) and "prompt_tokens" in usage_raw:
        usage = Usage(
            prompt_tokens=int(usage_raw.get("prompt_tokens") or 0),
            completion_tokens=int(usage_raw.get("completion_tokens") or 0),
        )
    # v0 records finish_reason on the response message; v1 puts it on the response.
    finish = msg.get("finish_reason") or raw.get("finish_reason")
    return Response(
        id=str(raw.get("id") or ""),
        created=int(raw.get("created") or 0),
        model=str(raw.get("model") or model),
        message=AssistantMessage(
            content=msg.get("content"),
            reasoning_content=msg.get("reasoning_content"),
            tool_calls=_tool_calls(msg.get("tool_calls")),
        ),
        finish_reason=finish if finish in _FINISH_REASONS else None,
        usage=usage,
        tokens=tokens,
    )


def _to_v1_tokens(raw: Any) -> TurnTokens | None:
    if not isinstance(raw, dict):
        return None
    if not raw.get("completion_ids") and not raw.get("prompt_ids"):
        return None
    return TurnTokens(
        prompt_ids=list(raw.get("prompt_ids") or []),
        completion_ids=list(raw.get("completion_ids") or []),
        completion_logprobs=list(raw.get("completion_logprobs") or []),
    )


def _timing(raw: Any) -> Timing:
    """Map the v0 timing record's generation/scoring durations onto a v1 ``Timing``
    (we only have durations, so each span is encoded as start=0, end=duration)."""

    def _dur(node: Any) -> float:
        if isinstance(node, dict):
            for key in ("duration", "duration_s", "seconds"):
                if isinstance(node.get(key), (int, float)):
                    return float(node[key])
            if isinstance(node.get("ms"), (int, float)):
                return float(node["ms"]) / 1000.0
        return 0.0

    raw = raw or {}
    return Timing(
        generation=TimeSpan(start=0.0, end=_dur(raw.get("generation"))),
        scoring=TimeSpan(start=0.0, end=_dur(raw.get("scoring"))),
    )


def rollout_output_to_trace(out: dict, task_idx: int) -> Trace:
    """Map a v0 ``RolloutOutput`` into a v1 ``Trace``, preserving the meta a native v1
    trace carries: per-turn prompt messages, the response message (content / reasoning /
    tool calls), ``finish_reason`` and ``usage``, the token ids/logprobs, and the task's
    system prompt / prompt / answer. ``is_truncated`` is a computed v1 field derived
    from the final turn's ``finish_reason`` and the stop condition."""
    model = str(out.get("model") or "")

    error = None
    raw_error = out.get("error")
    if raw_error:
        if isinstance(raw_error, dict):
            error = Error(
                type=str(raw_error.get("type") or "Error"),
                message=str(raw_error.get("message") or raw_error),
                traceback=raw_error.get("traceback"),
            )
        else:
            error = Error(type="Error", message=str(raw_error), traceback=None)

    trace: Trace = Trace[WireTask](
        task=_to_wire_task(task_idx, out.get("prompt"), out.get("answer")),
        rewards={"reward": float(out.get("reward") or 0.0)},
        metrics={k: float(v) for k, v in (out.get("metrics") or {}).items()},
        is_completed=bool(out.get("is_completed", True)),
        stop_condition=out.get("stop_condition"),
        errors=[error] if error else [],
        timing=_timing(out.get("timing")),
    )
    # Rebuild the message graph from the v0 steps. v0 tokens carry no per-message spans, so
    # attribution is coarse (the per-turn delta lands on the assistant node) — still linear.
    for step in out.get("trajectory") or []:
        if not isinstance(step, dict):
            continue
        tokens = _to_v1_tokens(step.get("tokens"))
        graph.prepare_turn(trace, _to_v1_messages(step.get("prompt"))).commit(
            _to_v1_response(step.get("response"), model, tokens)
        )
    return trace


def _to_wire_task(task_idx: int, prompt: Any, answer: Any) -> WireTask:
    """Carry the v0 prompt's meta onto the v1 task: the system message becomes
    ``system_prompt``, the user message(s) become ``prompt``, and the reference
    ``answer`` rides along as a taskset-extra field (``WireTask`` allows extras)."""
    system_prompt: str | None = None
    user_texts: list[str] = []
    for m in prompt or []:
        m = _as_dict(m)
        if not isinstance(m, dict):
            continue
        if m.get("role") == "system" and system_prompt is None:
            system_prompt = _text(m.get("content"))
        elif m.get("role") == "user":
            user_texts.append(_text(m.get("content")))
    extra = {"answer": answer} if answer else {}
    return WireTask(
        idx=task_idx,
        prompt="\n\n".join(user_texts),
        system_prompt=system_prompt,
        **extra,
    )


# --- the legacy env server ----------------------------------------------------


class LegacyEnvServer(EnvServer):
    """Serve a classic v0 ``verifiers`` environment over the v1 ZMQ protocol.

    Mirrors ``EnvServer`` (same ``_handle`` / ``run`` / ``run_server``), but loads a v0 env
    via ``verifiers.load_environment`` and runs ``env.run_rollout`` instead of a v1 episode.
    """

    def __init__(
        self,
        env_id: str,
        env_args: dict | None = None,
        address: str = "tcp://127.0.0.1:5000",
        extra_env_kwargs: dict | None = None,
    ) -> None:
        from verifiers import load_environment
        from verifiers.v1.ids import ensure_installed, env_name

        self.address = address
        # Install from the env hub on demand for an `org/name[@version]` id, then load the
        # v0 env by its module name (a local id is already importable).
        module = ensure_installed(env_id)
        self.taskset_id = env_name(env_id)
        self.env = load_environment(module, **(env_args or {}))
        if extra_env_kwargs:  # post-load knobs applied via the v0 env's setters
            self.env.set_kwargs(**extra_env_kwargs)
        # The formatted dataset rows are RolloutInputs (prompt + example_id); index by task_idx.
        # Eval-only v0 envs (e.g. aime2024) define no train split; serve the eval split instead.
        try:
            self.dataset = self.env.get_dataset()
        except ValueError:
            self.dataset = self.env.get_eval_dataset()
        self.tasks = self.dataset  # `len(self.tasks)` drives the `info` response
        self.requires_group_scoring = self.env.requires_group_rollouts
        self._clients: dict[tuple[str, str], Any] = {}

        self.ctx = zmq.asyncio.Context()
        self.frontend = self.ctx.socket(zmq.ROUTER)
        self.frontend.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self.frontend.setsockopt(zmq.SNDHWM, 0)
        self.frontend.setsockopt(zmq.RCVHWM, 0)
        self.frontend.setsockopt(zmq.LINGER, 0)
        self.frontend.bind(self.address)
        self.address = self.frontend.getsockopt_string(zmq.LAST_ENDPOINT)

    def serving(self):
        # The v0 bridge runs its own rollouts (no v1 shared tools / interception), so there
        # are no serving resources to enter.
        return contextlib.nullcontext()

    def _v0_client(self, client_config: ClientConfig, model: str):
        """Translate a v1 ``ClientConfig`` into a v0 client (cached). A renderer config
        (token-in/out, training) builds a v0 renderer client whose tokenizer is pinned to
        ``renderer_model_name`` (the base model) so a LoRA adapter name — served only for
        sampling — never drives tokenizer loading. An OpenAI config (chat-completions, eval)
        builds a v0 chat-completions client. The per-request ``model`` selects the sampling
        target in ``run_rollout``."""
        is_renderer = isinstance(client_config, TrainClientConfig)
        renderer_model = (
            client_config.renderer_model_name if is_renderer else None
        ) or model
        key = (client_config.model_dump_json(), renderer_model)
        if key not in self._clients:
            from verifiers.clients import resolve_client
            from verifiers.types import ClientConfig as V0ClientConfig

            if is_renderer:
                v0_config = V0ClientConfig(
                    client_type="renderer",
                    renderer_config=client_config.renderer,
                    renderer_model_name=renderer_model,
                    renderer_pool_size=client_config.pool_size,
                    api_base_url=client_config.base_url,
                    api_key_var=client_config.api_key_var,
                    extra_headers=dict(client_config.headers or {}),
                )
            else:
                v0_config = V0ClientConfig(
                    client_type="openai_chat_completions",
                    api_base_url=client_config.base_url,
                    api_key_var=client_config.api_key_var,
                    extra_headers=dict(client_config.headers or {}),
                )
            self._clients[key] = resolve_client(v0_config)
        return self._clients[key]

    async def _run_v0(
        self,
        task_idx: int,
        client_config: ClientConfig,
        model: str,
        sampling: SamplingConfig,
    ) -> dict:
        client = self._v0_client(client_config, model)
        return await self.env.run_rollout(
            input=dict(self.dataset[task_idx]),
            client=client,
            model=model,
            sampling_args=sampling.model_dump(exclude_none=True),
            state_columns=["trajectory"],
        )

    async def _run_rollout(self, req: RunRolloutRequest) -> RunRolloutResponse:
        out = await self._run_v0(req.task_idx, req.client, req.model, req.sampling)
        return RunRolloutResponse(
            trace=rollout_output_to_trace(out, req.task_idx).model_dump()
        )

    async def _run_group(self, req: RunGroupRequest) -> RunGroupResponse:
        client = self._v0_client(req.client, req.model)
        # run_group scores the rollouts together so group/preference reward funcs apply.
        outs = await self.env.run_group(
            group_inputs=[dict(self.dataset[req.task_idx]) for _ in range(req.n)],
            client=client,
            model=req.model,
            sampling_args=req.sampling.model_dump(exclude_none=True),
            state_columns=["trajectory"],
        )
        traces = [
            rollout_output_to_trace(out, req.task_idx).model_dump() for out in outs
        ]
        return RunGroupResponse(traces=traces)


# --- in-process v0 eval (the `eval` CLI's `--legacy.id` path) ------------------


def _eval_client(client_config: ClientConfig, model: str):
    """A v0 chat-completions client built from the v1 eval `ClientConfig` (base url + api
    key var + headers). Eval needs no token ids, so this skips the renderer pool the
    training bridge (`_v0_client`) builds."""
    from verifiers.clients import resolve_client
    from verifiers.types import ClientConfig as V0ClientConfig

    return resolve_client(
        V0ClientConfig(
            client_type="openai_chat_completions",
            api_base_url=client_config.base_url,
            api_key_var=client_config.api_key_var,
            extra_headers=dict(getattr(client_config, "headers", None) or {}),
        )
    )


def _legacy_output_dir(config) -> Path:
    """The legacy run's output dir, mirroring the native `output_path` shape but keyed by
    the v0 env id (`outputs/<id>--<model>--legacy/<uuid>`); honors `--output-dir`."""
    from verifiers.v1.ids import env_name

    if config.output_dir is not None:
        return config.output_dir
    name = f"{env_name(config.id)}--{config.model.replace('/', '--')}--legacy"
    return Path("outputs") / name / config.uuid


async def run_legacy_eval(config) -> list[Trace]:
    """In-process v0 eval used by the `eval` CLI when `config.is_legacy` (a legacy `id` is
    set, no v1 `taskset`).

    Loads the v0 env, runs `num_rollouts` per task with bounded concurrency, maps each v0
    `RolloutOutput` to a v1 `Trace` (`rollout_output_to_trace`), persists results as they
    land (the same `results.jsonl` / `config.toml` a native run writes), and returns the
    traces. The v0 env is run directly (`env.run_rollout`, no env server), so this needs no
    runtime / interception server. All v0 specifics live here; the CLI only branches on
    `config.is_legacy`."""
    import asyncio
    import random

    from verifiers import load_environment

    from verifiers.v1.cli.output import append_trace, save_config
    from verifiers.v1.ids import ensure_installed

    # Install from the env hub on demand for an `org/name[@version]` id (a local id is
    # already importable), then load by module name.
    env = load_environment(ensure_installed(config.id), **(config.args or {}))
    if config.extra_env_kwargs:  # post-load knobs (max_total_completion_tokens, …)
        env.set_kwargs(**config.extra_env_kwargs)
    dataset = env.get_eval_dataset()  # the eval split (falls back to train when unset)
    idxs = list(range(len(dataset)))
    if config.shuffle:
        random.Random(0).shuffle(idxs)  # fixed seed: same sample every run
    if config.num_tasks is not None:
        idxs = idxs[: config.num_tasks]

    client = _eval_client(config.client, config.model)
    sampling_args = config.sampling.model_dump(exclude_none=True)
    out_dir = _legacy_output_dir(config)
    save_config(config, out_dir)
    logger.info("results: %s", out_dir)
    logger.info(
        "running %dx%d v0 rollouts on %s (legacy: %s)",
        len(idxs),
        config.num_rollouts,
        config.model,
        config.id,
    )

    sem = asyncio.Semaphore(config.max_concurrent) if config.max_concurrent else None

    async def run_one(task_idx: int) -> Trace:
        async def go() -> Trace:
            out = await env.run_rollout(
                input=dict(dataset[task_idx]),
                client=client,
                model=config.model,
                sampling_args=sampling_args,
                state_columns=["trajectory"],
            )
            trace = rollout_output_to_trace(out, task_idx)
            append_trace(out_dir, trace)
            return trace

        if sem is None:
            return await go()
        async with sem:
            return await go()

    # `num_rollouts` rollouts per selected task, all bounded by the one semaphore.
    coros = [run_one(i) for i in idxs for _ in range(config.num_rollouts)]
    return list(await asyncio.gather(*coros))
