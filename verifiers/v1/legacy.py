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

import logging
from typing import Any

import zmq
import zmq.asyncio

from verifiers.v1.clients.config import ClientConfig
from verifiers.v1.serve.server import EnvServer
from verifiers.v1.serve.types import (
    RunGroupRequest,
    RunGroupResponse,
    RunRolloutRequest,
    RunRolloutResponse,
)
from verifiers.v1.task import WireTask
from verifiers.v1.trace import Error, TimeSpan, Timing, Trace, Turn
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
)

logger = logging.getLogger(__name__)


# --- v0 RolloutOutput -> v1 Trace mapping -----------------------------------


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
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "system":
            out.append(SystemMessage(content=_text(m.get("content"))))
        elif role == "user":
            out.append(UserMessage(content=_text(m.get("content"))))
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
                    content=_text(m.get("content")),
                )
            )
    return out


def _to_v1_response(raw: Any, model: str) -> Response:
    raw = raw or {}
    msg = raw.get("message", {}) if isinstance(raw, dict) else {}
    usage_raw = raw.get("usage") if isinstance(raw, dict) else None
    usage = None
    if isinstance(usage_raw, dict) and "prompt_tokens" in usage_raw:
        usage = Usage(
            prompt_tokens=int(usage_raw.get("prompt_tokens") or 0),
            completion_tokens=int(usage_raw.get("completion_tokens") or 0),
        )
    return Response(
        id=str(raw.get("id") or ""),
        created=int(raw.get("created") or 0),
        model=str(raw.get("model") or model),
        message=AssistantMessage(
            content=msg.get("content"),
            reasoning_content=msg.get("reasoning_content"),
            tool_calls=_tool_calls(msg.get("tool_calls")),
        ),
        finish_reason=raw.get("finish_reason"),
        usage=usage,
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
    """Map a v0 ``RolloutOutput`` into a v1 ``Trace``. The three training-critical token
    fields (prompt_ids / completion_ids / completion_logprobs) map 1:1; ``is_truncated`` is
    a computed v1 field, so we rely on the final turn's ``finish_reason`` / stop condition."""
    trajectory = [
        Turn(
            prompt=_to_v1_messages(step.get("prompt")),
            response=_to_v1_response(step.get("response"), str(out.get("model") or "")),
            tokens=_to_v1_tokens(step.get("tokens")),
        )
        for step in (out.get("trajectory") or [])
        if isinstance(step, dict)
    ]

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
        task=WireTask(idx=task_idx, instruction=_prompt_text(out.get("prompt"))),
        trajectory=trajectory,
        rewards={"reward": float(out.get("reward") or 0.0)},
        metrics={k: float(v) for k, v in (out.get("metrics") or {}).items()},
        is_completed=bool(out.get("is_completed", True)),
        stop_condition=out.get("stop_condition"),
        error=error,
        timing=_timing(out.get("timing")),
    )
    return trace


def _prompt_text(prompt: Any) -> str:
    return "\n\n".join(
        _text(m.get("content")) for m in (prompt or []) if isinstance(m, dict)
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
    ) -> None:
        from verifiers import load_environment

        self.address = address
        self.taskset_id = env_id
        self.env = load_environment(env_id, **(env_args or {}))
        # The formatted dataset rows are RolloutInputs (prompt + example_id); index by task_idx.
        self.dataset = self.env.get_dataset()
        self.tasks = self.dataset  # `len(self.tasks)` drives the `info` response
        self.requires_group_scoring = False
        self._clients: dict[tuple[str, str], Any] = {}

        self.ctx = zmq.asyncio.Context()
        self.frontend = self.ctx.socket(zmq.ROUTER)
        self.frontend.setsockopt(zmq.ROUTER_MANDATORY, 1)
        self.frontend.setsockopt(zmq.SNDHWM, 0)
        self.frontend.setsockopt(zmq.RCVHWM, 0)
        self.frontend.setsockopt(zmq.LINGER, 0)
        self.frontend.bind(self.address)
        self.address = self.frontend.getsockopt_string(zmq.LAST_ENDPOINT)

    def _v0_client(self, client_config: ClientConfig, model: str):
        """Translate the v1 renderer ``ClientConfig`` into a v0 renderer client (cached;
        a renderer builds the model's tokenizer pool on first use)."""
        key = (client_config.model_dump_json(), model)
        if key not in self._clients:
            from verifiers.clients import resolve_client
            from verifiers.types import ClientConfig as V0ClientConfig

            v0_config = V0ClientConfig(
                client_type="renderer",
                renderer_config=getattr(client_config, "renderer", None),
                renderer_model_name=model,
                renderer_pool_size=getattr(client_config, "pool_size", None),
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
            trace=rollout_output_to_trace(out, req.task_idx).to_wire()
        )

    async def _run_group(self, req: RunGroupRequest) -> RunGroupResponse:
        traces = []
        for _ in range(req.n):
            out = await self._run_v0(req.task_idx, req.client, req.model, req.sampling)
            traces.append(rollout_output_to_trace(out, req.task_idx).to_wire())
        return RunGroupResponse(traces=traces)
