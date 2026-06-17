"""v1 -> v0 compatibility for unchanged prime-rl/platform callers.

This module is the inverse of ``verifiers.v1.legacy``: it lets a native v1
Taskset/Harness environment present the old v0 ``Environment`` shape. Current
prime-rl can keep using the v0 ZMQ server, v0 clients, v0 datasets, and v0
``RolloutOutput`` records while the actual rollout lifecycle runs through v1.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
import time
from collections.abc import Mapping
from contextlib import AsyncExitStack
from typing import Any, Callable, get_type_hints

import numpy as np
from datasets import Dataset
from renderers.base import MultiModalData

from verifiers.clients import resolve_client
from verifiers.envs.environment import Environment as V0Environment
from verifiers.parsers.parser import Parser
from verifiers.rubrics.rubric import Rubric
from verifiers.types import (
    AssistantMessage as V0AssistantMessage,
    ClientConfig as V0ClientConfig,
    Message as V0Message,
    Messages as V0Messages,
    Response as V0Response,
    ResponseMessage as V0ResponseMessage,
    ResponseTokens as V0ResponseTokens,
    RolloutInput,
    RolloutOutput,
    RolloutTiming,
    SamplingArgs,
    State,
    SystemMessage as V0SystemMessage,
    TimeSpan as V0TimeSpan,
    TimeSpans as V0TimeSpans,
    Tool as V0Tool,
    ToolCall as V0ToolCall,
    ToolMessage as V0ToolMessage,
    Usage as V0Usage,
    UserMessage as V0UserMessage,
)
from verifiers.utils.client_utils import resolve_client_config
from verifiers.utils.save_utils import make_serializable, state_to_output
from verifiers.v1.clients import RolloutContext
from verifiers.v1.clients.client import Client as V1Client
from verifiers.v1.decorators import discover_decorated
from verifiers.v1.dialects import Dialect
from verifiers.v1.env import EnvConfig, Environment as V1Environment
from verifiers.v1.graph import MessageNode
from verifiers.v1.task import Task
from verifiers.v1.taskset import TasksetConfig
from verifiers.v1.trace import Trace
from verifiers.v1.types import (
    AssistantMessage as V1AssistantMessage,
    Message as V1Message,
    Messages as V1Messages,
    Response as V1Response,
    SamplingConfig,
    SystemMessage as V1SystemMessage,
    Tool as V1Tool,
    ToolCall as V1ToolCall,
    ToolMessage as V1ToolMessage,
    TurnTokens,
    Usage as V1Usage,
    UserMessage as V1UserMessage,
)

logger = logging.getLogger(__name__)

_V1_ENV_FIELDS = {
    "timeout",
    "retries",
    "max_turns",
    "max_input_tokens",
    "max_output_tokens",
    "max_total_tokens",
    "multiplex",
}
_FINISH_REASONS = {"stop", "length", "tool_calls", None}


class V1AsV0Rubric(Rubric):
    """Small rubric shim for Prime-RL's current group-scoring detection."""

    def __init__(self, taskset: object) -> None:
        self.individual_funcs = list(discover_decorated(taskset, "reward"))
        self.group_funcs = list(discover_decorated(taskset, "group_reward"))
        self._group_func_ids = {_func_identity(func) for func in self.group_funcs}
        super().__init__(
            funcs=[*self.individual_funcs, *self.group_funcs],
            weights=[
                float(getattr(func, "_vf_weight", 1.0))
                for func in [*self.individual_funcs, *self.group_funcs]
            ],
            parser=Parser(),
        )

    def _is_group_func(self, func) -> bool:  # type: ignore[override]
        return _func_identity(func) in self._group_func_ids


class V0ClientAsV1Client(V1Client):
    """Adapter from an already-resolved v0 client to the v1 client protocol."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self._states: dict[str, dict[str, Any]] = {}

    async def get_response(
        self,
        dialect: Dialect,
        body: dict,
        model: str,
        sampling_args: SamplingConfig,
        session_id: str | None = None,
    ) -> V1Response:
        prompt, tools = dialect.parse_request(body)
        v0_prompt = [_v1_message_to_v0(message) for message in prompt]
        v0_tools = [_v1_tool_to_v0(tool) for tool in tools] if tools else None
        state = self._state(session_id)
        sampling = _sampling_to_v0(sampling_args)
        response: V0Response = await self.client.get_response(
            prompt=v0_prompt,
            model=model,
            sampling_args=sampling,
            tools=v0_tools,
            state=state,
        )
        self._append_v0_step(state, v0_prompt, response)
        return _v0_response_to_v1(response, model)

    async def close(self) -> None:
        close = getattr(self.client, "close", None)
        if callable(close):
            await close()

    def _state(self, session_id: str | None) -> dict[str, Any]:
        key = session_id or "default"
        if key not in self._states:
            self._states[key] = {
                "trajectory_id": key,
                "trajectory": [],
                "metrics": {},
            }
        return self._states[key]

    def _append_v0_step(
        self,
        state: dict[str, Any],
        prompt: V0Messages,
        response: V0Response,
    ) -> None:
        message = response.message
        completion = [message]
        raw_tokens = message.tokens
        tokens = raw_tokens.model_dump() if raw_tokens is not None else None
        state["trajectory"].append(
            {
                "prompt": prompt,
                "completion": completion,
                "response": response,
                "tokens": tokens,
                "reward": None,
                "advantage": None,
                "is_truncated": bool(message.is_truncated),
                "trajectory_id": state["trajectory_id"],
                "extras": {},
            }
        )


class V1AsV0Environment(V0Environment):
    """A native v1 environment exposed through the classic v0 Environment API."""

    def __init__(
        self,
        v1_env: V1Environment,
        *,
        env_id: str | None = None,
        env_args: dict[str, Any] | None = None,
    ) -> None:
        self.v1_env = v1_env
        self.tasks = v1_env.taskset.load_tasks()
        self.task_by_idx = {int(task.idx): task for task in self.tasks}
        self._stack: AsyncExitStack | None = None
        self._context_lock = asyncio.Lock()
        self._shared_urls: dict[str, str] = {}
        self._interception_pool = None
        super().__init__(
            dataset=self._dataset,
            eval_dataset=self._dataset,
            rubric=V1AsV0Rubric(v1_env.taskset),
            env_id=env_id,
            env_args=env_args or {},
        )

    async def run_rollout(
        self,
        input: RolloutInput,
        client: Any,
        model: str,
        sampling_args: SamplingArgs,
        max_retries: int = 0,
        state_columns: list[str] | None = None,
        env_client=None,
    ) -> RolloutOutput:
        env_client = env_client or self.env_client
        if env_client is not None:
            if not isinstance(client, V0ClientConfig):
                raise ValueError(
                    f"client must be ClientConfig in server mode, got {type(client)}"
                )
            return await env_client.run_rollout(
                input,
                resolve_client_config(client),
                model,
                sampling_args,
                max_retries,
                state_columns,
            )
        trace = (
            await self._run_episode([input], client, model, sampling_args, max_retries)
        )[0]
        return await asyncio.to_thread(trace_to_rollout_output, trace, state_columns)

    async def rollout(
        self,
        input: RolloutInput,
        client: Any,
        model: str,
        sampling_args: SamplingArgs | None = None,
    ) -> State:
        traces = await self._run_episode(
            [input], client, model, sampling_args or {}, max_retries=0
        )
        return trace_to_v0_state(traces[0])

    async def run_group(
        self,
        group_inputs: list[RolloutInput],
        client: Any,
        model: str,
        sampling_args: SamplingArgs,
        max_retries: int = 0,
        state_columns: list[str] | None = None,
        env_client=None,
        **kwargs,
    ) -> list[RolloutOutput]:
        del kwargs
        env_client = env_client or self.env_client
        if env_client is not None:
            if not isinstance(client, V0ClientConfig):
                raise ValueError(
                    f"client must be ClientConfig in server mode, got {type(client)}"
                )
            return await env_client.run_group(
                group_inputs,
                resolve_client_config(client),
                model,
                sampling_args,
                max_retries,
                state_columns,
            )
        traces = await self._run_episode(
            group_inputs, client, model, sampling_args, max_retries
        )
        return list(
            await asyncio.gather(
                *[
                    asyncio.to_thread(trace_to_rollout_output, trace, state_columns)
                    for trace in traces
                ]
            )
        )

    async def _run_episode(
        self,
        inputs: list[RolloutInput],
        client: Any,
        model: str,
        sampling_args: SamplingArgs,
        max_retries: int,
    ) -> list[Trace]:
        if not inputs:
            return []
        example_ids = {int(row["example_id"]) for row in inputs}
        if len(example_ids) != 1:
            raise ValueError(
                f"v1 group inputs must share one example_id: {example_ids}"
            )
        task = self.task_by_idx[next(iter(example_ids))]
        await self._ensure_worker_contexts()
        resolved_client = (
            resolve_client(client) if isinstance(client, V0ClientConfig) else client
        )
        ctx = RolloutContext(
            client=V0ClientAsV1Client(resolved_client),
            model=model,
            sampling=SamplingConfig.model_validate(sampling_args or {}),
        )
        episode = self.v1_env.episode(task, ctx, n=len(inputs))
        if max_retries is not None:
            episode.retry = episode.retry.model_copy(
                update={"max_retries": max(0, int(max_retries))}
            )
        return await episode.run(
            shared_urls=self._shared_urls,
            interception=self._interception_pool,
        )

    async def _ensure_worker_contexts(self) -> None:
        if self._stack is not None:
            return
        async with self._context_lock:
            if self._stack is not None:
                return
            stack = AsyncExitStack()
            self._shared_urls = await stack.enter_async_context(
                self.v1_env.shared_tools(self.tasks)
            )
            self._interception_pool = await stack.enter_async_context(
                self.v1_env.interception_pool()
            )
            self._stack = stack

    async def _teardown(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
            self._interception_pool = None
            self._shared_urls = {}
        await super()._teardown()

    def set_kwargs(self, **kwargs) -> None:
        for key, value in kwargs.items():
            self._apply_v1_kwarg(key, value)
            setter = getattr(super(), f"set_{key}", None)
            if callable(setter):
                setter(value)
            else:
                setattr(self, key, value)

    def _apply_v1_kwarg(self, key: str, value: Any) -> None:
        if key in {"max_output_tokens", "max_total_completion_tokens"}:
            self.v1_env.config.max_output_tokens = (
                int(value) if value is not None else None
            )
            self.v1_env.limits.max_output_tokens = self.v1_env.config.max_output_tokens
        elif key in {"max_total_tokens", "max_seq_len"}:
            self.v1_env.config.max_total_tokens = (
                int(value) if value is not None else None
            )
            self.v1_env.limits.max_total_tokens = self.v1_env.config.max_total_tokens
        elif key == "max_input_tokens":
            self.v1_env.config.max_input_tokens = (
                int(value) if value is not None else None
            )
            self.v1_env.limits.max_input_tokens = self.v1_env.config.max_input_tokens
        elif key == "max_turns":
            self.v1_env.config.max_turns = int(value) if value is not None else None
            self.v1_env.limits.max_turns = self.v1_env.config.max_turns
        elif key == "timeout_seconds":
            timeout = float(value) if value is not None else None
            self.v1_env.config.timeout.rollout = timeout
            self.v1_env.harness_timeout = timeout

    def _dataset(self) -> Dataset:
        return Dataset.from_list([task_to_rollout_row(task) for task in self.tasks])


def as_v0_environment(
    env: Any,
    *,
    env_id: str | None = None,
    env_args: dict[str, Any] | None = None,
) -> Any:
    if isinstance(env, V1AsV0Environment):
        return env
    if isinstance(env, V1Environment):
        return V1AsV0Environment(env, env_id=env_id, env_args=env_args)
    return env


def is_v1_load_environment(load_fn: Callable[..., Any], module: Any) -> bool:
    params = inspect.signature(load_fn).parameters
    if "config" not in params:
        return False
    if hasattr(module, "load_taskset"):
        return True
    annotation = params["config"].annotation
    if annotation is inspect.Parameter.empty:
        return False
    return annotation is EnvConfig or getattr(annotation, "__name__", "") == "EnvConfig"


def load_v1_environment_from_module(
    module: Any,
    env_id: str,
    env_args: dict[str, Any],
) -> V1AsV0Environment:
    config = build_env_config(module, env_id, env_args)
    load_fn = getattr(module, "load_environment", None)
    if callable(load_fn) and is_v1_load_environment(load_fn, module):
        env = load_fn(config)
    else:
        env = V1Environment(config)
    return V1AsV0Environment(env, env_id=env_id, env_args=env_args)


def build_env_config(module: Any, env_id: str, env_args: dict[str, Any]) -> EnvConfig:
    args = dict(env_args)
    taskset_data = dict(args.pop("taskset", {}) or {})
    harness_data = dict(args.pop("harness", {}) or {})
    env_data: dict[str, Any] = {}

    for key in list(args):
        if key in _V1_ENV_FIELDS:
            env_data[key] = args.pop(key)
    taskset_data.update(args)

    taskset_data.setdefault("id", env_id)
    harness_data.setdefault("id", "default")

    # Resolve taskset annotation directly from the imported module when possible. That keeps
    # root v0 loading working even before a hub/local package is resolved by the v1 loader.
    load_taskset = getattr(module, "load_taskset", None)
    if callable(load_taskset):
        config_type = _first_param_annotation(load_taskset, TasksetConfig)
        if isinstance(config_type, type) and issubclass(config_type, TasksetConfig):
            taskset_data = config_type.model_validate(taskset_data)

    return EnvConfig.model_validate(
        {
            **env_data,
            "taskset": taskset_data,
            "harness": harness_data,
        }
    )


def task_to_rollout_row(task: Task) -> dict[str, Any]:
    info = {
        "v1_task": _json_safe(task.model_dump(mode="json", exclude_none=True)),
        "task_name": task.name,
    }
    answer = _task_answer(task)
    return {
        "example_id": int(task.idx),
        "prompt": [_v1_message_to_wire(message) for message in _task_prompt(task)],
        "answer": "" if answer is None else _json_dumps(answer),
        "info": _json_dumps(info),
    }


def trace_to_rollout_output(
    trace: Trace,
    state_columns: list[str] | None = None,
) -> RolloutOutput:
    state = trace_to_v0_state(trace)
    return state_to_output(state, state_columns or [])


def trace_to_v0_state(trace: Trace) -> State:
    prompt = [_v1_message_to_v0(message) for message in _task_prompt(trace.task)]
    completion = [_v1_message_to_v0(message) for message in trace.assistant_messages]
    info = _json_safe(
        {
            **dict(trace.info),
            "v1": {
                "trace_id": trace.id,
                "task_idx": trace.task.idx,
                "num_branches": trace.num_branches,
                "rewards": trace.rewards,
                "stop_condition": trace.stop_condition,
            },
        }
    )
    state = State(
        input={
            "example_id": int(trace.task.idx),
            "prompt": prompt,
            "answer": ""
            if _task_answer(trace.task) is None
            else _json_dumps(_task_answer(trace.task)),
            "info": info,
        }
    )
    state["task"] = _json_safe(trace.task.model_dump(mode="json", exclude_none=True))
    state["prompt"] = prompt
    state["completion"] = completion
    state["answer"] = (
        ""
        if _task_answer(trace.task) is None
        else _json_dumps(_task_answer(trace.task))
    )
    state["info"] = info
    state["reward"] = float(trace.reward)
    state["metrics"] = {k: float(v) for k, v in trace.metrics.items()}
    state["timing"] = _timing_to_v0(trace)
    state["is_completed"] = bool(trace.is_completed)
    state["is_truncated"] = bool(trace.is_truncated)
    state["stop_condition"] = trace.stop_condition
    state["error"] = _error_to_v0(trace.error)
    state["trajectory"] = trace_to_trajectory(trace)
    state["token_usage"] = _token_usage(trace)
    return state


def trace_to_trajectory(trace: Trace) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    sampled = [i for i, node in enumerate(trace.nodes) if node.sampled]
    last_sampled = sampled[-1] if sampled else None
    for node_id in sampled:
        node = trace.nodes[node_id]
        parent_ids = _ancestor_ids(trace.nodes, node.parent)
        prompt_nodes = [trace.nodes[i] for i in parent_ids]
        prompt_messages = [_v1_message_to_v0(n.message) for n in prompt_nodes]
        completion_message = _v1_message_to_v0(node.message)
        prompt_ids = [token for n in prompt_nodes for token in n.token_ids]
        prompt_ids.extend(
            token for token, mask in zip(node.token_ids, node.mask) if not mask
        )
        completion_ids = [
            token for token, mask in zip(node.token_ids, node.mask) if mask
        ]
        completion_logprobs = list(node.logprobs)
        if len(completion_logprobs) < len(completion_ids):
            completion_logprobs.extend(
                [0.0] * (len(completion_ids) - len(completion_logprobs))
            )
        tokens = {
            "prompt_ids": prompt_ids,
            "prompt_mask": [0] * len(prompt_ids),
            "completion_ids": completion_ids,
            "completion_mask": [1] * len(completion_ids),
            "completion_logprobs": completion_logprobs[: len(completion_ids)],
            "overlong_prompt": trace.stop_condition == "context_length",
            "is_truncated": bool(trace.is_truncated or node.finish_reason == "length"),
            "routed_experts": _step_routed_experts([*prompt_nodes, node]),
        }
        mm_data = _step_multi_modal_data(prompt_nodes)
        if mm_data is not None:
            tokens["multi_modal_data"] = mm_data
        response = V0Response(
            id=trace.id,
            created=int(trace.timing.start or time.time()),
            model="",
            usage=V0Usage(
                prompt_tokens=len(prompt_ids),
                reasoning_tokens=0,
                completion_tokens=len(completion_ids),
                total_tokens=len(prompt_ids) + len(completion_ids),
            ),
            message=V0ResponseMessage(
                content=getattr(completion_message, "content", None),
                reasoning_content=getattr(
                    completion_message, "reasoning_content", None
                ),
                tool_calls=getattr(completion_message, "tool_calls", None),
                finish_reason=node.finish_reason,
                is_truncated=bool(trace.is_truncated or node.finish_reason == "length"),
                tokens=V0ResponseTokens(**tokens),
            ),
        )
        out.append(
            {
                "prompt": prompt_messages,
                "completion": [completion_message],
                "response": response,
                "tokens": tokens,
                "reward": float(trace.reward) if node_id == last_sampled else None,
                "advantage": None,
                "is_truncated": bool(
                    trace.is_truncated or node.finish_reason == "length"
                ),
                "trajectory_id": trace.id,
                "extras": {"v1_node_id": node_id, "v1_trace_id": trace.id},
            }
        )
    return out


def _func_identity(func: Callable[..., Any]) -> object:
    return getattr(func, "__func__", func)


def _first_param_annotation(load_fn: Callable[..., Any], default: type) -> type:
    try:
        param = next(iter(inspect.signature(load_fn).parameters.values()))
    except StopIteration:
        return default
    if param.annotation is inspect.Parameter.empty:
        return default
    return get_type_hints(load_fn).get(param.name, param.annotation)


def _sampling_to_v0(sampling: SamplingConfig) -> dict[str, Any]:
    data = sampling.model_dump(exclude_none=True)
    extra = getattr(sampling, "__pydantic_extra__", None)
    if isinstance(extra, dict):
        data.update({k: v for k, v in extra.items() if v is not None})
    return data


def _v1_message_to_v0(message: V1Message) -> V0Message:
    if isinstance(message, V1SystemMessage):
        return V0SystemMessage(content=_content_to_v0(message.content))
    if isinstance(message, V1UserMessage):
        return V0UserMessage(content=_content_to_v0(message.content))
    if isinstance(message, V1AssistantMessage):
        return V0AssistantMessage(
            content=message.content,
            reasoning_content=message.reasoning_content,
            tool_calls=[_v1_tool_call_to_v0(c) for c in message.tool_calls or []]
            or None,
        )
    if isinstance(message, V1ToolMessage):
        return V0ToolMessage(tool_call_id=message.tool_call_id, content=message.content)
    raise TypeError(f"unsupported v1 message type: {type(message)}")


def _v1_message_to_wire(message: V1Message) -> dict[str, Any]:
    return _v1_message_to_v0(message).model_dump(exclude_none=True)


def _content_to_v0(content: Any) -> Any:
    if isinstance(content, list):
        return [
            part.model_dump() if hasattr(part, "model_dump") else part
            for part in content
        ]
    return content


def _v1_tool_to_v0(tool: V1Tool) -> V0Tool:
    return V0Tool(
        name=tool.name,
        description=tool.description,
        parameters=tool.parameters,
        strict=tool.strict,
    )


def _v1_tool_call_to_v0(call: V1ToolCall) -> V0ToolCall:
    return V0ToolCall(id=call.id, name=call.name, arguments=call.arguments)


def _v0_tool_call_to_v1(call: Any) -> V1ToolCall:
    data = call.model_dump() if hasattr(call, "model_dump") else dict(call)
    function = data.get("function") if isinstance(data.get("function"), dict) else data
    return V1ToolCall(
        id=str(data.get("id") or ""),
        name=str(function.get("name") or ""),
        arguments=str(function.get("arguments") or ""),
    )


def _v0_response_to_v1(response: V0Response, model: str) -> V1Response:
    message = response.message
    finish = message.finish_reason if message.finish_reason in _FINISH_REASONS else None
    usage = None
    if response.usage is not None:
        usage = V1Usage(
            prompt_tokens=int(response.usage.prompt_tokens),
            completion_tokens=int(response.usage.completion_tokens),
        )
    v1 = V1Response(
        id=response.id,
        created=response.created,
        model=response.model or model,
        message=V1AssistantMessage(
            content=_assistant_text(message.content),
            reasoning_content=message.reasoning_content,
            tool_calls=[_v0_tool_call_to_v1(c) for c in message.tool_calls or []]
            or None,
        ),
        finish_reason=finish,
        usage=usage,
        tokens=_v0_tokens_to_v1(message.tokens),
    )
    v1.raw = _v0_response_to_chat_completion(response, model)
    return v1


def _v0_tokens_to_v1(tokens: V0ResponseTokens | None) -> TurnTokens | None:
    if tokens is None:
        return None
    attribution = tokens.prompt_attribution
    spans = (
        attribution.message_token_spans()
        if hasattr(attribution, "message_token_spans")
        else None
    )
    return TurnTokens(
        prompt_ids=list(tokens.prompt_ids or []),
        completion_ids=list(tokens.completion_ids or []),
        completion_logprobs=list(tokens.completion_logprobs or []),
        message_spans=spans,
        multi_modal_data=tokens.multi_modal_data,
        routed_experts=tokens.routed_experts,
    )


def _v0_response_to_chat_completion(response: V0Response, model: str) -> dict[str, Any]:
    message = response.message
    wire_message: dict[str, Any] = {
        "role": "assistant",
        "content": _assistant_text(message.content),
    }
    if message.reasoning_content is not None:
        wire_message["reasoning_content"] = message.reasoning_content
    if message.tool_calls:
        wire_message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in message.tool_calls
        ]
    usage = None
    if response.usage is not None:
        usage = {
            "prompt_tokens": int(response.usage.prompt_tokens),
            "completion_tokens": int(response.usage.completion_tokens),
            "total_tokens": int(response.usage.total_tokens),
        }
    return {
        "id": response.id or "vf-v1-compat",
        "object": "chat.completion",
        "created": response.created or int(time.time()),
        "model": response.model or model,
        "choices": [
            {
                "index": 0,
                "message": wire_message,
                "finish_reason": message.finish_reason or "stop",
            }
        ],
        "usage": usage,
    }


def _assistant_text(content: Any) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", "")) for part in content if isinstance(part, Mapping)
        )
    return str(content)


def _task_prompt(task: Task) -> V1Messages:
    messages: V1Messages = []
    if task.system_prompt:
        messages.append(V1SystemMessage(content=task.system_prompt))
    instruction = task.instruction
    if isinstance(instruction, list):
        messages.extend(instruction)
    else:
        messages.append(V1UserMessage(content=str(instruction)))
    return messages


def _task_answer(task: Task) -> Any:
    for key in ("answer", "answer_key", "expected_result"):
        if hasattr(task, key):
            return getattr(task, key)
    return None


def _ancestor_ids(nodes: list[MessageNode], node_id: int | None) -> list[int]:
    out: list[int] = []
    while node_id is not None:
        out.append(node_id)
        node_id = nodes[node_id].parent
    out.reverse()
    return out


def _step_multi_modal_data(nodes: list[MessageNode]) -> MultiModalData | None:
    merged = MultiModalData()
    found = False
    for node in nodes:
        mmd = node.multi_modal_data
        if mmd is None or mmd.is_empty():
            continue
        found = True
        for modality, items in mmd.mm_items.items():
            merged.mm_items.setdefault(modality, []).extend(items)
        for modality, hashes in mmd.mm_hashes.items():
            merged.mm_hashes.setdefault(modality, []).extend(hashes)
        for modality, placeholders in mmd.mm_placeholders.items():
            merged.mm_placeholders.setdefault(modality, []).extend(placeholders)
    return merged if found else None


def _step_routed_experts(nodes: list[MessageNode]) -> dict[str, Any] | None:
    token_nodes = [node for node in nodes if node.token_ids]
    if not token_nodes or any(node.routed_experts is None for node in token_nodes):
        return None
    arrays = [
        node.routed_experts for node in token_nodes if node.routed_experts is not None
    ]
    merged = np.ascontiguousarray(np.concatenate(arrays, axis=0))
    return {
        "data": base64.b64encode(merged.tobytes()).decode("ascii"),
        "shape": list(merged.shape),
        "start": 0,
    }


def _timing_to_v0(trace: Trace) -> RolloutTiming:
    return RolloutTiming(
        start_time=trace.timing.start,
        setup=V0TimeSpan(start=trace.timing.setup.start, end=trace.timing.setup.end),
        generation=V0TimeSpan(
            start=trace.timing.generation.start,
            end=trace.timing.generation.end,
        ),
        scoring=V0TimeSpan(
            start=trace.timing.scoring.start,
            end=trace.timing.scoring.end,
        ),
        env=V0TimeSpans(
            spans=[
                V0TimeSpan(
                    start=trace.timing.finalize.start,
                    end=trace.timing.finalize.end,
                )
            ]
        ),
    )


def _error_to_v0(error: Any) -> dict[str, str] | None:
    if error is None:
        return None
    chain = error.traceback or f"{error.type}: {error.message}"
    return {
        "error": error.type,
        "message": error.message,
        "error_chain_repr": chain,
        "error_chain_str": chain,
    }


def _token_usage(trace: Trace) -> dict[str, float]:
    usage_input = 0
    usage_output = 0
    for node in trace.nodes:
        if node.usage is None:
            continue
        usage_input += int(node.usage.prompt_tokens)
        usage_output += int(node.usage.completion_tokens)
    return {
        "input_tokens": float(usage_input or trace.prompt_len),
        "output_tokens": float(usage_output or trace.completion_len),
        "final_input_tokens": float(trace.prompt_len),
        "final_output_tokens": float(trace.completion_len),
    }


def _json_dumps(value: Any) -> str:
    return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True)


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=make_serializable))
