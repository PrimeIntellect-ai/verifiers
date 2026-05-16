import functools
from collections.abc import Iterable, Mapping
from typing import Any, TypeAlias, cast

import httpx
from openai import (
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    PermissionDeniedError,
)
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_message_function_tool_call_param import (
    ChatCompletionMessageFunctionToolCallParam,
    Function,
)
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion_tool_message_param import (
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)
from openai.types.shared_params import FunctionDefinition

from verifiers.clients.client import Client
from verifiers.errors import (
    EmptyModelResponseError,
    InvalidModelResponseError,
    OverlongPromptError,
)
from verifiers.types import (
    AssistantMessage,
    ClientConfig,
    FinishReason,
    Message,
    Messages,
    Response,
    ResponseMessage,
    ResponseTokens,
    SamplingArgs,
    SystemMessage,
    TextMessage,
    Tool,
    ToolCall,
    ToolMessage,
    Usage,
    UserMessage,
)
from verifiers.utils.client_utils import setup_openai_client
from verifiers.utils.response_utils import parse_routed_experts


def handle_openai_overlong_prompt(func):
    """Decorator to handle overlong prompt errors from the model API."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except (AuthenticationError, PermissionDeniedError):
            raise
        except BadRequestError as e:
            error_text = e.response.text.lower()
            context_length_phrases = [
                "this model's maximum context length is",
                "is longer than the model's context length",
                "is longer than the maximum model length",
                "exceeds the model's context length",
                "exceed the configured limit",
                "exceeds the configured limit",
                "exceeded model",
                "prompt_too_long",
                "context length",
                "maximum model length",
            ]
            if any(phrase in error_text for phrase in context_length_phrases):
                raise OverlongPromptError from e
            raise

    return wrapper


def get_usage_field(usage: Any, key: str) -> Any:
    """Get the usage field from a Pydantic model or dict."""
    if isinstance(usage, Mapping):
        return usage.get(key)
    return getattr(usage, key, None)


def content_to_text(content: Any) -> str:
    """Get all text content from OAI message content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, Mapping):
                if part.get("type") == "text":
                    text = part.get("text")
                    if isinstance(text, str):
                        chunks.append(text)
                continue
            text = getattr(part, "text", None)
            if isinstance(text, str):
                chunks.append(text)
        return " ".join(chunks).strip()
    return ""


DEFAULT_REASONING_FIELDS = [
    "reasoning",  # vLLM, Together AI, OpenRouter
    "reasoning_content",  # DeepSeek, Qwen/DashScope, SGLang, Fireworks AI, Kimi/Moonshot
    "reasoning_details",  # OpenRouter, MiniMax
]


def parse_reasoning_content(message: Any) -> str | None:
    message_dict = message.model_dump()
    if not isinstance(message_dict, dict):
        return None

    for field in DEFAULT_REASONING_FIELDS:
        value = message_dict.get(field)
        if isinstance(value, str):
            return value
    return None


OpenAIChatMessage: TypeAlias = ChatCompletionMessageParam
OpenAIChatMessages: TypeAlias = list[OpenAIChatMessage]
OpenAIChatResponse: TypeAlias = ChatCompletion
OpenAITool: TypeAlias = ChatCompletionToolParam

_JSON_WS = b" \t\r\n"


def _skip_json_ws(raw: bytes, idx: int, end: int | None = None) -> int:
    if end is None:
        end = len(raw)
    while idx < end and raw[idx] in _JSON_WS:
        idx += 1
    return idx


def _find_json_string_end(raw: bytes, idx: int, end: int | None = None) -> int | None:
    if end is None:
        end = len(raw)
    while idx < end:
        quote = raw.find(b'"', idx, end)
        if quote < 0:
            return None

        backslashes = 0
        prev = quote - 1
        while prev >= idx and raw[prev] == 92:  # backslash
            backslashes += 1
            prev -= 1
        if backslashes % 2 == 0:
            return quote
        idx = quote + 1
    return None


def _find_matching_json_object_end(raw: bytes, idx: int) -> int | None:
    depth = 0
    end = len(raw)
    while idx < end:
        ch = raw[idx]
        if ch == 34:  # quote
            string_end = _find_json_string_end(raw, idx + 1)
            if string_end is None:
                return None
            idx = string_end + 1
            continue
        if ch == 123:  # {
            depth += 1
        elif ch == 125:  # }
            depth -= 1
            if depth == 0:
                return idx + 1
        idx += 1
    return None


def _find_json_key(
    raw: bytes,
    key: bytes,
    *,
    start: int = 0,
    end: int | None = None,
) -> tuple[int, int] | None:
    if end is None:
        end = len(raw)
    pattern = b'"' + key + b'"'
    idx = start
    while idx < end:
        key_idx = raw.find(pattern, idx, end)
        if key_idx < 0:
            return None

        prev = key_idx - 1
        while prev >= start and raw[prev] in _JSON_WS:
            prev -= 1

        colon_idx = _skip_json_ws(raw, key_idx + len(pattern), end)
        if (
            colon_idx < end
            and raw[colon_idx] == 58  # :
            and (prev < start or raw[prev] in (123, 44))  # { or ,
        ):
            return key_idx, colon_idx

        idx = key_idx + len(pattern)
    return None


def _strip_routed_experts_data(raw: bytes) -> tuple[bytes, memoryview | None]:
    search_start = 0
    while True:
        routed_key = _find_json_key(raw, b"routed_experts", start=search_start)
        if routed_key is None:
            return raw, None

        value_start = _skip_json_ws(raw, routed_key[1] + 1)
        if raw[value_start : value_start + 4] == b"null":
            search_start = value_start + 4
            continue
        if value_start >= len(raw) or raw[value_start] != 123:  # {
            search_start = routed_key[0] + 1
            continue

        object_end = _find_matching_json_object_end(raw, value_start)
        if object_end is None:
            return raw, None

        data_key = _find_json_key(
            raw,
            b"data",
            start=value_start + 1,
            end=object_end - 1,
        )
        if data_key is None:
            search_start = object_end
            continue

        data_quote = _skip_json_ws(raw, data_key[1] + 1, object_end)
        if data_quote >= object_end or raw[data_quote] != 34:  # quote
            search_start = object_end
            continue

        data_start = data_quote + 1
        data_end = _find_json_string_end(raw, data_start, object_end)
        if data_end is None:
            return raw, None

        routed_data = memoryview(raw)[data_start:data_end]
        stripped = raw[:data_start] + raw[data_end:]
        return stripped, routed_data


def _parse_chat_completion_with_routed_experts_sidecar(raw: bytes) -> ChatCompletion:
    stripped, routed_data = _strip_routed_experts_data(raw)
    response = ChatCompletion.model_validate_json(stripped)
    if routed_data is None:
        return response
    if not response.choices:
        return response

    choice_extra = response.choices[0].model_extra or {}
    routed_experts = choice_extra.get("routed_experts")
    if isinstance(routed_experts, dict):
        routed_experts["data"] = routed_data
    return response


async def _post_chat_completion_with_routed_experts_sidecar(
    client: AsyncOpenAI,
    path: str,
    *,
    body: dict[str, Any],
    extra_headers: Any = None,
) -> ChatCompletion:
    raw_response = await client.post(
        path,
        body=body,
        cast_to=httpx.Response,
        options={"headers": extra_headers} if extra_headers else {},
    )
    return _parse_chat_completion_with_routed_experts_sidecar(raw_response.content)


class OpenAIChatCompletionsClient(
    Client[
        AsyncOpenAI,
        OpenAIChatMessages,
        OpenAIChatResponse,
        OpenAITool,
    ]
):
    """Wrapper for Chat Completions API via AsyncOpenAI client."""

    def setup_client(self, config: ClientConfig) -> AsyncOpenAI:
        return setup_openai_client(config)

    async def close(self) -> None:
        await self.client.close()

    async def to_native_prompt(
        self, messages: Messages
    ) -> tuple[OpenAIChatMessages, dict]:
        def normalize_content_part(part: Any) -> dict[str, Any]:
            if isinstance(part, Mapping):
                return dict(part)
            if hasattr(part, "model_dump"):
                return cast(dict[str, Any], part.model_dump())
            raise ValueError(f"Invalid content part type: {type(part)}")

        def normalize_content(content: Any) -> Any:
            if isinstance(content, list):
                return [normalize_content_part(p) for p in content]
            return content

        def from_chat_message(message: Message) -> OpenAIChatMessage:
            if isinstance(message, SystemMessage):
                return ChatCompletionSystemMessageParam(
                    role="system", content=normalize_content(message.content)
                )
            elif isinstance(message, UserMessage):
                return ChatCompletionUserMessageParam(
                    role="user", content=normalize_content(message.content)
                )
            elif isinstance(message, AssistantMessage):
                if message.tool_calls is not None:
                    oai_tool_calls: (
                        list[ChatCompletionMessageFunctionToolCallParam] | None
                    ) = [
                        ChatCompletionMessageFunctionToolCallParam(
                            type="function",
                            id=tool_call.id,
                            function=Function(
                                name=tool_call.name,
                                arguments=tool_call.arguments,
                            ),
                        )
                        for tool_call in message.tool_calls
                    ]
                else:
                    oai_tool_calls = None
                return ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=cast(Any, normalize_content(message.content)),
                    tool_calls=cast(Any, oai_tool_calls),
                    reasoning_content=message.reasoning_content,  # type: ignore[arg-type]
                )
            elif isinstance(message, ToolMessage):
                return ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=message.tool_call_id,
                    content=cast(Any, normalize_content(message.content)),
                )
            elif isinstance(message, TextMessage):
                return ChatCompletionUserMessageParam(
                    role="user", content=message.content
                )
            else:
                raise ValueError(f"Invalid chat message: {message}")

        return [from_chat_message(message) for message in messages], {}

    async def to_native_tool(self, tool: Tool) -> OpenAITool:
        if tool.strict is None:
            function = FunctionDefinition(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
            )
        else:
            function = FunctionDefinition(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
                strict=tool.strict,
            )
        return OpenAITool(
            type="function",
            function=function,
        )

    @handle_openai_overlong_prompt
    async def get_native_response(
        self,
        prompt: OpenAIChatMessages,
        model: str,
        sampling_args: SamplingArgs,
        tools: list[OpenAITool] | None = None,
        **kwargs,
    ) -> OpenAIChatResponse:
        def normalize_sampling_args(sampling_args: SamplingArgs):
            sampling_args = dict(sampling_args)
            api_base_url = None
            if hasattr(self.client, "base_url"):
                api_base_url = str(self.client.base_url)
            elif self._config is not None:
                api_base_url = self._config.api_base_url
            reasoning_effort = sampling_args.pop("reasoning_effort", None)
            model_id = model.lower().split("/")[-1].replace(".", "-").replace("_", "-")
            is_anthropic_route = (
                "openrouter.ai" in (api_base_url or "").lower()
                or "pinference.ai" in (api_base_url or "").lower()
            )
            if (
                reasoning_effort is not None
                and model_id.startswith("claude-")
                and is_anthropic_route
            ):
                # OpenRouter/Pinference route Anthropic reasoning_effort through extra_body.
                extra_body = dict(sampling_args.get("extra_body") or {})
                extra_body["verbosity"] = reasoning_effort
                reasoning = dict(extra_body.get("reasoning") or {})
                reasoning.setdefault("enabled", True)
                extra_body["reasoning"] = reasoning
                sampling_args["extra_body"] = extra_body
            elif reasoning_effort is not None:
                sampling_args["reasoning_effort"] = reasoning_effort
            if "max_tokens" in sampling_args:
                sampling_args["max_completion_tokens"] = sampling_args.pop("max_tokens")
            return {k: v for k, v in sampling_args.items() if v is not None}

        # Audio inputs require text-only modality unless explicitly requested.
        has_audio = False
        for message in prompt:
            content = message.get("content") if isinstance(message, dict) else None
            if isinstance(content, list):
                for part in content:
                    part_type = None
                    if isinstance(part, dict):
                        part_type = str(part.get("type", ""))
                    elif hasattr(part, "type"):
                        part_type = str(getattr(part, "type"))
                    if part_type and part_type.startswith("input_audio"):
                        has_audio = True
                        break
            if has_audio:
                break

        if has_audio and "modalities" not in sampling_args:
            sampling_args = {**sampling_args, "modalities": ["text"]}

        extra_headers = kwargs.pop("extra_headers", None)

        body: dict[str, Any] = dict(
            model=model,
            messages=prompt,
            **normalize_sampling_args(sampling_args),
        )
        if tools:
            body["tools"] = tools

        if hasattr(self.client, "post"):
            return await _post_chat_completion_with_routed_experts_sidecar(
                self.client,
                "/chat/completions",
                body=body,
                extra_headers=extra_headers,
            )

        if tools:
            return await self.client.chat.completions.create(
                model=model,
                messages=prompt,
                tools=tools,
                extra_headers=extra_headers,
                **normalize_sampling_args(sampling_args),
            )
        return await self.client.chat.completions.create(
            model=model,
            messages=prompt,
            extra_headers=extra_headers,
            **normalize_sampling_args(sampling_args),
        )

    async def raise_from_native_response(self, response: OpenAIChatResponse) -> None:
        if response is None:
            raise EmptyModelResponseError("Model returned no response")
        if response.choices is None:
            raise EmptyModelResponseError("Model returned no response choices")
        if not len(response.choices) == 1:
            raise InvalidModelResponseError(
                f"Model returned {len(response.choices)} choices, expected 1"
            )
        message = response.choices[0].message
        has_content = bool(content_to_text(getattr(message, "content", None)))
        has_tool_calls = bool(getattr(message, "tool_calls", None))
        has_reasoning = bool(parse_reasoning_content(message))
        if not (has_content or has_tool_calls or has_reasoning):
            raise EmptyModelResponseError(
                "Model returned no content, reasoning, and did not call any tools"
            )

    async def from_native_response(self, response: OpenAIChatResponse) -> Response:
        def parse_single_tool_call(tool_call: Any) -> ToolCall | None:
            if isinstance(tool_call, ChatCompletionMessageFunctionToolCall):
                return ToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=tool_call.function.arguments,
                )
            if isinstance(tool_call, Mapping):
                tool_call_id = tool_call.get("id")
                function_obj = tool_call.get("function")
                if isinstance(function_obj, Mapping):
                    name = function_obj.get("name")
                    arguments = function_obj.get("arguments")
                else:
                    name = tool_call.get("name")
                    arguments = tool_call.get("arguments")
                if (
                    isinstance(tool_call_id, str)
                    and isinstance(name, str)
                    and isinstance(arguments, str)
                ):
                    return ToolCall(
                        id=tool_call_id,
                        name=name,
                        arguments=arguments,
                    )
                return None

            function_obj = getattr(tool_call, "function", None)
            name = getattr(function_obj, "name", None) if function_obj else None
            arguments = (
                getattr(function_obj, "arguments", None) if function_obj else None
            )
            tool_call_id = getattr(tool_call, "id", None)
            if (
                isinstance(tool_call_id, str)
                and isinstance(name, str)
                and isinstance(arguments, str)
            ):
                return ToolCall(
                    id=tool_call_id,
                    name=name,
                    arguments=arguments,
                )
            return None

        def parse_tool_calls(response: OpenAIChatResponse) -> list[ToolCall]:
            raw_tool_calls = getattr(response.choices[0].message, "tool_calls", None)
            if raw_tool_calls is None:
                return []
            if isinstance(raw_tool_calls, (str, bytes)):
                return []
            if isinstance(raw_tool_calls, Mapping):
                tool_calls_iter = [raw_tool_calls]
            elif isinstance(raw_tool_calls, list):
                tool_calls_iter = raw_tool_calls
            elif isinstance(raw_tool_calls, Iterable):
                tool_calls_iter = list(cast(Iterable[Any], raw_tool_calls))
            else:
                return []
            result: list[ToolCall] = []
            for tool_call in tool_calls_iter:
                parsed = parse_single_tool_call(tool_call)
                if parsed is not None:
                    result.append(parsed)
            return result

        def parse_usage(response: OpenAIChatResponse) -> Usage | None:
            usage = getattr(response, "usage", None)
            if usage is None:
                return None
            prompt_tokens = get_usage_field(usage, "prompt_tokens")
            completion_tokens = get_usage_field(usage, "completion_tokens")
            if not isinstance(prompt_tokens, int) or not isinstance(
                completion_tokens, int
            ):
                prompt_tokens = get_usage_field(usage, "input_tokens")
                completion_tokens = get_usage_field(usage, "output_tokens")
            total_tokens = get_usage_field(usage, "total_tokens")
            if not isinstance(prompt_tokens, int) or not isinstance(
                completion_tokens, int
            ):
                return None
            if not isinstance(total_tokens, int):
                total_tokens = prompt_tokens + completion_tokens
            return Usage(
                prompt_tokens=prompt_tokens,
                reasoning_tokens=0,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

        def parse_is_truncated(response: OpenAIChatResponse) -> bool:
            return getattr(response.choices[0], "finish_reason", None) == "length"

        def parse_finish_reason(response: OpenAIChatResponse) -> FinishReason:
            match getattr(response.choices[0], "finish_reason", None):
                case "stop":
                    return "stop"
                case "length":
                    return "length"
                case "tool_calls":
                    return "tool_calls"
                case _:
                    return None

        def parse_tokens(response: OpenAIChatResponse) -> ResponseTokens | None:
            assert len(response.choices) == 1, "Response should always have one choice"
            choice = response.choices[0]
            if not hasattr(choice, "token_ids"):
                return None
            if not hasattr(response, "prompt_token_ids"):
                return None
            if not hasattr(response.choices[0], "logprobs"):
                return None
            if response.choices[0].logprobs is None:
                return None
            has_logprobs_obj = (
                hasattr(response.choices[0].logprobs, "content")
                and response.choices[0].logprobs.content is not None
            )
            has_logprobs_dict = (
                isinstance(response.choices[0].logprobs, dict)
                and "content" in response.choices[0].logprobs.keys()
                and response.choices[0].logprobs["content"] is not None
            )
            if not (has_logprobs_obj or has_logprobs_dict):
                return None
            prompt_ids = getattr(response, "prompt_token_ids")
            if prompt_ids is None:
                return None
            completion_ids = getattr(response.choices[0], "token_ids")
            if completion_ids is None:
                return None
            prompt_mask = [0] * len(prompt_ids)
            completion_mask = [1] * len(completion_ids)
            if has_logprobs_obj:
                assert response.choices[0].logprobs.content is not None
                logprobs_content = response.choices[0].logprobs.content
                completion_logprobs = [token.logprob for token in logprobs_content]
            else:
                assert isinstance(response.choices[0].logprobs, dict)
                logprobs_content = response.choices[0].logprobs["content"]
                completion_logprobs = [token["logprob"] for token in logprobs_content]

            choice_extra = choice.model_extra or {}
            routed_experts = parse_routed_experts(choice_extra.get("routed_experts"))
            return ResponseTokens(
                prompt_ids=prompt_ids,
                prompt_mask=prompt_mask,
                completion_ids=completion_ids,
                completion_mask=completion_mask,
                completion_logprobs=completion_logprobs,
                routed_experts=routed_experts,
            )

        def parse_reasoning_content_from_response(
            response: OpenAIChatResponse,
        ) -> str | None:
            return parse_reasoning_content(response.choices[0].message)

        response_id = getattr(response, "id", "")
        if not isinstance(response_id, str):
            response_id = ""
        created = getattr(response, "created", 0)
        if not isinstance(created, int):
            created = 0
        model = getattr(response, "model", "")
        if not isinstance(model, str):
            model = ""

        return Response(
            id=response_id,
            created=created,
            model=model,
            usage=parse_usage(response),
            message=ResponseMessage(
                content=response.choices[0].message.content,
                reasoning_content=parse_reasoning_content_from_response(response),
                finish_reason=parse_finish_reason(response),
                is_truncated=parse_is_truncated(response),
                tokens=parse_tokens(response),
                tool_calls=parse_tool_calls(response) or None,
            ),
        )
