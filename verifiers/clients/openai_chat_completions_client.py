import functools
from collections.abc import Iterable, Mapping
from typing import Any, TypeAlias, cast

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
                "exceeds the model's context length",
                "exceed the configured limit",
                "exceeds the configured limit",
                "exceeded model",
                "prompt_too_long",
                "context length",
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

OpenAIChatMessage: TypeAlias = ChatCompletionMessageParam
OpenAIChatMessages: TypeAlias = list[OpenAIChatMessage]
OpenAIChatResponse: TypeAlias = ChatCompletion
OpenAITool: TypeAlias = ChatCompletionToolParam


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
                    content=content_to_text(message.content),
                )
            elif isinstance(message, TextMessage):
                return ChatCompletionUserMessageParam(
                    role="user", content=message.content
                )
            else:
                raise ValueError(f"Invalid chat message: {message}")

        return [from_chat_message(message) for message in messages], {}

    async def to_native_tool(self, tool: Tool) -> OpenAITool:
        return OpenAITool(
            type="function",
            function=FunctionDefinition(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
                strict=tool.strict if tool.strict is not None else True,
            ),
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

        if tools:
            response = await self.client.chat.completions.create(
                model=model,
                messages=prompt,
                tools=tools,
                **normalize_sampling_args(sampling_args),
            )
        else:
            response = await self.client.chat.completions.create(
                model=model,
                messages=prompt,
                **normalize_sampling_args(sampling_args),
            )
        return response

    async def raise_from_native_response(self, response: OpenAIChatResponse) -> None:
        if response is None:
            raise EmptyModelResponseError("Model returned no response")
        if response.choices is None:
            raise EmptyModelResponseError("Model returned no response choices")
        if not len(response.choices) == 1:
            raise InvalidModelResponseError(
                f"Model returned {len(response.choices)} choices, expected 1"
            )
        if not (
            response.choices[0].message.content
            or response.choices[0].message.tool_calls
        ):
            raise EmptyModelResponseError(
                "Model returned no content and did not call any tools"
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
            prompt_mask = [0] * len(prompt_ids)
            completion_ids = getattr(response.choices[0], "token_ids")
            completion_mask = [1] * len(completion_ids)
            if has_logprobs_obj:
                assert response.choices[0].logprobs.content is not None
                logprobs_content = response.choices[0].logprobs.content
                completion_logprobs = [token.logprob for token in logprobs_content]
            else:
                assert isinstance(response.choices[0].logprobs, dict)
                logprobs_content = response.choices[0].logprobs["content"]
                completion_logprobs = [token["logprob"] for token in logprobs_content]
            return ResponseTokens(
                prompt_ids=prompt_ids,
                prompt_mask=prompt_mask,
                completion_ids=completion_ids,
                completion_mask=completion_mask,
                completion_logprobs=completion_logprobs,
            )

        def parse_reasoning_content(response: OpenAIChatResponse) -> str | None:
            message_dict = response.choices[0].message.model_dump()
            if not isinstance(message_dict, dict):
                return None
            for field in DEFAULT_REASONING_FIELDS:
                if field in message_dict:
                    return message_dict[field]
            return None

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
                reasoning_content=parse_reasoning_content(response),
                finish_reason=parse_finish_reason(response),
                is_truncated=parse_is_truncated(response),
                tokens=parse_tokens(response),
                tool_calls=parse_tool_calls(response) or None,
            ),
        )
