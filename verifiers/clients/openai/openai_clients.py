import functools
from collections.abc import Iterable, Mapping
from typing import Any, TypeAlias, cast

from openai import AsyncOpenAI, BadRequestError
from openai.types import Completion
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
    ModelError,
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
    State,
    SystemMessage,
    Tool,
    ToolCall,
    ToolMessage,
    Usage,
    UserMessage,
)
from verifiers.utils.client_utils import setup_openai_client
from verifiers.utils.token_utils import get_prompt_ids


def handle_overlong_prompt(func):
    """Decorator to handle overlong prompt errors from the model API."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if isinstance(e, BadRequestError):
                error_text = e.response.text.lower()
                context_length_phrases = [
                    "this model's maximum context length is",
                    "is longer than the model's context length",
                    "exceeds the model's context length",
                    "exceed the configured limit",
                    "exceeds the configured limit",
                    "exceeded model",
                ]
                if any(phrase in error_text for phrase in context_length_phrases):
                    raise OverlongPromptError from e
            raise ModelError from e

    return wrapper


DEFAULT_REASONING_FIELDS = [
    "reasoning",  # vLLM
    "reasoning_content",  # DeepSeek API
]


def _get_usage_field(usage: Any, key: str) -> Any:
    if isinstance(usage, Mapping):
        return usage.get(key)
    return getattr(usage, key, None)


OpenAITextMessages = str
OpenAITextResponse = Completion


def _content_to_text(content: Any) -> str:
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


class OAICompletionsClient(
    Client[
        AsyncOpenAI,
        OpenAITextMessages,
        OpenAITextResponse,
        None,
    ]
):
    """Wrapper for Completions API via AsyncOpenAI client."""

    def setup_client(self, config: ClientConfig) -> AsyncOpenAI:
        return setup_openai_client(config)

    async def to_native_prompt(
        self, messages: Messages
    ) -> tuple[OpenAITextMessages, dict]:
        if isinstance(messages, str):
            return messages, {}
        prompt = ""
        for message in messages:
            if isinstance(message, str):
                prompt += message
            elif isinstance(message, dict):
                prompt += _content_to_text(message.get("content", ""))
            else:
                prompt += _content_to_text(message.content)
        return prompt, {}

    async def to_native_tool(self, tool: Tool) -> None:
        raise ValueError("Tools are not supported for Completions API")

    @handle_overlong_prompt
    async def get_native_response(
        self,
        prompt: OpenAITextMessages,
        model: str,
        sampling_args: SamplingArgs,
        tools: list[None] | None = None,
        **kwargs,
    ) -> OpenAITextResponse:
        assert tools is None, "Tools are not supported for Completions API"

        def normalize_sampling_args(sampling_args: SamplingArgs):
            return {k: v for k, v in sampling_args.items() if v is not None}

        response = await self.client.completions.create(
            model=model,
            prompt=prompt,
            **normalize_sampling_args(sampling_args),
        )
        return response

    async def raise_from_native_response(self, response: OpenAITextResponse) -> None:
        if response is None:
            raise EmptyModelResponseError("Model returned no response")
        if response.choices is None:
            raise EmptyModelResponseError("Model returned no response choices")
        if not len(response.choices) == 1:
            raise InvalidModelResponseError(
                f"Model returned {len(response.choices)} choices, expected 1"
            )
        if not response.choices[0].text:
            raise EmptyModelResponseError("Model returned no content")

    async def from_native_response(self, response: OpenAITextResponse) -> Response:
        def parse_usage(response: OpenAITextResponse) -> Usage | None:
            usage = getattr(response, "usage", None)
            if usage is None:
                return None
            prompt_tokens = _get_usage_field(usage, "prompt_tokens")
            completion_tokens = _get_usage_field(usage, "completion_tokens")
            if not isinstance(prompt_tokens, int) or not isinstance(
                completion_tokens, int
            ):
                prompt_tokens = _get_usage_field(usage, "input_tokens")
                completion_tokens = _get_usage_field(usage, "output_tokens")
            total_tokens = _get_usage_field(usage, "total_tokens")
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

        def parse_finish_reason(response: OpenAITextResponse) -> FinishReason:
            match response.choices[0].finish_reason:
                case "stop":
                    return "stop"
                case "length":
                    return "length"
                case _:
                    return None

        def parse_is_truncated(response: OpenAITextResponse) -> bool:
            return response.choices[0].finish_reason == "length"

        def parse_tokens(response: OpenAITextResponse) -> ResponseTokens | None:
            if not hasattr(response.choices[0], "prompt_token_ids"):
                return None
            if not hasattr(response.choices[0], "token_ids"):
                return None
            if not hasattr(response.choices[0], "logprobs"):
                return None
            if response.choices[0].logprobs is None:
                return None
            if not hasattr(response.choices[0].logprobs, "token_logprobs"):
                return None
            prompt_ids = getattr(response.choices[0], "prompt_token_ids")
            prompt_mask = [0] * len(prompt_ids)
            completion_ids = getattr(response.choices[0], "token_ids")
            completion_mask = [1] * len(completion_ids)
            completion_logprobs = getattr(
                response.choices[0].logprobs, "token_logprobs"
            )
            return ResponseTokens(
                prompt_ids=prompt_ids,
                prompt_mask=prompt_mask,
                completion_ids=completion_ids,
                completion_mask=completion_mask,
                completion_logprobs=completion_logprobs,
            )

        return Response(
            id=response.id,
            created=response.created,
            model=response.model,
            usage=parse_usage(response),
            message=ResponseMessage(
                content=response.choices[0].text,
                finish_reason=parse_finish_reason(response),
                is_truncated=parse_is_truncated(response),
                tokens=parse_tokens(response),
                reasoning_content=None,
                tool_calls=None,
            ),
        )


OpenAIChatMessage: TypeAlias = ChatCompletionMessageParam
OpenAIChatMessages: TypeAlias = list[OpenAIChatMessage]
OpenAIChatResponse: TypeAlias = ChatCompletion
OpenAITool: TypeAlias = ChatCompletionToolParam


class OAIChatCompletionsClient(
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
        def parse_legacy_tool_call(tool_call: Any) -> tuple[str, str, str]:
            if isinstance(tool_call, Mapping):
                tc_id = tool_call.get("id")
                function_obj = tool_call.get("function")
                if isinstance(function_obj, Mapping):
                    name = function_obj.get("name")
                    arguments = function_obj.get("arguments")
                else:
                    name = tool_call.get("name")
                    arguments = tool_call.get("arguments")
            else:
                tc_id = getattr(tool_call, "id", None)
                function_obj = getattr(tool_call, "function", None)
                if function_obj is not None:
                    name = getattr(function_obj, "name", None)
                    arguments = getattr(function_obj, "arguments", None)
                else:
                    name = getattr(tool_call, "name", None)
                    arguments = getattr(tool_call, "arguments", None)

            if not isinstance(tc_id, str):
                tc_id = ""
            if not isinstance(name, str):
                name = ""
            if not isinstance(arguments, str):
                arguments = "{}"
            return tc_id, name, arguments

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

        def from_legacy_chat_message(message: Mapping[str, Any]) -> OpenAIChatMessage:
            role = message.get("role")
            if role == "system":
                return ChatCompletionSystemMessageParam(
                    role="system", content=normalize_content(message.get("content", ""))
                )
            elif role == "user":
                return ChatCompletionUserMessageParam(
                    role="user", content=normalize_content(message.get("content", ""))
                )
            elif role == "assistant":
                tool_calls = message.get("tool_calls")
                if tool_calls is not None:
                    oai_tool_calls: (
                        list[ChatCompletionMessageFunctionToolCallParam] | None
                    ) = []
                    for tool_call in cast(list[Any], tool_calls):
                        tc_id, name, arguments = parse_legacy_tool_call(tool_call)
                        oai_tool_calls.append(
                            ChatCompletionMessageFunctionToolCallParam(
                                type="function",
                                id=tc_id,
                                function=Function(
                                    name=name,
                                    arguments=arguments,
                                ),
                            )
                        )
                else:
                    oai_tool_calls = None
                return ChatCompletionAssistantMessageParam(
                    role="assistant",
                    content=cast(Any, normalize_content(message.get("content", ""))),
                    tool_calls=cast(Any, oai_tool_calls),
                    reasoning_content=message.get("reasoning_content"),  # type: ignore[arg-type]
                )
            elif role == "tool":
                tool_call_id = message.get("tool_call_id")
                if not isinstance(tool_call_id, str):
                    tool_call_id = ""
                return ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=tool_call_id,
                    content=message.get("content", ""),
                )
            else:
                raise ValueError(f"Invalid chat message: {message}")

        def from_chat_message(
            message: Message | Mapping[str, Any],
        ) -> OpenAIChatMessage:
            if isinstance(message, Mapping):
                return from_legacy_chat_message(cast(Mapping[str, Any], message))
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
                    content=_content_to_text(message.content),
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

    @handle_overlong_prompt
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
            prompt_tokens = _get_usage_field(usage, "prompt_tokens")
            completion_tokens = _get_usage_field(usage, "completion_tokens")
            if not isinstance(prompt_tokens, int) or not isinstance(
                completion_tokens, int
            ):
                prompt_tokens = _get_usage_field(usage, "input_tokens")
                completion_tokens = _get_usage_field(usage, "output_tokens")
            total_tokens = _get_usage_field(usage, "total_tokens")
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
                reasoning_content=parse_reasoning_content(response)
                if self.interleaved_thinking
                else None,
                finish_reason=parse_finish_reason(response),
                is_truncated=parse_is_truncated(response),
                tokens=parse_tokens(response),
                tool_calls=parse_tool_calls(response),
            ),
        )


class OAIChatCompletionsTokenClient(OAIChatCompletionsClient):
    """Wrapper for custom vLLM route /v1/chat/completions/tokens via AsyncOpenAI client. To be used for interleaved thinking."""

    @handle_overlong_prompt
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
            sampling_args["logprobs"] = True
            extra_body = dict(return_token_ids=True)
            if "extra_body" in sampling_args:
                sampling_args["extra_body"].update(extra_body)
            else:
                sampling_args["extra_body"] = extra_body
            return sampling_args

        sampling_args = normalize_sampling_args(sampling_args)
        state = cast(State, kwargs.pop("state"))
        # use /v1/chat/completions for first turn to avoid redundant tokenization
        if len(state["trajectory"]) == 0:
            return await super().get_native_response(
                prompt, model, sampling_args, tools
            )
        prompt_ids = await get_prompt_ids(state, prompt, tools, self.client)
        extra_body = sampling_args.pop("extra_body", {})
        body = dict(
            model=model,
            messages=prompt,
            tools=tools,
            tokens=prompt_ids,
            **sampling_args,
            **extra_body,
        )

        return await self.client.post(
            "/chat/completions/tokens",
            body=body,
            cast_to=ChatCompletion,
        )
