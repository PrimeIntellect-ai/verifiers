"""AWS Bedrock Converse client for verifiers.

Implements the Client interface using boto3's Bedrock Converse API.
Requires the ``bedrock`` extra: ``pip install verifiers[bedrock]``.

Authentication and region use the standard boto3 credential chain and config
(env vars, ~/.aws/credentials, ~/.aws/config, SSO, instance role).
Set ``AWS_PROFILE`` to select a named profile, ``AWS_REGION`` to select a region.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from verifiers.clients.client import Client
from verifiers.types import (
    AssistantMessage,
    ClientConfig,
    FinishReason,
    Messages,
    Response,
    ResponseMessage,
    SamplingArgs,
    SystemMessage,
    TextMessage,
    Tool,
    ToolCall,
    ToolMessage,
    Usage,
    UserMessage,
)


def _setup_bedrock_client(config: ClientConfig) -> Any:
    """Create a boto3 bedrock-runtime client from verifiers config."""
    try:
        import boto3
    except ModuleNotFoundError as e:
        raise ImportError(
            "BedrockConverseClient requires the bedrock extra; install "
            "`verifiers[bedrock]`."
        ) from e

    # boto3 natively respects AWS_PROFILE, AWS_REGION, AWS_ACCESS_KEY_ID,
    # ~/.aws/config, instance roles, etc. No verifiers config repurposing needed.
    session = boto3.Session()
    return session.client("bedrock-runtime")


class BedrockConverseClient(Client[Any, dict, dict, dict]):
    """Verifiers client that calls AWS Bedrock Converse API via boto3."""

    def setup_client(self, config: ClientConfig) -> Any:
        return _setup_bedrock_client(config)

    async def close(self) -> None:
        pass

    async def to_native_prompt(self, messages: Messages) -> tuple[dict, dict]:
        system: list[dict] = []
        conversation: list[dict] = []
        pending_tool_results: list[dict] = []

        def flush_tool_results() -> None:
            nonlocal pending_tool_results
            if not pending_tool_results:
                return
            conversation.append({"role": "user", "content": pending_tool_results})
            pending_tool_results = []

        for msg in messages:
            if isinstance(msg, ToolMessage):
                content_text = msg.content if isinstance(msg.content, str) else str(msg.content)
                pending_tool_results.append({"toolResult": {
                    "toolUseId": msg.tool_call_id,
                    "content": [{"text": content_text}],
                }})
                continue

            flush_tool_results()

            if isinstance(msg, SystemMessage):
                text = msg.content if isinstance(msg.content, str) else _flatten(msg.content)
                system.append({"text": text})
            elif isinstance(msg, UserMessage):
                conversation.append({"role": "user", "content": _to_content_blocks(msg.content)})
            elif isinstance(msg, AssistantMessage):
                blocks: list[dict] = []
                text = msg.content if isinstance(msg.content, str) else _flatten(msg.content)
                if text:
                    blocks.append({"text": text})
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        blocks.append({
                            "toolUse": {
                                "toolUseId": tc.id,
                                "name": tc.name,
                                "input": json.loads(tc.arguments) if isinstance(tc.arguments, str) else tc.arguments,
                            }
                        })
                conversation.append({"role": "assistant", "content": blocks or [{"text": ""}]})
            elif isinstance(msg, TextMessage):
                conversation.append({"role": "user", "content": [{"text": msg.content}]})

        flush_tool_results()

        prompt = {"messages": conversation}
        if system:
            prompt["system"] = system
        return prompt, {}

    async def to_native_tool(self, tool: Tool) -> dict:
        return {
            "toolSpec": {
                "name": tool.name,
                "description": tool.description,
                "inputSchema": {"json": tool.parameters},
            }
        }

    async def get_native_response(
        self,
        prompt: dict,
        model: str,
        sampling_args: SamplingArgs,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> dict:
        kwargs.pop("state", None)

        inference_config: dict[str, Any] = {}
        sa = dict(sampling_args) if sampling_args else {}

        max_tokens = sa.get("max_tokens")
        inference_config["maxTokens"] = max_tokens if max_tokens is not None else 4096
        if sa.get("temperature") is not None:
            inference_config["temperature"] = sa["temperature"]
        if sa.get("top_p") is not None:
            inference_config["topP"] = sa["top_p"]
        stop = sa.get("stop")
        if stop:
            inference_config["stopSequences"] = [stop] if isinstance(stop, str) else stop

        call_kwargs: dict[str, Any] = {
            "modelId": model,
            "messages": prompt["messages"],
        }
        if "system" in prompt:
            call_kwargs["system"] = prompt["system"]
        if inference_config:
            call_kwargs["inferenceConfig"] = inference_config
        if tools:
            call_kwargs["toolConfig"] = {"tools": tools}

        loop = asyncio.get_running_loop()
        try:
            response = await loop.run_in_executor(None, lambda: self.client.converse(**call_kwargs))
        except self.client.exceptions.ValidationException as e:
            error_text = str(e).lower()
            if "too long" in error_text or "context length" in error_text or "token" in error_text:
                from verifiers.errors import OverlongPromptError
                raise OverlongPromptError from e
            raise
        return response

    async def raise_from_native_response(self, response: dict) -> None:
        pass

    async def from_native_response(self, response: dict) -> Response:
        output = response.get("output", {})
        message = output.get("message", {})
        content_blocks = message.get("content", [])

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in content_blocks:
            if "text" in block:
                text_parts.append(block["text"])
            elif "toolUse" in block:
                tu = block["toolUse"]
                tool_calls.append(ToolCall(
                    id=tu["toolUseId"],
                    name=tu["name"],
                    arguments=json.dumps(tu.get("input", {})),
                ))

        stop_reason = response.get("stopReason", "end_turn")
        finish_map: dict[str, FinishReason] = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
        }

        usage = response.get("usage", {})
        input_tokens = usage.get("inputTokens", 0)
        output_tokens = usage.get("outputTokens", 0)

        return Response(
            id=response.get("ResponseMetadata", {}).get("RequestId", ""),
            model="",
            created=int(time.time()),
            usage=Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                reasoning_tokens=0,
                total_tokens=input_tokens + output_tokens,
            ),
            message=ResponseMessage(
                content="".join(text_parts),
                reasoning_content=None,
                thinking_blocks=None,
                tool_calls=tool_calls or None,
                finish_reason=finish_map.get(stop_reason, "stop"),
                is_truncated=(stop_reason == "max_tokens"),
                tokens=None,
            ),
        )


def _to_content_blocks(content: Any) -> list[dict]:
    if isinstance(content, str):
        return [{"text": content}]
    if isinstance(content, list):
        blocks = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    blocks.append({"text": part["text"]})
            elif hasattr(part, "model_dump"):
                dumped = part.model_dump()
                if dumped.get("type") == "text":
                    blocks.append({"text": dumped["text"]})
            elif isinstance(part, str):
                blocks.append({"text": part})
        return blocks or [{"text": ""}]
    return [{"text": str(content)}]


def _flatten(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
            elif hasattr(p, "model_dump"):
                dumped = p.model_dump()
                if dumped.get("type") == "text":
                    parts.append(dumped.get("text", ""))
        return " ".join(parts)
    return str(content)
