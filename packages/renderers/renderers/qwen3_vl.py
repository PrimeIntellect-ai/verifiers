"""Qwen3-VL renderer.

Mirrors the Qwen3-VL chat template: Qwen3 JSON tool calls plus multimodal
placeholders for user/tool image and video content.
"""

from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
from transformers import AutoProcessor
from transformers.tokenization_utils import PreTrainedTokenizer

from renderers.base import Message, ParsedResponse, RenderedTokens, ToolSpec
from renderers.bridges import chatml_bridge
from renderers.parsing import parse_qwen3

_TOOLS_HEADER = (
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>"
)

_TOOLS_FOOTER = (
    "\n</tools>\n\n"
    "For each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>"
)

_IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
_VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"


class Qwen3VLRenderer:
    """Deterministic message to token renderer for Qwen3-VL models."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        *,
        add_vision_id: bool = False,
    ):
        self._tokenizer = tokenizer
        self._add_vision_id = add_vision_id

        self._im_start = self._token_id("<|im_start|>")
        self._im_end = self._token_id("<|im_end|>")
        self._endoftext = self._token_id("<|endoftext|>")
        self._tool_call = self._token_id("<tool_call>")
        self._tool_call_end = self._token_id("</tool_call>")
        self._tool_response = self._token_id("<tool_response>")
        self._tool_response_end = self._token_id("</tool_response>")
        self._processor = None

    def _token_id(self, token: str) -> int:
        tid = self._tokenizer.convert_tokens_to_ids(token)
        assert isinstance(tid, int) and tid != self._tokenizer.unk_token_id, (
            f"Special token {token!r} not found in tokenizer vocabulary"
        )
        return tid

    def _encode(self, text: str) -> list[int]:
        if not text:
            return []
        return self._tokenizer.encode(text, add_special_tokens=False)

    def _get_processor(self):
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(
                self._tokenizer.name_or_path,
                trust_remote_code=True,
                use_fast=True,
            )
        return self._processor

    @staticmethod
    def _is_multimodal_item(item: Any) -> bool:
        if not isinstance(item, dict):
            return False
        item_type = item.get("type")
        return item_type in {"image", "image_url", "video"} or any(
            key in item for key in ("image", "image_url", "video")
        )

    @classmethod
    def _has_multimodal_content(cls, messages: list[Message]) -> bool:
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            if any(cls._is_multimodal_item(item) for item in content):
                return True
        return False

    @staticmethod
    def _load_image_from_item(item: dict[str, Any]) -> Image.Image | None:
        if item.get("type") == "image":
            image = item.get("image")
            if image is not None and hasattr(image, "save"):
                return image
            return None

        url = ""
        if item.get("type") == "image_url":
            url = (item.get("image_url") or {}).get("url", "")
        elif "image_url" in item:
            url = (item.get("image_url") or {}).get("url", "")

        if not isinstance(url, str) or not url:
            return None

        if url.startswith("file://"):
            return Image.open(Path(url.removeprefix("file://"))).convert("RGB")
        if url.startswith("data:image"):
            return Image.open(BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert(
                "RGB"
            )
        return None

    @classmethod
    def _prepare_messages_for_processor(
        cls, messages: list[Message]
    ) -> list[dict[str, Any]]:
        prepared: list[dict[str, Any]] = []
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                prepared.append(
                    {**message, "content": [{"type": "text", "text": content}]}
                )
                continue

            if not isinstance(content, list):
                prepared.append(dict(message))
                continue

            new_content: list[dict[str, Any]] = []
            for item in content:
                if isinstance(item, str):
                    new_content.append({"type": "text", "text": item})
                    continue
                if not isinstance(item, dict):
                    raise TypeError(f"Unexpected content item type: {type(item)}")

                if cls._is_multimodal_item(item):
                    if image := cls._load_image_from_item(item):
                        new_content.append({"type": "image", "image": image})
                    else:
                        new_content.append(dict(item))
                    continue

                if item.get("type") == "text" or "text" in item:
                    new_content.append({"type": "text", "text": item.get("text", "")})
                    continue

                new_content.append(dict(item))

            prepared.append({**message, "content": new_content})

        return prepared

    @staticmethod
    def _render_text_content(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(item["text"])
            return "".join(parts)
        raise TypeError(f"Unexpected content type: {type(content)}")

    @staticmethod
    def _render_multimodal_content(
        content: Any,
        *,
        image_count: int,
        video_count: int,
        add_vision_id: bool,
    ) -> tuple[str, int, int]:
        if content is None:
            return "", image_count, video_count
        if isinstance(content, str):
            return content, image_count, video_count
        if not isinstance(content, list):
            raise TypeError(f"Unexpected content type: {type(content)}")

        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                raise TypeError(f"Unexpected content item type: {type(item)}")

            if item.get("type") == "image" or "image" in item or "image_url" in item:
                image_count += 1
                if add_vision_id:
                    parts.append(f"Picture {image_count}: ")
                parts.append(_IMAGE_PLACEHOLDER)
            elif item.get("type") == "video" or "video" in item:
                video_count += 1
                if add_vision_id:
                    parts.append(f"Video {video_count}: ")
                parts.append(_VIDEO_PLACEHOLDER)
            elif "text" in item:
                parts.append(item["text"])

        return "".join(parts), image_count, video_count

    def render(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> RenderedTokens:
        if self._has_multimodal_content(messages):
            token_ids = self.render_ids(
                messages, tools=tools, add_generation_prompt=add_generation_prompt
            )
            return RenderedTokens(
                token_ids=token_ids,
                message_indices=[-1] * len(token_ids),
            )

        if not messages:
            raise ValueError("No messages provided.")

        tokens: list[int] = []
        indices: list[int] = []

        def emit_ids(ids: list[int], msg_idx: int) -> None:
            tokens.extend(ids)
            indices.extend([msg_idx] * len(ids))

        def emit_special(token_id: int, msg_idx: int) -> None:
            tokens.append(token_id)
            indices.append(msg_idx)

        def emit_text(text: str, msg_idx: int) -> None:
            emit_ids(self._encode(text), msg_idx)

        first_is_system = messages[0].get("role") == "system"

        if tools:
            sys_idx = 0 if first_is_system else -1
            emit_special(self._im_start, sys_idx)
            tool_text = "system\n"
            if first_is_system:
                sys_content = self._render_text_content(messages[0].get("content"))
                tool_text += sys_content + "\n\n"
            tool_text += _TOOLS_HEADER
            for tool in tools:
                tool_text += "\n" + json.dumps(tool, ensure_ascii=False)
            tool_text += _TOOLS_FOOTER
            emit_text(tool_text, sys_idx)
            emit_special(self._im_end, sys_idx)
            emit_text("\n", sys_idx)
        elif first_is_system:
            emit_special(self._im_start, 0)
            sys_content = self._render_text_content(messages[0].get("content"))
            emit_text("system\n" + sys_content, 0)
            emit_special(self._im_end, 0)
            emit_text("\n", 0)

        image_count = 0
        video_count = 0

        for i, msg in enumerate(messages):
            role = msg["role"]

            if role == "system":
                continue

            if role == "user":
                content, image_count, video_count = self._render_multimodal_content(
                    msg.get("content"),
                    image_count=image_count,
                    video_count=video_count,
                    add_vision_id=self._add_vision_id,
                )
                emit_special(self._im_start, i)
                emit_text("user\n" + content, i)
                emit_special(self._im_end, i)
                emit_text("\n", i)

            elif role == "assistant":
                self._render_assistant(
                    msg, i, emit_special=emit_special, emit_text=emit_text
                )

            elif role == "tool":
                content, image_count, video_count = self._render_multimodal_content(
                    msg.get("content"),
                    image_count=image_count,
                    video_count=video_count,
                    add_vision_id=self._add_vision_id,
                )
                self._render_tool(
                    messages,
                    i,
                    content,
                    emit_special=emit_special,
                    emit_text=emit_text,
                )

            else:
                raise ValueError(f"Unexpected message role: {role}")

        if add_generation_prompt:
            emit_special(self._im_start, -1)
            emit_text("assistant\n", -1)

        return RenderedTokens(token_ids=tokens, message_indices=indices)

    def render_ids(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        add_generation_prompt: bool = False,
    ) -> list[int]:
        if self._has_multimodal_content(messages):
            # Apply the chat template at the text level only. vLLM's
            # /v1/generate expects one <|image_pad|> placeholder per image and
            # performs the patch-grid expansion itself given multi_modal_data;
            # using tokenize=True here would emit already-expanded placeholders
            # and vLLM would double-expand, scrambling image embeddings on any
            # turn with more than one image.
            text = self._get_processor().apply_chat_template(
                self._prepare_messages_for_processor(messages),
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                **({} if tools is None else {"tools": tools}),
            )
            return self._tokenizer.encode(text, add_special_tokens=False)

        return self.render(
            messages, tools=tools, add_generation_prompt=add_generation_prompt
        ).token_ids

    def parse_response(self, token_ids: list[int]) -> ParsedResponse:
        return parse_qwen3(
            self._tokenizer,
            token_ids,
            stop_ids={self._im_end, self._endoftext},
            tool_call_id=self._tool_call,
            tool_call_end_id=self._tool_call_end,
        )

    def get_stop_token_ids(self) -> list[int]:
        return [self._im_end, self._endoftext]

    def bridge_to_next_turn(
        self,
        previous_prompt_ids: list[int],
        previous_completion_ids: list[int],
        new_messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
    ) -> list[int] | None:
        return chatml_bridge(
            self, previous_prompt_ids, previous_completion_ids, new_messages, tools=tools
        )

    def _render_assistant(
        self,
        msg: Message,
        msg_idx: int,
        *,
        emit_special,
        emit_text,
    ) -> None:
        content = self._render_text_content(msg.get("content"))
        original_content = msg.get("content")
        tool_calls = msg.get("tool_calls") or []

        emit_special(self._im_start, msg_idx)

        prefix = "assistant\n" + content
        if not tool_calls:
            emit_text(prefix, msg_idx)
        else:
            for tc_idx, tc in enumerate(tool_calls):
                if tc_idx == 0:
                    separator = "\n" if original_content else ""
                    emit_text(prefix + separator, msg_idx)
                else:
                    emit_text("\n", msg_idx)

                func = tc.get("function") or tc
                name = func.get("name", "")
                arguments = func.get("arguments", {})
                args_str = (
                    arguments
                    if isinstance(arguments, str)
                    else json.dumps(arguments, ensure_ascii=False)
                )

                emit_special(self._tool_call, msg_idx)
                emit_text(
                    '\n{"name": "' + name + '", "arguments": ' + args_str + "}\n",
                    msg_idx,
                )
                emit_special(self._tool_call_end, msg_idx)

        emit_special(self._im_end, msg_idx)
        emit_text("\n", msg_idx)

    def _render_tool(
        self,
        messages: list[Message],
        msg_idx: int,
        content: str,
        *,
        emit_special,
        emit_text,
    ) -> None:
        prev_is_tool = msg_idx > 0 and messages[msg_idx - 1]["role"] == "tool"
        next_is_tool = (
            msg_idx + 1 < len(messages) and messages[msg_idx + 1]["role"] == "tool"
        )

        if not prev_is_tool:
            emit_special(self._im_start, msg_idx)
            emit_text("user", msg_idx)

        emit_text("\n", msg_idx)
        emit_special(self._tool_response, msg_idx)
        emit_text("\n" + content + "\n", msg_idx)
        emit_special(self._tool_response_end, msg_idx)

        if not next_is_tool:
            emit_special(self._im_end, msg_idx)
            emit_text("\n", msg_idx)
