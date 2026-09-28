"""A native request and the task-facing projection bound to its editable fields."""

from collections.abc import Callable
from dataclasses import dataclass, field

from verifiers.v1.types import MessageContent, Request, ToolMessage, UserMessage


@dataclass
class ContentTarget:
    parent: dict
    key: str
    encode: Callable[[MessageContent], object]
    preserve_blocks: tuple[str, ...] = ()
    decode: Callable[[object], MessageContent] | None = None

    def replace(self, message: UserMessage | ToolMessage) -> UserMessage | ToolMessage:
        content = self.encode(message.content)
        projected = self.decode(content) if self.decode else message.content
        if self.preserve_blocks:
            # One Anthropic user message can contain both tool results and user content.
            # Replace only the latter, preserving the results and their native metadata.
            replacement = (
                [{"type": "text", "text": content}]
                if isinstance(content, str)
                else content
            )
            content = []
            inserted = False
            for block in self.parent[self.key]:
                if block.get("type") in self.preserve_blocks:
                    content.append(block)
                elif not inserted:
                    content.extend(replacement)
                    inserted = True
        self.parent[self.key] = content
        return message.model_copy(update={"content": projected})


@dataclass
class NativeRequest:
    """Native JSON remains authoritative; only explicitly bound content can be edited.

    Targets are indexed while projecting, so applying a hook never needs to reconstruct
    how native blocks were split or folded into task messages. Provider declarations are
    separate from the executable tool signatures exposed to tasks.
    """

    body: dict
    view: Request
    targets: dict[int, ContentTarget] = field(default_factory=dict)
    provider_tools: list[dict] = field(default_factory=list)

    def replace(self, after: Request) -> None:
        before = self.view
        if len(before.messages) != len(after.messages) or before.tools != after.tools:
            raise ValueError("request edits must preserve message positions and tools")
        edits = []
        for index, (old, new) in enumerate(zip(before.messages, after.messages)):
            if old == new:
                continue
            target = self.targets.get(index)
            if target is None or type(new) is not type(old):
                raise ValueError("request edits can only replace user or tool content")
            if isinstance(new, ToolMessage) and (
                new.tool_call_id != old.tool_call_id or new.name != old.name
            ):
                raise ValueError("request edits cannot change tool call IDs or names")
            edits.append((index, target, new))
        messages = list(after.messages)
        for index, target, message in edits:
            messages[index] = target.replace(message)
        self.view = after.model_copy(update={"messages": messages})


def provider_declaration(tool: dict) -> dict:
    """A trace-safe copy; connection credentials stay on the native exchange only."""
    declaration = {
        key: value
        for key, value in tool.items()
        if key.lower() not in ("authorization", "headers")
    }
    if tool.get("type") == "namespace":
        declaration["tools"] = [
            provider_declaration(t) for t in tool.get("tools") or []
        ]
    return declaration
