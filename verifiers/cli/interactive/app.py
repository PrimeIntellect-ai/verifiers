import asyncio
import base64
import hashlib
import io
import json
import mimetypes
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote, unquote_to_bytes, urlparse

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message as TextualMessage
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, Select, Static, TextArea

from verifiers.errors import Error
from verifiers.types import (
    AssistantMessage,
    Message,
    MessageContent,
    Messages,
    Tool,
    ToolCall,
)


class InteractiveSessionExit(Error):
    """Raised when the human operator quits an interactive rollout.

    A ``vf.Error`` so both rollout paths finish cleanly on quit: direct
    clients record it via the rollout loop's error handling, and
    intercepted endpoints record it on the rollout state instead of
    crashing the request handler.
    """


class ArgTextArea(TextArea):
    """Multi-line tool-argument cell: Enter adds the call, Shift+Enter newlines.

    Ctrl+J also inserts a newline for terminals that report Shift+Enter as
    plain Enter.
    """

    class AddRequested(TextualMessage):
        pass

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.AddRequested())
            return
        if event.key in ("shift+enter", "ctrl+j"):
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        await super()._on_key(event)


class AnswerScreen(ModalScreen[None]):
    """Modal viewer for the rollout's answer field."""

    CSS = """
    AnswerScreen {
        align: center middle;
    }

    AnswerScreen > VerticalScroll {
        width: 80%;
        max-height: 80%;
        height: auto;
        background: $surface;
    }
    """

    BINDINGS = [
        ("escape", "close", "Close"),
        ("a", "close", "Close"),
        ("q", "close", "Close"),
    ]

    def __init__(self, answer: str) -> None:
        super().__init__()
        self._answer = answer

    def compose(self) -> ComposeResult:
        with VerticalScroll():
            yield Static(
                Panel(
                    Text(self._answer),
                    title="Rollout answer",
                    border_style="yellow",
                )
            )

    def action_close(self) -> None:
        self.dismiss()


@dataclass(frozen=True)
class TurnResponse:
    content: MessageContent | None
    tool_calls: list[ToolCall] | None
    reasoning: str | None = None


@dataclass(frozen=True)
class ImagePayload:
    path: Path
    size: tuple[int, int]
    source: str


class InteractiveRolloutApp(App[None]):
    """Persistent TUI for human-driven model turns."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #main {
        height: 1fr;
        min-height: 0;
    }

    #messages-pane {
        width: 2fr;
        height: 1fr;
        border: solid $primary;
        padding: 1;
    }

    #tools-pane {
        width: 1fr;
        height: 1fr;
        border: solid $success;
        padding: 1;
    }

    #messages-content {
        height: auto;
    }

    #tools-content {
        height: auto;
    }

    #response-pane {
        height: auto;
        border: solid $accent;
        padding: 0 1;
    }

    #reasoning, #message {
        height: 3;
    }

    .field-row {
        height: auto;
        margin-top: 1;
    }

    #arguments .field-row {
        margin-top: 0;
    }

    .field-label {
        width: 12;
        margin-right: 1;
        text-align: right;
        color: $text-muted;
    }

    .field-row TextArea, .field-row Input, .field-row Select {
        width: 1fr;
    }

    #arguments {
        height: auto;
    }

    #arguments TextArea {
        height: auto;
        min-height: 3;
        max-height: 9;
    }

    #status {
        height: auto;
        color: $warning;
    }

    #turn-preview {
        height: auto;
        margin-top: 1;
    }

    #actions {
        height: auto;
        margin-top: 1;
        margin-bottom: 1;
    }

    #actions Button {
        height: 1;
        border: none;
        margin-right: 2;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+c", "quit", "Quit"),
        ("a", "show_answer", "Answer"),
    ]

    def __init__(
        self,
        *,
        show_prompt: bool = True,
        show_tools: bool = True,
        max_content_chars: int = 20000,
        answer: str | None = None,
        allow_remote_images: bool = False,
    ) -> None:
        super().__init__()
        self.ready = asyncio.Event()
        self._answer = answer
        self._allow_remote_images = allow_remote_images
        self._quit_requested = False
        self._future: asyncio.Future[TurnResponse] | None = None
        self._turn = 0
        self._tools: list[Tool] = []
        self._pending_tool_calls: list[ToolCall] = []
        self._arg_inputs: dict[
            str, tuple[Input | TextArea, Mapping[str, Any], bool]
        ] = {}
        self._image_cache: dict[str, RenderableType] = {}
        self._image_payload_cache: dict[str, ImagePayload] = {}
        self._show_prompt = show_prompt
        self._show_tools = show_tools
        self._max_content_chars = max_content_chars
        self._terminal_image_type = self._detect_terminal_image_type()

    @staticmethod
    def _detect_terminal_image_type() -> type | None:
        """Resolve the terminal image renderable while the terminal is still free.

        textual_image picks its protocol by querying the terminal once at
        import time. Importing lazily during message rendering loses the
        query reply to the running app's raw-mode input reader, silently
        downgrading to the lossy text fallbacks, so the import must happen
        before ``run_async``.
        """
        try:
            from textual_image.renderable import Image as TerminalImage
        except ModuleNotFoundError:
            return None
        module = TerminalImage.__module__
        if module.endswith(".halfcell") or module.endswith(".unicode"):
            return None
        return TerminalImage

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main"):
            with VerticalScroll(id="messages-pane"):
                yield Static(id="messages-content")
            with VerticalScroll(id="tools-pane"):
                yield Static(id="tools-content")
        with Vertical(id="response-pane"):
            with Horizontal(classes="field-row"):
                yield Static("Reasoning", classes="field-label")
                yield TextArea(id="reasoning", soft_wrap=True)
            with Horizontal(classes="field-row"):
                yield Static("Message", classes="field-label")
                yield TextArea(id="message", soft_wrap=True)
            with Horizontal(classes="field-row", id="tool-picker-row"):
                yield Static("Tool calls", classes="field-label")
                yield Select([], id="tool-picker", prompt="Add a tool call...")
            yield Vertical(id="arguments")
            yield Static("", id="turn-preview")
            yield Static("", id="status")
            with Horizontal(id="actions"):
                yield Button("Add tool call", id="add-tool-call")
                yield Button("Clear tool calls", id="clear-tool-calls")
                yield Button("Submit turn", id="submit", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "Verifiers interactive rollout"
        self.sub_title = "Environment starting"
        self.query_one("#messages-pane", VerticalScroll).loading = True
        self._set_status(
            "Environment is setting up; the first turn will appear here. "
            "This can take a while for sandboxed environments."
        )
        self.ready.set()

    def action_quit(self) -> None:
        self._quit_requested = True
        if self._future is not None and not self._future.done():
            self._future.set_exception(InteractiveSessionExit())
            self._future = None
        self.exit()

    def action_show_answer(self) -> None:
        if self._answer is None:
            self._set_status("This rollout has no answer field.")
            return
        self.push_screen(AnswerScreen(self._answer))

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "show_answer" and self._answer is None:
            return None
        return True

    async def ask(self, messages: Messages, tools: list[Tool]) -> TurnResponse:
        if self._quit_requested:
            raise InteractiveSessionExit()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[TurnResponse] = loop.create_future()
        # Assign synchronously so a quit that lands before ``load_turn`` runs
        # (it is deferred via ``call_later``) still fails this exact future.
        self._future = future

        def load_turn() -> None:
            if self._quit_requested or future.done():
                return
            self._turn += 1
            self._tools = list(tools)
            self._pending_tool_calls.clear()
            self._image_cache.clear()
            self._image_payload_cache.clear()
            self.query_one("#messages-pane", VerticalScroll).loading = False
            self.sub_title = f"Turn {self._turn}"
            self._arg_inputs.clear()
            self.query_one("#reasoning", TextArea).load_text("")
            self.query_one("#message", TextArea).load_text("")
            if self._show_prompt:
                self._render_messages(messages)
            else:
                self.query_one("#messages-content", Static).update(
                    Panel("Prompt display disabled.", title="Messages")
                )
            if self._show_tools:
                self._render_tools(self._tools)
            else:
                self.query_one("#tools-content", Static).update(
                    Panel("Tool display disabled.", title="Tools")
                )
            self._configure_tool_picker()
            self._render_turn_preview()

        self.call_later(load_turn)
        return await future

    def _configure_tool_picker(self) -> None:
        picker = self.query_one("#tool-picker", Select)
        picker.set_options((Text(tool.name), tool.name) for tool in self._tools)
        picker.value = Select.BLANK
        self.query_one("#tool-picker-row", Horizontal).display = bool(self._tools)
        self._update_tool_form(None)

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "tool-picker":
            return
        value = event.value
        self._update_tool_form(None if value == Select.BLANK else str(value))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "add-tool-call":
            self._add_current_tool_call()
        elif event.button.id == "clear-tool-calls":
            self._pending_tool_calls.clear()
            self._render_turn_preview()
            self._set_status("Tool calls cleared.")
        elif event.button.id == "submit":
            self._submit_current_turn()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        _ = event
        self._add_current_tool_call()

    def on_arg_text_area_add_requested(self, event: ArgTextArea.AddRequested) -> None:
        _ = event
        self._add_current_tool_call()

    def on_input_changed(self, event: Input.Changed) -> None:
        _ = event
        self._render_turn_preview()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        _ = event
        self._render_turn_preview()

    def _submit_current_turn(self) -> None:
        future = self._future
        if future is None or future.done():
            return
        try:
            response = self._build_response()
        except ValueError as exc:
            self._set_status(str(exc))
            return
        future.set_result(response)
        self._future = None
        self._set_status("Turn submitted. Waiting for environment response.")

    def _build_response(self) -> TurnResponse:
        message = self.query_one("#message", TextArea).text
        reasoning = self.query_one("#reasoning", TextArea).text
        content = message if message.strip() else None
        tool_calls = list(self._pending_tool_calls)
        if self._current_form_included():
            tool_calls.append(self._build_current_tool_call())
        if content is None and not tool_calls:
            raise ValueError("Type a message or add at least one tool call.")
        return TurnResponse(
            content=content,
            tool_calls=tool_calls or None,
            reasoning=reasoning if reasoning.strip() else None,
        )

    def _selected_tool_name(self) -> str | None:
        value = self.query_one("#tool-picker", Select).value
        return None if value == Select.BLANK else str(value)

    def _current_form_included(self) -> bool:
        """Whether the tool form currently being edited joins the submitted turn.

        Only a form the operator actually filled in is auto-included on submit,
        so selecting a tool to inspect its schema never blocks a message-only
        turn. Argument-less calls are added explicitly via ``Add tool call``.
        """
        if self._selected_tool_name() is None:
            return False
        return self._current_tool_form_has_values()

    def _add_current_tool_call(self) -> None:
        try:
            self._pending_tool_calls.append(self._build_current_tool_call())
        except ValueError as exc:
            self._set_status(str(exc))
            return
        self._clear_argument_inputs()
        self._render_turn_preview()
        self._set_status("Tool call added. Add another or submit the turn.")

    def _build_current_tool_call(self) -> ToolCall:
        value = self._selected_tool_name()
        if value is None:
            raise ValueError("Select a tool before adding a tool call.")
        tool = self._tool_by_name(value)
        if tool is None:
            raise ValueError(f"Unknown selected tool: {value!r}")
        arguments: dict[str, Any] = {}
        for name, (input_widget, schema, required) in self._arg_inputs.items():
            raw = self._arg_raw_value(input_widget)
            if not raw.strip():
                if required:
                    raise ValueError(f"Missing required argument: {name}")
                continue
            arguments[name] = self.parse_human_value(raw, schema)
        return ToolCall(
            id=f"call_{uuid.uuid4().hex[:12]}",
            name=tool.name,
            arguments=json.dumps(
                arguments,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )

    def _update_tool_form(self, tool_name: str | None) -> None:
        container = self.query_one("#arguments", Vertical)
        self._arg_inputs.clear()
        container.remove_children()
        container.display = tool_name is not None
        self.query_one("#add-tool-call", Button).display = tool_name is not None
        self._render_turn_preview()
        if tool_name is None:
            self._set_status("Compose the turn: message, reasoning, and/or tool calls.")
            return

        tool = self._tool_by_name(tool_name)
        if tool is None:
            self._set_status(f"Unknown tool: {tool_name}")
            return
        rows = self._argument_rows(tool)
        if not rows:
            container.mount(Static("No declared arguments."))
        else:
            for name, schema, required in rows:
                label = Static(
                    name if required else f"{name}?",
                    classes="field-label",
                )
                placeholder = self.argument_placeholder(schema, required=required)
                arg_type = self.schema_type(schema)
                is_multiline = (
                    arg_type == "string" or arg_type is None
                ) and not schema.get("enum")
                input_widget: Input | TextArea = (
                    ArgTextArea(placeholder=placeholder, classes="arg-input")
                    if is_multiline
                    else Input(placeholder=placeholder, classes="arg-input")
                )
                container.mount(Horizontal(label, input_widget, classes="field-row"))
                self._arg_inputs[name] = (input_widget, schema, required)
        self._set_status(
            f"Fill arguments for {tool.name}. "
            "Enter adds the call; Shift+Enter (or Ctrl+J) inserts a newline."
        )

    @staticmethod
    def _arg_raw_value(widget: Input | TextArea) -> str:
        # Return the value verbatim: multiline string arguments (code, YAML)
        # carry meaningful leading/trailing whitespace and newlines that must
        # survive to ``parse_human_value``. Callers strip only for empty checks.
        return widget.value if isinstance(widget, Input) else widget.text

    def _current_tool_form_has_values(self) -> bool:
        return any(
            self._arg_raw_value(input_widget).strip()
            for input_widget, _, _ in self._arg_inputs.values()
        )

    def _clear_argument_inputs(self) -> None:
        for input_widget, _, _ in self._arg_inputs.values():
            if isinstance(input_widget, Input):
                input_widget.value = ""
            else:
                input_widget.load_text("")

    def _render_turn_preview(self) -> None:
        self.query_one("#clear-tool-calls", Button).display = bool(
            self._pending_tool_calls
        )
        self.query_one("#turn-preview", Static).update(
            Panel(
                self._turn_preview_renderable(),
                title="Turn to submit",
                border_style="yellow",
            )
        )

    def _turn_preview_renderable(self) -> RenderableType:
        """Show exactly what Submit will send as the assistant turn."""
        reasoning = self.query_one("#reasoning", TextArea).text.strip()
        message = self.query_one("#message", TextArea).text.strip()
        items: list[RenderableType] = []
        if reasoning:
            items.append(
                Text(f"reasoning: {self._preview_text(reasoning)}", style="dim italic")
            )
        if message:
            items.append(Text(f"message: {self._preview_text(message)}"))
        calls = [(call.name, call.arguments) for call in self._pending_tool_calls]
        if self._current_form_included():
            form_args: dict[str, Any] = {}
            for name, (input_widget, schema, _) in self._arg_inputs.items():
                raw = self._arg_raw_value(input_widget)
                if not raw:
                    continue
                try:
                    form_args[name] = self.parse_human_value(raw, schema)
                except ValueError:
                    form_args[name] = raw
            calls.append(
                (
                    f"{self._selected_tool_name()} (current form)",
                    json.dumps(form_args, ensure_ascii=False, separators=(",", ":")),
                )
            )
        for name, arguments in calls:
            line = Text("tool call: ", style="bold")
            line.append(name)
            line.append(f" {self._preview_text(arguments)}", style="dim")
            items.append(line)
        if not items:
            return Text("(empty — type a message or add a tool call)", style="dim")
        return Group(*items)

    def _preview_text(self, value: str) -> str:
        collapsed = " ".join(value.split())
        if len(collapsed) <= 200:
            return collapsed
        return collapsed[:200] + "..."

    def _render_messages(self, messages: Messages) -> None:
        self.query_one("#messages-content", Static).update(
            Panel(
                self._messages_renderable(messages),
                title="Model-visible messages",
                border_style="cyan",
            )
        )
        self.set_timer(0.01, self._scroll_messages_to_end)

    def _render_tools(self, tools: list[Tool]) -> None:
        self.query_one("#tools-content", Static).update(
            Panel(self._tools_renderable(tools), title="Tools", border_style="green")
        )
        self.call_after_refresh(self._scroll_tools_to_home)

    def _scroll_messages_to_end(self) -> None:
        self.query_one("#messages-pane", VerticalScroll).scroll_end(
            animate=False,
            force=True,
            immediate=True,
        )

    def _scroll_tools_to_home(self) -> None:
        self.query_one("#tools-pane", VerticalScroll).scroll_home(
            animate=False,
            force=True,
            immediate=True,
        )

    def _messages_renderable(self, messages: Messages) -> RenderableType:
        panels: list[RenderableType] = []
        for index, message in enumerate(messages, start=1):
            panels.append(self._message_panel(index, message))
        return Group(*panels) if panels else Text("No messages.")

    def _message_panel(self, index: int, message: Message) -> Panel:
        role = getattr(message, "role", "message")
        items = self._content_renderables(getattr(message, "content", None))
        if isinstance(message, AssistantMessage) and message.reasoning_content:
            if items:
                items.insert(0, Text(""))
            items.insert(
                0,
                Text(
                    f"reasoning: {self._truncate_text(message.reasoning_content)}",
                    style="dim italic",
                ),
            )
        if isinstance(message, AssistantMessage) and message.tool_calls:
            items.append(Text(""))
            items.append(Text(self._tool_calls_to_text(message.tool_calls)))
        tool_call_id = getattr(message, "tool_call_id", None)
        if isinstance(tool_call_id, str):
            items.insert(0, Text(f"tool_call_id: {tool_call_id}"))
            items.insert(1, Text(""))
        return Panel(
            Group(*items) if items else Text("(empty)"),
            title=f"{index}. {role}",
        )

    def _tools_renderable(self, tools: list[Tool]) -> RenderableType:
        if not tools:
            return Text("No tools are available this turn.")
        items: list[RenderableType] = []
        for index, tool in enumerate(tools, start=1):
            table = Table(title=f"{index}. {tool.name}", show_header=True)
            table.add_column("Argument", style="bold")
            table.add_column("Type")
            table.add_column("Required")
            table.add_column("Description")
            for name, schema, required in self._argument_rows(tool):
                description = schema.get("description")
                table.add_row(
                    name,
                    self.schema_type(schema) or "",
                    "yes" if required else "no",
                    description if isinstance(description, str) else "",
                )
            if not self._argument_rows(tool):
                table.add_row("(none declared)", "", "", "")
            description = Text(tool.description or "", style="dim")
            items.append(Group(description, table))
        return Group(*items)

    def _argument_rows(self, tool: Tool) -> list[tuple[str, Mapping[str, Any], bool]]:
        properties = tool.parameters.get("properties")
        if not isinstance(properties, Mapping):
            return []
        required = tool.parameters.get("required")
        required_names = (
            {item for item in required if isinstance(item, str)}
            if isinstance(required, list)
            else set()
        )
        rows: list[tuple[str, Mapping[str, Any], bool]] = []
        for name, raw_schema in properties.items():
            if not isinstance(name, str):
                continue
            schema = (
                cast(Mapping[str, Any], raw_schema)
                if isinstance(raw_schema, Mapping)
                else {}
            )
            rows.append((name, schema, name in required_names))
        return rows

    def _tool_by_name(self, name: str) -> Tool | None:
        for tool in self._tools:
            if tool.name == name:
                return tool
        return None

    def _set_status(self, message: str) -> None:
        self.query_one("#status", Static).update(message)

    def _content_renderables(
        self, content: MessageContent | None
    ) -> list[RenderableType]:
        if content is None:
            return []
        if isinstance(content, str):
            return [Text(self._truncate_text(content))]
        parts: list[RenderableType] = []
        for part in content:
            part_data = self._object_to_mapping(part)
            part_type = part_data.get("type")
            if part_type == "text":
                parts.append(Text(self._truncate_text(str(part_data.get("text", "")))))
            elif part_type in {"image_url", "input_image"}:
                parts.append(self._image_part_renderable(part_data))
            elif part_type == "input_audio":
                parts.append(Text("[audio]", style="magenta"))
            else:
                parts.append(
                    Text(
                        self._truncate_text(
                            json.dumps(part_data, indent=2, ensure_ascii=False)
                        )
                    )
                )
        return parts

    def _image_part_renderable(self, part_data: Mapping[str, Any]) -> RenderableType:
        url = self._image_url_from_part(part_data)
        if not isinstance(url, str):
            return Text("[image_url]")
        cached = self._image_cache.get(url)
        if cached is not None:
            return cached
        is_remote = url.startswith("http://") or url.startswith("https://")
        if is_remote and not self._allow_remote_images:
            # Do not issue outbound requests for environment-supplied URLs by
            # default: it is an SSRF vector (localhost/private ranges) and the
            # blocking fetch would freeze the event loop. Opt in with
            # ``--allow-remote-images``.
            renderable: RenderableType = Panel(
                Text(
                    f"Remote image not fetched (pass --allow-remote-images):"
                    f"\n{self._truncate_text(url)}",
                    style="yellow",
                ),
                title="image_url",
                border_style="magenta",
            )
            self._image_cache[url] = renderable
            return renderable
        try:
            payload = self.load_image_payload(url)
            image_renderable = self._terminal_image_renderable(payload)
            if image_renderable is None:
                renderable = self._exact_image_panel(payload)
            else:
                renderable = Panel(
                    image_renderable,
                    title=self._image_title(payload),
                    subtitle=str(payload.path),
                    border_style="magenta",
                )
        except Exception as exc:
            renderable = Text(f"[image_url: {url}] ({exc})", style="red")
        self._image_cache[url] = renderable
        return renderable

    def _image_url_from_part(self, part_data: Mapping[str, Any]) -> str | None:
        image_url = part_data.get("image_url")
        if isinstance(image_url, Mapping):
            url = image_url.get("url")
            return url if isinstance(url, str) else None
        if isinstance(image_url, str):
            return image_url
        url = part_data.get("url")
        return url if isinstance(url, str) else None

    def _truncate_text(self, value: str) -> str:
        if len(value) <= self._max_content_chars:
            return value
        return value[: self._max_content_chars] + "\n...[truncated]"

    def _image_title(self, payload: ImagePayload) -> str:
        return f"image_url ({payload.size[0]}x{payload.size[1]})"

    def _terminal_image_renderable(
        self, payload: ImagePayload
    ) -> RenderableType | None:
        if self._terminal_image_type is None:
            return None
        return self._terminal_image_type(payload.path)

    def _exact_image_panel(self, payload: ImagePayload) -> Panel:
        body = Text()
        body.append("Terminal image protocol unavailable.\n", style="yellow")
        body.append("Exact image bytes saved at:\n")
        body.append(str(payload.path), style="bold")
        body.append("\n\n")
        body.append(f"source: {self._truncate_text(payload.source)}")
        return Panel(
            body,
            title=self._image_title(payload),
            border_style="magenta",
        )

    def load_image_payload(self, url: str) -> ImagePayload:
        cached = self._image_payload_cache.get(url)
        if cached is not None:
            return cached
        payload = self.load_image_url(url)
        self._image_payload_cache[url] = payload
        return payload

    @staticmethod
    def load_image_url(url: str) -> ImagePayload:
        from PIL import Image

        if url.startswith("data:"):
            header, separator, encoded_payload = url.partition(",")
            if not separator:
                raise ValueError("Malformed data URL")
            if ";base64" in header:
                data = base64.b64decode(encoded_payload)
            else:
                data = unquote_to_bytes(encoded_payload)
            suffix = InteractiveRolloutApp._suffix_for_data_url_header(header)
            path = InteractiveRolloutApp._write_image_bytes(data, suffix=suffix)
            size = Image.open(io.BytesIO(data)).size
            return ImagePayload(path=path, size=size, source=header)
        if url.startswith("http://") or url.startswith("https://"):
            import requests

            response = requests.get(url, timeout=10)
            response.raise_for_status()
            suffix = InteractiveRolloutApp._suffix_for_url(
                url,
                response.headers.get("content-type"),
            )
            path = InteractiveRolloutApp._write_image_bytes(
                response.content,
                suffix=suffix,
            )
            size = Image.open(io.BytesIO(response.content)).size
            return ImagePayload(path=path, size=size, source=url)
        if url.startswith("file://"):
            parsed = urlparse(url)
            path = Path(unquote(parsed.path))
            size = Image.open(path).size
            return ImagePayload(path=path, size=size, source=url)
        path = Path(url)
        size = Image.open(path).size
        return ImagePayload(path=path, size=size, source=url)

    @staticmethod
    def _write_image_bytes(data: bytes, *, suffix: str) -> Path:
        digest = hashlib.sha256(data).hexdigest()
        directory = Path(tempfile.gettempdir()) / "verifiers-interactive-images"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{digest}{suffix}"
        if not path.exists():
            path.write_bytes(data)
        return path

    @staticmethod
    def _suffix_for_data_url_header(header: str) -> str:
        media_type = header.partition(":")[2].partition(";")[0]
        return mimetypes.guess_extension(media_type) or ".img"

    @staticmethod
    def _suffix_for_url(url: str, content_type: str | None) -> str:
        if content_type:
            media_type = content_type.partition(";")[0].strip()
            suffix = mimetypes.guess_extension(media_type)
            if suffix:
                return suffix
        suffix = Path(urlparse(url).path).suffix
        return suffix or ".img"

    def _tool_calls_to_text(self, tool_calls: list[ToolCall]) -> str:
        def render(tool_call: ToolCall) -> str:
            # ``arguments`` is a JSON string; decode it so the pane shows the
            # actual argument object instead of a doubly-escaped string.
            try:
                arguments: Any = json.loads(tool_call.arguments)
            except json.JSONDecodeError:
                arguments = tool_call.arguments
            return json.dumps(
                {
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "arguments": arguments,
                },
                indent=2,
                ensure_ascii=False,
            )

        return "tool_calls:\n" + "\n".join(render(tc) for tc in tool_calls)

    def _object_to_mapping(self, value: object) -> Mapping[str, Any]:
        if isinstance(value, Mapping):
            return cast(Mapping[str, Any], value)
        model_dump = getattr(value, "model_dump", None)
        if callable(model_dump):
            return cast(Mapping[str, Any], model_dump(exclude_none=True))
        return {"value": value}

    @staticmethod
    def argument_placeholder(schema: Mapping[str, Any], *, required: bool) -> str:
        parts: list[str] = []
        arg_type = InteractiveRolloutApp.schema_type(schema)
        if arg_type is not None:
            parts.append(arg_type)
        description = schema.get("description")
        if isinstance(description, str) and description:
            parts.append(description)
        enum = schema.get("enum")
        if isinstance(enum, list) and enum:
            choices = ", ".join(f"{idx + 1}:{value}" for idx, value in enumerate(enum))
            parts.append(f"[{choices}]")
        if not required:
            parts.append("(optional — blank omits)")
        return " — ".join(parts)

    @staticmethod
    def parse_human_value(raw: str, schema: Mapping[str, Any]) -> Any:
        enum = schema.get("enum")
        if isinstance(enum, list) and enum:
            token = raw.strip()
            # Prefer an exact typed match so numeric/boolean enums keep their
            # type (e.g. entering "10" for [10, 20] returns int 10, not "10").
            try:
                parsed = json.loads(token)
            except json.JSONDecodeError:
                parsed = token
            if parsed in enum:
                return parsed
            if token in enum:
                return token
            # Fall back to 1-based index selection for string-labelled enums.
            if token.isdigit() and 1 <= int(token) <= len(enum):
                return enum[int(token) - 1]
            raise ValueError(f"Expected one of: {', '.join(map(str, enum))}.")

        arg_type = InteractiveRolloutApp.schema_type(schema)
        if arg_type == "string" or arg_type is None:
            return raw
        if arg_type == "integer":
            try:
                return int(raw)
            except ValueError as exc:
                raise ValueError("Expected an integer.") from exc
        if arg_type == "number":
            try:
                return float(raw)
            except ValueError as exc:
                raise ValueError("Expected a number.") from exc
        if arg_type == "boolean":
            lowered = raw.strip().lower()
            if lowered in {"true", "t", "yes", "y", "1"}:
                return True
            if lowered in {"false", "f", "no", "n", "0"}:
                return False
            raise ValueError("Expected yes/no or true/false.")
        if arg_type in {"object", "array"}:
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Expected a JSON {arg_type} value.") from exc
            if arg_type == "object" and not isinstance(parsed, dict):
                raise ValueError("Expected a JSON object.")
            if arg_type == "array" and not isinstance(parsed, list):
                raise ValueError("Expected a JSON array.")
            return parsed
        if arg_type == "null":
            if raw.strip().lower() == "null":
                return None
            raise ValueError("Expected null.")
        return raw

    @staticmethod
    def schema_type(schema: Mapping[str, Any]) -> str | None:
        raw_type = schema.get("type")
        if isinstance(raw_type, str):
            return raw_type
        if isinstance(raw_type, list):
            for item in raw_type:
                if isinstance(item, str) and item != "null":
                    return item
        return None
