import asyncio
import base64
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.panel import Panel
from textual.containers import VerticalScroll
from textual.widgets import Input, Select, TextArea

from verifiers.cli.interactive.app import (
    InteractiveRolloutApp,
    InteractiveSessionExit,
    TurnResponse,
)
from verifiers.cli.interactive.client import HumanClient
from verifiers.scripts import play
from verifiers.types import Tool, ToolCall, UserMessage


def lookup_tool() -> Tool:
    return Tool(
        name="lookup_item",
        description="Look up an item by numeric id.",
        parameters={
            "type": "object",
            "properties": {"item_id": {"type": "integer"}},
            "required": ["item_id"],
        },
    )


def label_tool() -> Tool:
    return Tool(
        name="label_item",
        description="Attach a label to the current item.",
        parameters={
            "type": "object",
            "properties": {"label": {"type": "string"}},
            "required": ["label"],
        },
    )


def tiny_png_data_url() -> str:
    from PIL import Image

    buffer = io.BytesIO()
    image = Image.new("RGB", (2, 2), (255, 0, 0))
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def test_app_parses_argument_values_from_json_schema_types():
    assert InteractiveRolloutApp.parse_human_value("4", {"type": "integer"}) == 4
    assert InteractiveRolloutApp.parse_human_value("yes", {"type": "boolean"}) is True
    assert (
        InteractiveRolloutApp.parse_human_value("2", {"enum": ["red", "blue"]})
        == "blue"
    )
    assert InteractiveRolloutApp.parse_human_value('["a"]', {"type": "array"}) == ["a"]


def test_app_rejects_wrong_argument_type():
    with pytest.raises(ValueError, match="Expected an integer"):
        InteractiveRolloutApp.parse_human_value("four", {"type": "integer"})


def test_app_parses_typed_enums_by_exact_value():
    parse = InteractiveRolloutApp.parse_human_value
    # Numeric/boolean enum members keep their type when entered literally.
    result = parse("10", {"type": "integer", "enum": [10, 20]})
    assert result == 10 and isinstance(result, int)
    assert parse("false", {"type": "boolean", "enum": [True, False]}) is False
    # 1-based index selection still works for string-labelled enums.
    assert parse("2", {"enum": ["red", "blue"]}) == "blue"
    # An exact string member wins over index interpretation.
    assert parse("2", {"enum": ["2", "1"]}) == "2"


def test_app_preserves_string_argument_whitespace():
    schema = {"type": "string"}
    assert InteractiveRolloutApp.parse_human_value("  a\n  b\n", schema) == "  a\n  b\n"


@pytest.mark.asyncio
async def test_app_required_whitespace_only_string_is_sent():
    tool = Tool(
        name="write",
        description="write content",
        parameters={
            "type": "object",
            "properties": {"content": {"type": "string"}},
            "required": ["content"],
        },
    )
    app = InteractiveRolloutApp()
    async with app.run_test() as pilot:
        ask_task = asyncio.create_task(app.ask([UserMessage(content="go")], [tool]))
        await pilot.pause()
        app.query_one("#tool-picker", Select).value = "write"
        await pilot.pause()
        app.query_one(".arg-input", TextArea).load_text("   ")
        app._submit_current_turn()
        response = await ask_task

    assert response.tool_calls is not None
    assert json.loads(response.tool_calls[0].arguments) == {"content": "   "}


def test_tool_calls_text_shows_decoded_arguments():
    app = InteractiveRolloutApp()
    text = app._tool_calls_to_text(
        [ToolCall(id="c1", name="lookup_item", arguments='{"item_id":7}')]
    )
    assert '"item_id": 7' in text
    assert '\\"' not in text


def test_app_exposes_tool_argument_rows_from_schema():
    app = InteractiveRolloutApp()

    rows = app._argument_rows(lookup_tool())

    assert rows == [("item_id", {"type": "integer"}, True)]


@pytest.mark.asyncio
async def test_app_uses_scrollable_prompt_and_tool_panes():
    app = InteractiveRolloutApp()
    task = asyncio.create_task(app.run_async(headless=True))
    await app.ready.wait()

    assert isinstance(app.query_one("#messages-pane"), VerticalScroll)
    assert isinstance(app.query_one("#tools-pane"), VerticalScroll)

    app.exit()
    await task


@pytest.mark.asyncio
async def test_app_scrolls_prompt_pane_to_latest_turn():
    app = InteractiveRolloutApp()
    task = asyncio.create_task(app.run_async(headless=True))
    await app.ready.wait()
    messages = [
        UserMessage(content=f"turn {index}\n" + ("detail\n" * 4)) for index in range(20)
    ]
    ask_task = asyncio.create_task(app.ask(messages, []))
    await asyncio.sleep(0.2)

    pane = app.query_one("#messages-pane", VerticalScroll)
    assert pane.max_scroll_y > 0
    assert pane.scroll_y == pane.max_scroll_y

    app.query_one("#message", TextArea).load_text("done")
    app._submit_current_turn()
    await ask_task
    app.exit()
    await task


@pytest.mark.asyncio
async def test_app_headless_message_submit():
    app = InteractiveRolloutApp()
    task = asyncio.create_task(app.run_async(headless=True))
    await app.ready.wait()
    ask_task = asyncio.create_task(app.ask([UserMessage(content="hello")], []))
    await asyncio.sleep(0.1)

    app.query_one("#message", TextArea).load_text("final answer")
    app._submit_current_turn()
    response = await ask_task
    app.exit()
    await task

    assert response.content == "final answer"
    assert response.tool_calls is None
    assert response.reasoning is None


@pytest.mark.asyncio
async def test_app_headless_reasoning_submit():
    app = InteractiveRolloutApp()
    task = asyncio.create_task(app.run_async(headless=True))
    await app.ready.wait()
    ask_task = asyncio.create_task(app.ask([UserMessage(content="hello")], []))
    await asyncio.sleep(0.1)

    app.query_one("#reasoning", TextArea).load_text("the answer is obvious")
    app.query_one("#message", TextArea).load_text("final answer")
    app._submit_current_turn()
    response = await ask_task

    next_ask_task = asyncio.create_task(app.ask([UserMessage(content="again")], []))
    await asyncio.sleep(0.1)
    assert app.query_one("#reasoning", TextArea).text == ""
    assert app.query_one("#message", TextArea).text == ""
    app.query_one("#message", TextArea).load_text("second answer")
    app._submit_current_turn()
    next_response = await next_ask_task
    app.exit()
    await task

    assert response.content == "final answer"
    assert response.reasoning == "the answer is obvious"
    assert next_response.content == "second answer"
    assert next_response.reasoning is None


@pytest.mark.asyncio
async def test_app_headless_message_with_tool_calls_submit():
    app = InteractiveRolloutApp()
    task = asyncio.create_task(app.run_async(headless=True))
    await app.ready.wait()
    ask_task = asyncio.create_task(
        app.ask([UserMessage(content="hello")], [lookup_tool()])
    )
    await asyncio.sleep(0.1)

    app.query_one("#tool-picker", Select).value = "lookup_item"
    await asyncio.sleep(0.1)
    list(app.query(Input))[0].value = "7"
    app.query_one("#reasoning", TextArea).load_text("need item 7 first")
    app.query_one("#message", TextArea).load_text("looking that up")
    app._submit_current_turn()
    response = await ask_task
    app.exit()
    await task

    assert response.content == "looking that up"
    assert response.reasoning == "need item 7 first"
    assert response.tool_calls is not None
    assert response.tool_calls[0].name == "lookup_item"
    assert json.loads(response.tool_calls[0].arguments) == {"item_id": 7}


def code_tool() -> Tool:
    return Tool(
        name="ipython",
        description="Run code in an IPython cell.",
        parameters={
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        },
    )


@pytest.mark.asyncio
async def test_app_message_only_turn_with_tool_selected():
    app = InteractiveRolloutApp()
    async with app.run_test() as pilot:
        ask_task = asyncio.create_task(
            app.ask([UserMessage(content="hi")], [lookup_tool()])
        )
        await pilot.pause()
        # Select a tool to inspect its schema but leave the form blank.
        app.query_one("#tool-picker", Select).value = "lookup_item"
        await pilot.pause()
        app.query_one("#message", TextArea).load_text("just a message")
        app._submit_current_turn()
        response = await ask_task

    assert response.content == "just a message"
    assert response.tool_calls is None


@pytest.mark.parametrize(
    "url",
    [
        "http://169.254.169.254/latest",
        "file:///etc/passwd",
        "/etc/hosts",
    ],
)
def test_external_images_not_loaded_by_default(url):
    from rich.console import Console

    app = InteractiveRolloutApp()
    renderable = app._image_part_renderable(
        {"type": "image_url", "image_url": {"url": url}}
    )
    assert isinstance(renderable, Panel)
    buffer = io.StringIO()
    Console(file=buffer, width=100).print(renderable)
    assert "allow-external-images" in buffer.getvalue()


def test_data_url_image_still_renders_by_default():
    # Inline data: images are self-contained and always render.
    app = InteractiveRolloutApp()
    renderable = app._image_part_renderable(
        {"type": "image_url", "image_url": {"url": tiny_png_data_url()}}
    )
    assert isinstance(renderable, Panel)


def test_schema_type_resolves_optional_anyof():
    schema = {"anyOf": [{"type": "boolean"}, {"type": "null"}], "title": "b"}
    assert InteractiveRolloutApp.schema_type(schema) == "boolean"
    # An optional bool entered as "false" is parsed to the JSON boolean.
    assert InteractiveRolloutApp.parse_human_value("false", schema) is False


def test_schema_type_leaves_genuine_union_as_freeform():
    # int | str must not be forced to one arm's type; stay free-form (None).
    schema = {"anyOf": [{"type": "integer"}, {"type": "string"}]}
    assert InteractiveRolloutApp.schema_type(schema) is None
    assert InteractiveRolloutApp.parse_human_value("hello", schema) == "hello"


@pytest.mark.asyncio
async def test_app_string_argument_supports_multiline_code():
    app = InteractiveRolloutApp()
    async with app.run_test() as pilot:
        ask_task = asyncio.create_task(
            app.ask([UserMessage(content="run something")], [code_tool()])
        )
        await pilot.pause()
        app.query_one("#tool-picker", Select).value = "ipython"
        await pilot.pause()
        cell = app.query_one(".arg-input", TextArea)
        cell.focus()
        await pilot.pause()
        await pilot.press("l", "s")
        await pilot.press("shift+enter")
        await pilot.press("p", "w", "d")
        assert cell.text == "ls\npwd"

        await pilot.press("enter")
        assert [call.name for call in app._pending_tool_calls] == ["ipython"]
        assert json.loads(app._pending_tool_calls[0].arguments) == {"code": "ls\npwd"}
        assert cell.text == ""

        app._submit_current_turn()
        response = await ask_task

    assert response.tool_calls is not None
    assert json.loads(response.tool_calls[0].arguments) == {"code": "ls\npwd"}


@pytest.mark.asyncio
async def test_app_turn_preview_shows_composed_turn():
    from rich.console import Console

    app = InteractiveRolloutApp()
    task = asyncio.create_task(app.run_async(headless=True))
    await app.ready.wait()
    ask_task = asyncio.create_task(
        app.ask([UserMessage(content="hello")], [lookup_tool()])
    )
    await asyncio.sleep(0.1)

    app.query_one("#tool-picker", Select).value = "lookup_item"
    await asyncio.sleep(0.1)
    list(app.query(Input))[0].value = "7"
    app.query_one("#reasoning", TextArea).load_text("check item seven")
    app.query_one("#message", TextArea).load_text("on it")
    buffer = io.StringIO()
    Console(file=buffer, width=120).print(app._turn_preview_renderable())
    preview = buffer.getvalue()

    app._submit_current_turn()
    await ask_task
    app.exit()
    await task

    assert "check item seven" in preview
    assert "on it" in preview
    assert "lookup_item" in preview
    assert "7" in preview


@pytest.mark.asyncio
async def test_app_headless_tool_submit():
    app = InteractiveRolloutApp()
    task = asyncio.create_task(app.run_async(headless=True))
    await app.ready.wait()
    ask_task = asyncio.create_task(
        app.ask([UserMessage(content="hello")], [lookup_tool()])
    )
    await asyncio.sleep(0.1)

    app.query_one("#tool-picker", Select).value = "lookup_item"
    await asyncio.sleep(0.1)
    list(app.query(Input))[0].value = "7"
    app._submit_current_turn()
    response = await ask_task
    app.exit()
    await task

    assert response.tool_calls is not None
    assert response.tool_calls[0].name == "lookup_item"
    assert json.loads(response.tool_calls[0].arguments) == {"item_id": 7}


@pytest.mark.asyncio
async def test_app_headless_parallel_tool_submit():
    app = InteractiveRolloutApp()
    task = asyncio.create_task(app.run_async(headless=True))
    await app.ready.wait()
    ask_task = asyncio.create_task(
        app.ask([UserMessage(content="hello")], [lookup_tool(), label_tool()])
    )
    await asyncio.sleep(0.1)

    app.query_one("#tool-picker", Select).value = "lookup_item"
    await asyncio.sleep(0.1)
    list(app.query(Input))[0].value = "7"
    app._add_current_tool_call()

    app.query_one("#tool-picker", Select).value = "label_item"
    await asyncio.sleep(0.1)
    app.query_one(".arg-input", TextArea).load_text("urgent")
    app._submit_current_turn()
    response = await ask_task
    app.exit()
    await task

    assert response.tool_calls is not None
    assert [tool_call.name for tool_call in response.tool_calls] == [
        "lookup_item",
        "label_item",
    ]
    assert json.loads(response.tool_calls[0].arguments) == {"item_id": 7}
    assert json.loads(response.tool_calls[1].arguments) == {"label": "urgent"}


@pytest.mark.asyncio
async def test_app_ask_after_quit_raises_session_exit():
    app = InteractiveRolloutApp()
    task = asyncio.create_task(app.run_async(headless=True))
    await app.ready.wait()

    app.request_quit()
    await task
    with pytest.raises(InteractiveSessionExit):
        await app.ask([UserMessage(content="hello")], [])


@pytest.mark.asyncio
async def test_client_start_surfaces_tui_startup_failure():
    # If the TUI task dies before signalling ready, start() must raise the
    # startup error instead of hanging on ready.wait() forever.
    client = HumanClient(headless=True)

    async def boom(*args, **kwargs):
        raise RuntimeError("no terminal")

    client._app.run_async = boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="no terminal"):
        await asyncio.wait_for(client.start(), timeout=5)


@pytest.mark.asyncio
async def test_app_quit_via_q_keypress_fails_pending_turn():
    # Exercises the real binding path: Textual awaits the async ``action_quit``,
    # which must still fail the pending turn and stop the app.
    app = InteractiveRolloutApp()
    async with app.run_test() as pilot:
        ask_task = asyncio.create_task(app.ask([UserMessage(content="hi")], []))
        await pilot.pause()
        app.set_focus(None)
        await pilot.press("q")
        await pilot.pause()
        with pytest.raises(InteractiveSessionExit):
            await asyncio.wait_for(ask_task, timeout=2)
    assert app._quit_requested is True


@pytest.mark.asyncio
async def test_app_answer_popup():
    from verifiers.cli.interactive.app import AnswerScreen

    app = InteractiveRolloutApp(answer="42 apples")
    async with app.run_test() as pilot:
        ask_task = asyncio.create_task(app.ask([UserMessage(content="hi")], []))
        await pilot.pause()
        app.set_focus(None)
        await pilot.press("a")
        assert isinstance(app.screen, AnswerScreen)
        assert "42 apples" in app.screen._answer
        await pilot.press("escape")
        assert not isinstance(app.screen, AnswerScreen)

        app.query_one("#message", TextArea).load_text("done")
        app._submit_current_turn()
        await ask_task


def test_app_answer_binding_hidden_without_answer():
    assert InteractiveRolloutApp().check_action("show_answer", ()) is None
    assert InteractiveRolloutApp(answer="42").check_action("show_answer", ()) is True


@pytest.mark.asyncio
async def test_quit_mid_rollout_records_session_exit():
    from datasets import Dataset

    import verifiers as vf

    env = vf.ToolEnv(
        tools=[],
        dataset=Dataset.from_list(
            [{"prompt": [{"role": "user", "content": "go"}], "example_id": 0}]
        ),
        rubric=vf.Rubric(funcs=[]),
        max_turns=3,
    )
    env.set_score_rollouts(False)
    client = HumanClient(headless=True)
    await client.start()

    async def quit_soon():
        await asyncio.sleep(0.3)
        client._app.request_quit()

    quitter = asyncio.create_task(quit_soon())
    state = await asyncio.wait_for(
        env._run_rollout_state(
            {"prompt": [{"role": "user", "content": "go"}], "example_id": 0},
            client,
            "human",
            {},
        ),
        timeout=15,
    )
    await quitter
    await client.close()

    assert isinstance(state.get("error"), InteractiveSessionExit)


@pytest.mark.asyncio
async def test_app_quit_fails_pending_turn():
    app = InteractiveRolloutApp()
    task = asyncio.create_task(app.run_async(headless=True))
    await app.ready.wait()
    ask_task = asyncio.create_task(app.ask([UserMessage(content="hello")], []))
    await asyncio.sleep(0.1)

    app.request_quit()
    with pytest.raises(InteractiveSessionExit):
        await ask_task
    await task


@pytest.mark.asyncio
async def test_human_client_converts_app_response_to_vf_response(monkeypatch):
    client = HumanClient(headless=True)

    async def fake_start():
        return None

    async def fake_ask(prompt, tools):
        assert prompt == [UserMessage(content="Your move.")]
        assert tools == [lookup_tool()]
        return TurnResponse(
            content=None,
            tool_calls=[
                ToolCall(
                    id="call_1",
                    name="lookup_item",
                    arguments=json.dumps({"item_id": 4}),
                )
            ],
        )

    monkeypatch.setattr(client, "start", fake_start)
    monkeypatch.setattr(client._app, "ask", fake_ask)

    response = await client.get_response(
        prompt=[UserMessage(content="Your move.")],
        model="human",
        sampling_args={},
        tools=[lookup_tool()],
    )

    assert response.message.finish_reason == "tool_calls"
    assert response.message.tool_calls is not None
    assert response.message.tool_calls[0].name == "lookup_item"
    assert json.loads(response.message.tool_calls[0].arguments) == {"item_id": 4}


@pytest.mark.asyncio
async def test_human_client_converts_message_response(monkeypatch):
    client = HumanClient(headless=True)

    async def fake_start():
        return None

    async def fake_ask(prompt, tools):
        _ = prompt, tools
        return TurnResponse(
            content="final answer",
            tool_calls=None,
            reasoning="thinking out loud",
        )

    monkeypatch.setattr(client, "start", fake_start)
    monkeypatch.setattr(client._app, "ask", fake_ask)

    response = await client.get_response(
        prompt=[UserMessage(content="Answer.")],
        model="human",
        sampling_args={},
        tools=None,
    )

    assert response.message.finish_reason == "stop"
    assert response.message.content == "final answer"
    assert response.message.reasoning_content == "thinking out loud"
    assert response.message.tool_calls is None


def test_app_renders_image_url_content_parts():
    app = InteractiveRolloutApp()
    data_url = tiny_png_data_url()

    payload = InteractiveRolloutApp.load_image_url(data_url)
    renderables = app._content_renderables(
        [
            {"type": "text", "text": "describe this"},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    )

    assert payload.size == (2, 2)
    assert payload.path.read_bytes().startswith(b"\x89PNG")
    assert isinstance(renderables[1], Panel)


def test_play_parse_args_env_args_and_defaults():
    args = play.parse_args(["wordle", "--env-args", '{"mode": "text"}'])
    assert args.env_id == "wordle"
    assert args.env_args == {"mode": "text"}


def test_play_parse_args_accepts_env_id_after_options():
    args = play.parse_args(["--split", "eval", "wordle"])
    assert args.env_id == "wordle"
    assert args.split == "eval"


def test_play_parse_args_env_id_optional():
    args = play.parse_args(["--split", "eval"])
    assert args.env_id is None


def test_play_infers_env_id_from_current_environment_dir(tmp_path: Path):
    env_dir = tmp_path / "workspace" / "environments" / "sample_env"
    env_dir.mkdir(parents=True)
    (env_dir / "pyproject.toml").write_text(
        '[project]\nname = "sample-env"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )

    assert play.infer_current_env_id(env_dir) == "sample-env"


def test_play_resolves_default_env_dir_from_nested_environment(tmp_path: Path):
    env_root = tmp_path / "workspace" / "environments"
    env_dir = env_root / "sample_env"
    env_dir.mkdir(parents=True)

    assert play.resolve_env_dir_path("./environments", env_dir) == str(env_root)


def test_run_interactive_rollout_constructs_tui_client(monkeypatch):
    captured: dict[str, object] = {}

    class DummyEnv:
        def set_score_rollouts(self, value):
            captured["score_rollouts"] = value

        def get_eval_dataset(self, seed=None):
            _ = seed
            return [
                {
                    "prompt": [{"role": "user", "content": "hi"}],
                    "example_id": 0,
                    "answer": "42",
                }
            ]

        async def _run_rollout_state(self, rollout_input, client, model, sampling_args):
            captured["rollout_input"] = rollout_input
            captured["client"] = client
            captured["model"] = model
            captured["sampling_args"] = sampling_args
            return {"timing": {}, "example_id": 0}

    async def fake_start(self):
        _ = self

    monkeypatch.setattr(play, "prepare_local_env_import", lambda *_: None)
    monkeypatch.setattr(play.vf, "load_environment", lambda *_, **__: DummyEnv())
    monkeypatch.setattr(HumanClient, "start", fake_start)

    args = SimpleNamespace(
        env_id="demo",
        env_dir_path="./environments",
        env_args={},
        no_score=True,
        split="eval",
        index=0,
        shuffle=False,
        shuffle_seed=None,
        hide_prompt=False,
        hide_tools=True,
        allow_external_images=False,
        model="human",
        sampling_args={},
    )

    asyncio.run(play.run_interactive_rollout(args))

    assert captured["score_rollouts"] is False
    assert isinstance(captured["client"], HumanClient)
    assert captured["client"]._app._answer == "42"
    assert captured["model"] == "human"


def test_run_interactive_rollout_infers_current_env(monkeypatch, tmp_path: Path):
    captured: dict[str, object] = {}
    env_root = tmp_path / "workspace" / "environments"
    env_dir = env_root / "sample_env"
    env_dir.mkdir(parents=True)
    (env_dir / "pyproject.toml").write_text(
        '[project]\nname = "sample-env"\nversion = "0.1.0"\n',
        encoding="utf-8",
    )

    class DummyEnv:
        def set_score_rollouts(self, value):
            captured["score_rollouts"] = value

        def get_eval_dataset(self, seed=None):
            _ = seed
            return [{"prompt": [{"role": "user", "content": "hi"}], "example_id": 0}]

        async def _run_rollout_state(self, rollout_input, client, model, sampling_args):
            _ = rollout_input, client, model, sampling_args
            return {"timing": {}, "example_id": 0}

    def fake_prepare_local_env_import(env_id, env_dir_path):
        captured["env_id"] = env_id
        captured["env_dir_path"] = env_dir_path

    async def fake_start(self):
        _ = self

    monkeypatch.chdir(env_dir)
    monkeypatch.setattr(play, "prepare_local_env_import", fake_prepare_local_env_import)
    monkeypatch.setattr(play.vf, "load_environment", lambda *_, **__: DummyEnv())
    monkeypatch.setattr(HumanClient, "start", fake_start)

    args = SimpleNamespace(
        env_id=None,
        env_dir_path="./environments",
        env_args={},
        no_score=True,
        split="eval",
        index=0,
        shuffle=False,
        shuffle_seed=None,
        hide_prompt=False,
        hide_tools=True,
        allow_external_images=False,
        model="human",
        sampling_args={},
    )

    asyncio.run(play.run_interactive_rollout(args))

    assert captured["env_id"] == "sample-env"
    assert captured["env_dir_path"] == str(env_root)


def test_prepare_local_env_import_adds_selected_environment_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    env_dir = tmp_path / "environments"
    package_dir = env_dir / "sample_env"
    package_dir.mkdir(parents=True)
    (package_dir / "sample_env.py").write_text("", encoding="utf-8")
    original_path = list(sys.path)
    monkeypatch.setattr(sys, "path", list(original_path))

    play.prepare_local_env_import("sample-env", str(env_dir))

    assert sys.path[0] == str(package_dir.resolve())
