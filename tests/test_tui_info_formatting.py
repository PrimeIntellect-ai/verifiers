import json
from io import StringIO

import pytest
from rich.console import Console
from rich.text import Text
from textual.app import App
from textual.containers import VerticalScroll
from textual.widgets import Label, OptionList, Static, TextArea, Tree

from verifiers.scripts.tui import (
    BrowseRunsScreen,
    CopyScreen,
    LazyRunResults,
    RolloutCopyScreen,
    RunBrowserTree,
    RunInfo,
    VerifiersTUI,
    ViewRunScreen,
    _compute_run_overview_stats,
    _extract_numeric_metric_values,
    format_info_for_details,
)


def _render_to_text(renderable: object, width: int = 180) -> str:
    buffer = StringIO()
    Console(
        file=buffer,
        force_terminal=False,
        color_system=None,
        width=width,
    ).print(renderable)
    return buffer.getvalue()


class ViewRunHarness(App[None]):
    def __init__(self, screen: ViewRunScreen):
        super().__init__()
        self._screen = screen

    def on_mount(self) -> None:
        self.push_screen(self._screen)


def test_lazy_run_results_counts_actual_file_length_when_metadata_is_stale(
    tmp_path,
) -> None:
    run_dir = tmp_path / "demo-run"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        json.dumps({"num_examples": 3, "rollouts_per_example": 1}),
        encoding="utf-8",
    )
    (run_dir / "results.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"reward": 0.1}),
                json.dumps({"reward": 0.2}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records = LazyRunResults(
        RunInfo(
            env_id="demo-env",
            model="openai/gpt-5",
            run_id="run-1",
            path=run_dir,
        )
    )

    try:
        assert records.count_hint() == 3
        assert len(records) == 2
        assert records.count_hint() == 2
        assert records[2] == {}
    finally:
        records.close()


def test_format_info_for_details_handles_dict() -> None:
    info = {"status": "ok", "attempt": 2}

    rendered = format_info_for_details(info)

    assert rendered == json.dumps(info, ensure_ascii=False, indent=2)


def test_format_info_for_details_parses_json_string() -> None:
    info = '{"status":"ok","nested":{"value":1}}'

    rendered = format_info_for_details(info)

    assert rendered == json.dumps(
        {"status": "ok", "nested": {"value": 1}},
        ensure_ascii=False,
        indent=2,
    )


def test_format_info_for_details_preserves_large_content() -> None:
    info = {"payload": [f"line-{i}" for i in range(200)]}

    rendered = format_info_for_details(info)

    assert "line-199" in rendered
    assert "(truncated;" not in rendered


def test_format_info_for_details_handles_non_serializable_data() -> None:
    info: dict[str, object] = {"callback": lambda: "x"}

    rendered = format_info_for_details(info)

    assert "callback" in rendered
    assert "function" in rendered


def test_extract_numeric_metric_values_includes_metrics_and_reward_signals() -> None:
    record = {
        "metrics": {"judge": 0.25, "tool_calls": 3},
        "info": json.dumps({"reward_signals": {"format_reward": 1.0}}),
        "sub_llm_completion_tokens": 144,
        "prompt": "ignored",
    }

    rendered = _extract_numeric_metric_values(record)

    assert rendered == {
        "judge": 0.25,
        "tool_calls": 3.0,
        "format_reward": 1.0,
        "sub_llm_completion_tokens": 144.0,
    }


def test_build_run_details_includes_rollout_metric_stats(tmp_path) -> None:
    run_dir = tmp_path / "demo-run"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "avg_reward": 0.75,
                "num_examples": 2,
                "rollouts_per_example": 1,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "results.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "reward": 0.5,
                        "metrics": {
                            "sub_llm_completion_tokens": 877,
                            "sub_llm_call_count": 2,
                        },
                    }
                ),
                json.dumps(
                    {
                        "reward": 1.0,
                        "metrics": {
                            "sub_llm_completion_tokens": 56519,
                            "sub_llm_call_count": 4,
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    run = RunInfo(
        env_id="demo-env",
        model="openai/gpt-5",
        run_id="run-1",
        path=run_dir,
    )

    rendered = _render_to_text(
        BrowseRunsScreen({})._build_run_details(run, _compute_run_overview_stats(run))
    )

    assert "Rollout metrics" in rendered
    assert "Average" in rendered
    assert "Min" in rendered
    assert "Max" in rendered
    assert "sub-LLM completion tokens" in rendered
    assert "28,698" in rendered
    assert "877" in rendered
    assert "56,519" in rendered
    assert "sub-LLM call count" in rendered
    assert "Distribution" in rendered


def test_copy_screen_uses_custom_labels() -> None:
    copy_screen = CopyScreen(
        "run-1 reward 0.75",
        "details text",
        "completion",
        prompt_label="Selection",
        completion_label="Details",
        title="Copy Details",
    )

    assert copy_screen._prompt_label == "Selection"
    assert copy_screen._completion_label == "Details"
    assert copy_screen._title == "Copy Details"
    assert copy_screen._prompt_text == "run-1 reward 0.75"
    assert copy_screen._completion_text == "details text"


def test_copy_screen_defaults_invalid_start_column_to_completion() -> None:
    copy_screen = CopyScreen("prompt text", "completion text", "sideways")

    assert copy_screen._active_column == "completion"


def test_populate_tree_includes_run_reward_in_label(tmp_path) -> None:
    run_dir = tmp_path / "demo-run"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        json.dumps({"avg_reward": 0.75}),
        encoding="utf-8",
    )
    (run_dir / "results.jsonl").write_text("{}\n", encoding="utf-8")

    run = RunInfo(
        env_id="demo-env",
        model="openai/gpt-5",
        run_id="run-1",
        path=run_dir,
    )
    screen = BrowseRunsScreen({"demo-env": {"openai/gpt-5": [run]}})
    tree = Tree("Completed evals")

    first_run_node = screen._populate_tree(tree)

    assert first_run_node is not None
    assert first_run_node.label.plain == "run-1  0.750"


@pytest.mark.asyncio
async def test_browse_run_screen_moves_browser_shortcuts_to_footer(tmp_path) -> None:
    run_dir = tmp_path / "demo-run"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        json.dumps({"avg_reward": 0.75}),
        encoding="utf-8",
    )
    (run_dir / "results.jsonl").write_text("{}\n", encoding="utf-8")

    run = RunInfo(
        env_id="demo-env",
        model="openai/gpt-5",
        run_id="run-1",
        path=run_dir,
    )

    async with VerifiersTUI({"demo-env": {"openai/gpt-5": [run]}}).run_test() as pilot:
        await pilot.pause()

        tree = pilot.app.screen.query_one("#run-browser-tree", RunBrowserTree)
        bindings = {binding.key: binding for binding in tree.BINDINGS}
        labels = [
            label.content.plain
            for label in pilot.app.screen.query(Label)
            if isinstance(label.content, Text)
        ]

        assert bindings["enter"].show is True
        assert bindings["enter"].description == "Open/toggle"
        assert bindings["space"].show is True
        assert bindings["space"].description == "Toggle folder"
        assert "Enter opens runs  Space toggles folders  c copies" not in labels


@pytest.mark.asyncio
async def test_browse_run_screen_offsets_details_content_from_scrollbar(
    tmp_path,
) -> None:
    run_dir = tmp_path / "demo-run"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        json.dumps({"avg_reward": 0.75}),
        encoding="utf-8",
    )
    (run_dir / "results.jsonl").write_text("{}\n", encoding="utf-8")

    run = RunInfo(
        env_id="demo-env",
        model="openai/gpt-5",
        run_id="run-1",
        path=run_dir,
    )

    async with VerifiersTUI({"demo-env": {"openai/gpt-5": [run]}}).run_test() as pilot:
        await pilot.pause()

        scroll = pilot.app.screen.query_one(
            "#run-browser-details-scroll", VerticalScroll
        )
        details = pilot.app.screen.query_one("#run-browser-details", Static)

        assert scroll.styles.padding.left == 2
        assert scroll.styles.padding.right == 1
        assert scroll.styles.scrollbar_gutter == "stable"
        assert details.styles.margin.right == 8


@pytest.mark.asyncio
async def test_view_run_screen_populates_rollout_rewards_for_all_rows(tmp_path) -> None:
    run_dir = tmp_path / "demo-run"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "avg_reward": 0.5,
                "num_examples": 3,
                "rollouts_per_example": 1,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "results.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "reward": 0.1,
                        "prompt": [{"role": "user", "content": "first prompt"}],
                        "completion": [
                            {"role": "assistant", "content": "first sample"}
                        ],
                    }
                ),
                json.dumps(
                    {
                        "reward": 0.2,
                        "prompt": [{"role": "user", "content": "second prompt"}],
                        "completion": [
                            {"role": "assistant", "content": "second sample"}
                        ],
                    }
                ),
                json.dumps(
                    {
                        "reward": 0.3,
                        "prompt": [{"role": "user", "content": "third prompt"}],
                        "completion": [
                            {"role": "assistant", "content": "third sample"}
                        ],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    run = RunInfo(
        env_id="demo-env",
        model="openai/gpt-5",
        run_id="run-1",
        path=run_dir,
    )
    screen = ViewRunScreen(run)

    async with ViewRunHarness(screen).run_test() as pilot:
        await pilot.pause()
        rollout_list = screen.query_one("#rollout-list", OptionList)
        first_prompt = rollout_list.get_option_at_index(0).prompt
        second_prompt = rollout_list.get_option_at_index(1).prompt
        third_prompt = rollout_list.get_option_at_index(2).prompt
        first_text = (
            first_prompt.plain if isinstance(first_prompt, Text) else str(first_prompt)
        )
        second_text = (
            second_prompt.plain
            if isinstance(second_prompt, Text)
            else str(second_prompt)
        )
        third_text = (
            third_prompt.plain if isinstance(third_prompt, Text) else str(third_prompt)
        )

        assert screen.records._cache.keys() == {0, 1, 2}
        assert "reward 0.100" in first_text
        assert "first sample" in first_text
        assert "reward 0.200" in second_text
        assert "second sample" in second_text
        assert "reward 0.300" in third_text
        assert "third sample" in third_text


@pytest.mark.asyncio
async def test_view_run_screen_ignores_metadata_rollout_count_when_file_is_short(
    tmp_path,
) -> None:
    run_dir = tmp_path / "demo-run"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "avg_reward": 0.5,
                "num_examples": 3,
                "rollouts_per_example": 2,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "reward": 0.5,
                "prompt": [{"role": "user", "content": "only prompt"}],
                "completion": [{"role": "assistant", "content": "only sample"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    run = RunInfo(
        env_id="demo-env",
        model="openai/gpt-5",
        run_id="run-1",
        path=run_dir,
    )
    screen = ViewRunScreen(run)

    async with ViewRunHarness(screen).run_test() as pilot:
        await pilot.pause()
        rollout_list = screen.query_one("#rollout-list", OptionList)
        first_prompt = rollout_list.get_option_at_index(0).prompt
        first_text = (
            first_prompt.plain if isinstance(first_prompt, Text) else str(first_prompt)
        )

        assert rollout_list.option_count == 1
        assert screen.records.count_hint() == 1
        assert "reward 0.500" in first_text
        assert "only sample" in first_text


def test_record_preview_uses_error_when_completion_is_empty_payload(tmp_path) -> None:
    run_dir = tmp_path / "demo-run"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text("{}", encoding="utf-8")
    (run_dir / "results.jsonl").write_text("{}\n", encoding="utf-8")

    screen = ViewRunScreen(
        RunInfo(
            env_id="demo-env",
            model="openai/gpt-5",
            run_id="run-1",
            path=run_dir,
        )
    )

    preview = screen._record_preview(
        {
            "prompt": [{"role": "user", "content": "original prompt"}],
            "completion": [{}],
            "error": {
                "error": "ModelError",
                "error_chain_str": "ModelError -> BadRequestError",
            },
        }
    )

    assert "ModelError" in preview
    assert "BadRequestError" in preview
    assert "original prompt" not in preview


def test_format_prompt_or_completion_handles_non_dict_entries(tmp_path) -> None:
    run_dir = tmp_path / "demo-run"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text("{}", encoding="utf-8")
    (run_dir / "results.jsonl").write_text("{}\n", encoding="utf-8")

    screen = ViewRunScreen(
        RunInfo(
            env_id="demo-env",
            model="openai/gpt-5",
            run_id="run-1",
            path=run_dir,
        )
    )

    rendered = screen._format_prompt_or_completion(
        [
            "raw message",
            {"role": "assistant", "content": "structured message"},
        ]
    )

    assert rendered.plain == "raw message\n\nassistant: structured message\n\n"


def test_view_run_screen_builds_rollout_copy_items_from_viewer_sections(
    tmp_path,
) -> None:
    run_dir = tmp_path / "demo-run"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "avg_reward": 0.75,
                "num_examples": 1,
                "rollouts_per_example": 1,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "reward": 0.75,
                "task": "Solve the puzzle",
                "answer": "42",
                "stop_condition": "done",
                "metrics": {"judge": 1.0},
                "token_usage": {"input_tokens": 123, "output_tokens": 45},
                "timing": {"generation_ms": 12, "scoring_ms": 3, "total_ms": 15},
                "info": {"trace": "ok"},
                "prompt": [{"role": "user", "content": "Solve it"}],
                "completion": [
                    {
                        "role": "assistant",
                        "content": "Checking",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "search",
                                    "arguments": {"query": "weather"},
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "content": "Sunny",
                    },
                    {"role": "assistant", "content": "It is sunny."},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    run = RunInfo(
        env_id="demo-env",
        model="openai/gpt-5",
        run_id="run-1",
        path=run_dir,
    )
    screen = ViewRunScreen(run)
    items = {
        item.key: item for item in screen._build_rollout_copy_items(screen.records[0])
    }

    assert "snapshot" in items
    assert "history" in items
    assert "details" in items
    assert "details:details-task" in items
    assert "Current Rollout" in items["snapshot"].body
    assert "Completion History" in items["snapshot"].body
    assert "Details (active: Task)" in items["snapshot"].body
    assert "Task\nSolve the puzzle" in items["details:details-task"].body
    assert "tool 1  search" in items["history"].body
    assert "Sunny" in items["history"].body
    assert "Tokens\ninput_tokens: 123" in items["details"].body


@pytest.mark.asyncio
async def test_view_run_screen_copy_action_opens_rollout_copy_screen(tmp_path) -> None:
    run_dir = tmp_path / "demo-run"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "avg_reward": 0.5,
                "num_examples": 1,
                "rollouts_per_example": 1,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "reward": 0.5,
                "task": "Task body",
                "prompt": [{"role": "user", "content": "hello"}],
                "completion": [{"role": "assistant", "content": "world"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    run = RunInfo(
        env_id="demo-env",
        model="openai/gpt-5",
        run_id="run-1",
        path=run_dir,
    )
    screen = ViewRunScreen(run)

    async with ViewRunHarness(screen).run_test() as pilot:
        await pilot.pause()
        await pilot.press("c")
        await pilot.pause()

        assert isinstance(pilot.app.screen, RolloutCopyScreen)

        copy_targets = pilot.app.screen.query_one("#rollout-copy-targets", OptionList)
        preview = pilot.app.screen.query_one("#rollout-copy-preview", TextArea)
        first_prompt = copy_targets.get_option_at_index(0).prompt
        first_text = (
            first_prompt.plain if isinstance(first_prompt, Text) else str(first_prompt)
        )

        assert first_text == "Full rollout snapshot"
        assert "hello" in preview.text
        assert "world" in preview.text
