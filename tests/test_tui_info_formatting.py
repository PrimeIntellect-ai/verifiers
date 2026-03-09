import json
from io import StringIO

import pytest
from rich.console import Console
from rich.text import Text
from textual.app import App
from textual.containers import VerticalScroll
from textual.widgets import OptionList, Static, TextArea, Tree

from verifiers.scripts.tui import (
    BrowseRunsScreen,
    CopyScreen,
    RolloutCopyScreen,
    RunInfo,
    VerifiersTUI,
    ViewRunScreen,
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

    rendered = _render_to_text(BrowseRunsScreen({})._build_run_details(run))

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
async def test_view_run_screen_loads_rollout_previews_after_mount(tmp_path) -> None:
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
