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
    CompareRunsScreen,
    CopyScreen,
    LazyRunResults,
    MathDisplayBlock,
    MathMarkdown,
    MathParagraph,
    RolloutCopyScreen,
    RunBrowserTree,
    RunInfo,
    VerifiersTUI,
    ViewRunScreen,
    _build_reward_distribution_table,
    _compute_run_overview_stats,
    _extract_numeric_metric_values,
    _varying_run_setting_keys,
    format_info_for_details,
    make_math_parser,
    render_block_math,
    render_inline_math,
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


def test_view_run_screen_info_details_include_saved_state_columns(tmp_path) -> None:
    run_dir = tmp_path / "demo-run"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "num_examples": 1,
                "rollouts_per_example": 1,
                "state_columns": ["judge_response", "attempt_count"],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "reward": 0.75,
                "info": {"trace": "ok"},
                "judge_response": {"winner": "assistant"},
                "attempt_count": 3,
                "prompt": [{"role": "user", "content": "Solve it"}],
                "completion": [{"role": "assistant", "content": "Done"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    screen = ViewRunScreen(
        RunInfo(
            env_id="demo-env",
            model="openai/gpt-5",
            run_id="run-1",
            path=run_dir,
        )
    )

    rendered = screen._build_info_text(screen.records[0]).plain.rstrip()

    assert "Info" in rendered
    assert '"trace": "ok"' in rendered
    assert "judge_response" in rendered
    assert '"winner": "assistant"' in rendered
    assert "attempt_count" in rendered
    assert "\n3" in rendered


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


def test_make_math_parser_parses_inline_block_and_amsmath_tokens() -> None:
    parser = make_math_parser()
    tokens = parser.parse(
        r"""
Inline $E = mc^2$

$$
\sum_x p(x)
$$

\begin{align}
f(x) &= x^2 + 1 \\
g(x) &= \frac{1}{1 + e^{-x}}
\end{align}
"""
    )

    inline_token = next(token for token in tokens if token.type == "inline")

    assert any(child.type == "math_inline" for child in inline_token.children or [])
    assert any(token.type == "math_block" for token in tokens)
    assert any(token.type == "amsmath" for token in tokens)


def test_latex_renderers_convert_common_math_to_plain_text() -> None:
    inline = render_inline_math(r"\alpha = \frac{1}{2}")
    block = render_block_math(
        r"""
\begin{align}
f(x) &= x^2 + 1 \\
g(x) &= \frac{1}{1 + e^{-x}}
\end{align}
""".strip()
    )

    assert "α" in inline
    assert "1/2" in inline
    assert "\\begin" not in block
    assert "\\end" not in block
    assert "\\frac" not in block
    assert "f(x)" in block
    assert "g(x)" in block


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
    assert "sub llm completion tokens" in rendered
    assert "28,698" in rendered
    assert "877" in rendered
    assert "56,519" in rendered
    assert "sub llm call count" in rendered
    assert "Distribution" in rendered


def test_reward_distribution_uses_exact_zero_and_one_buckets() -> None:
    rendered = _render_to_text(
        _build_reward_distribution_table(
            [0.0, 0.0, 0.1, 0.4, 0.8, 1.0],
            "Rollout rewards",
        )
    )
    lines = [line.strip() for line in rendered.splitlines()]

    assert "=0" in rendered
    assert "=1" in rendered
    assert "0-<0.25" in rendered
    assert "<0" not in lines
    assert ">=1.0" not in lines


def test_build_run_details_includes_run_settings(tmp_path) -> None:
    run_dir = tmp_path / "demo-run"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "base_url": "https://api.example/v1",
                "num_examples": 12,
                "rollouts_per_example": 4,
                "pass_threshold": 0.7,
                "sampling_args": {
                    "temperature": 0.2,
                    "max_tokens": 512,
                },
                "env_args": {
                    "difficulty": "hard",
                    "split": "validation",
                },
                "state_columns": ["judge_response"],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "results.jsonl").write_text(
        json.dumps({"reward": 1.0, "metrics": {}}) + "\n",
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

    assert "Run settings" in rendered
    assert "endpoint" in rendered
    assert "https://api.example/v1" in rendered
    assert "examples" in rendered
    assert "rollouts/example" in rendered
    assert "pass threshold" in rendered
    assert "sampling.temperature" in rendered
    assert "sampling.max_tokens" in rendered
    assert "env.difficulty" in rendered
    assert "env.split" in rendered
    assert "state columns" in rendered


def test_model_details_include_setting_variations(tmp_path) -> None:
    first_run_dir = tmp_path / "run-a"
    first_run_dir.mkdir()
    (first_run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "avg_reward": 0.25,
                "sampling_args": {"temperature": 0.2},
                "env_args": {"split": "train"},
            }
        ),
        encoding="utf-8",
    )
    (first_run_dir / "results.jsonl").write_text("{}\n", encoding="utf-8")

    second_run_dir = tmp_path / "run-b"
    second_run_dir.mkdir()
    (second_run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "avg_reward": 0.75,
                "sampling_args": {"temperature": 0.8},
                "env_args": {"split": "validation"},
            }
        ),
        encoding="utf-8",
    )
    (second_run_dir / "results.jsonl").write_text("{}\n", encoding="utf-8")

    first_run = RunInfo(
        env_id="demo-env",
        model="openai/gpt-5",
        run_id="run-1",
        path=first_run_dir,
    )
    second_run = RunInfo(
        env_id="demo-env",
        model="openai/gpt-5",
        run_id="run-2",
        path=second_run_dir,
    )

    rendered = _render_to_text(
        BrowseRunsScreen(
            {"demo-env": {"openai/gpt-5": [first_run, second_run]}}
        )._build_model_details("demo-env", "openai/gpt-5")
    )

    assert "Setting variations" in rendered
    assert "sampling.temperature" in rendered
    assert "env.split" in rendered
    assert "0.2 (1 run)" in rendered
    assert "0.8 (1 run)" in rendered
    assert "train (1 run)" in rendered
    assert "validation (1 run)" in rendered


def test_compare_runs_screen_renders_settings_and_reward_buckets(tmp_path) -> None:
    first_run_dir = tmp_path / "run-a"
    first_run_dir.mkdir()
    (first_run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "avg_reward": 0.25,
                "sampling_args": {"temperature": 0.2, "max_tokens": 256},
                "env_args": {"split": "train"},
            }
        ),
        encoding="utf-8",
    )
    (first_run_dir / "results.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"reward": 0.0, "metrics": {}}),
                json.dumps({"reward": 0.5, "metrics": {}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    second_run_dir = tmp_path / "run-b"
    second_run_dir.mkdir()
    (second_run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "avg_reward": 0.75,
                "sampling_args": {"temperature": 0.8, "max_tokens": 512},
                "env_args": {"split": "validation"},
            }
        ),
        encoding="utf-8",
    )
    (second_run_dir / "results.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"reward": 1.0, "metrics": {}}),
                json.dumps({"reward": 1.0, "metrics": {}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    first_run = RunInfo(
        env_id="demo-env",
        model="openai/gpt-5",
        run_id="run-1",
        path=first_run_dir,
    )
    second_run = RunInfo(
        env_id="demo-env",
        model="openai/gpt-5",
        run_id="run-2",
        path=second_run_dir,
    )
    screen = CompareRunsScreen(
        "demo-env",
        "openai/gpt-5",
        [first_run, second_run],
    )
    stats = {
        first_run.path: _compute_run_overview_stats(first_run),
        second_run.path: _compute_run_overview_stats(second_run),
    }

    # Simulate _finish_loading_comparison_stats setup
    screen._stats_by_path = stats
    screen._setting_keys, screen._run_settings = _varying_run_setting_keys(screen.runs)

    rendered = _render_to_text(
        screen._build_comparison_outcomes(),
        width=220,
    )

    assert "Outcome groups" in rendered
    assert "temperature" in rendered
    assert "max_tokens" in rendered
    assert "split" in rendered
    assert "rollouts" in rendered
    assert (
        "unique" in rendered
    )  # "unique prompts" header (may be truncated at narrow widths)
    assert "=0" in rendered
    assert "=1" in rendered
    assert "█" in rendered
    assert "50%" in rendered
    assert "100%" in rendered


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

        assert bindings["left"].show is True
        assert bindings["left"].description == "Parent folder"
        assert bindings["right"].show is True
        assert bindings["right"].description == "Expand/next folder"
        assert bindings["enter"].show is True
        assert bindings["enter"].description == "Open/toggle"
        assert bindings["space"].show is True
        assert bindings["space"].description == "Toggle folder"
        assert any(
            binding.key == "v" and binding.description == "Compare"
            for binding in pilot.app.screen.BINDINGS
        )
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
async def test_browse_run_screen_highlights_details_pane_when_focused(
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

        tree = pilot.app.screen.query_one("#run-browser-tree", RunBrowserTree)
        scroll = pilot.app.screen.query_one(
            "#run-browser-details-scroll", VerticalScroll
        )

        assert pilot.app.focused is tree
        assert tree.styles.background_tint.a > 0
        assert scroll.styles.background_tint.a == 0

        await pilot.press("tab")
        await pilot.pause()

        assert pilot.app.focused is scroll
        assert tree.styles.background_tint.a == 0
        assert scroll.styles.background_tint.a > 0


@pytest.mark.asyncio
async def test_browse_run_screen_opens_compare_mode_for_model(tmp_path) -> None:
    first_run_dir = tmp_path / "run-a"
    first_run_dir.mkdir()
    (first_run_dir / "metadata.json").write_text(
        json.dumps({"avg_reward": 0.25, "sampling_args": {"temperature": 0.2}}),
        encoding="utf-8",
    )
    (first_run_dir / "results.jsonl").write_text("{}\n", encoding="utf-8")

    second_run_dir = tmp_path / "run-b"
    second_run_dir.mkdir()
    (second_run_dir / "metadata.json").write_text(
        json.dumps({"avg_reward": 0.75, "sampling_args": {"temperature": 0.8}}),
        encoding="utf-8",
    )
    (second_run_dir / "results.jsonl").write_text("{}\n", encoding="utf-8")

    first_run = RunInfo(
        env_id="demo-env",
        model="openai/gpt-5",
        run_id="run-1",
        path=first_run_dir,
    )
    second_run = RunInfo(
        env_id="demo-env",
        model="openai/gpt-5",
        run_id="run-2",
        path=second_run_dir,
    )

    async with VerifiersTUI(
        {"demo-env": {"openai/gpt-5": [first_run, second_run]}}
    ).run_test() as pilot:
        await pilot.pause()

        tree = pilot.app.screen.query_one("#run-browser-tree", RunBrowserTree)
        assert tree.cursor_node is not None
        assert tree.cursor_node.data.kind == "run"

        await pilot.press("left")
        await pilot.pause()
        assert tree.cursor_node is not None
        assert tree.cursor_node.data.kind == "model"

        await pilot.press("v")
        await pilot.pause()

        assert isinstance(pilot.app.screen, CompareRunsScreen)


@pytest.mark.asyncio
async def test_browse_run_tree_left_arrow_moves_to_visible_parent(tmp_path) -> None:
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

        assert tree.cursor_node is not None
        assert tree.cursor_node.data.kind == "run"

        await pilot.press("left")
        await pilot.pause()
        assert tree.cursor_node is not None
        assert tree.cursor_node.data.kind == "model"

        await pilot.press("left")
        await pilot.pause()
        assert tree.cursor_node is not None
        assert tree.cursor_node.data.kind == "env"

        await pilot.press("left")
        await pilot.pause()
        assert tree.cursor_node is not None
        assert tree.cursor_node.data.kind == "env"


@pytest.mark.asyncio
async def test_browse_run_tree_space_collapses_parent_folder_from_run_leaf(
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

        tree = pilot.app.screen.query_one("#run-browser-tree", RunBrowserTree)
        assert tree.cursor_node is not None
        model_node = tree.cursor_node.parent

        assert model_node is not None
        assert model_node.allow_expand is True
        assert model_node.is_expanded is True

        await pilot.press("space")
        await pilot.pause()

        assert tree.cursor_node is model_node
        assert tree.cursor_node.data.kind == "model"
        assert model_node.is_expanded is False


@pytest.mark.asyncio
async def test_browse_run_tree_right_arrow_moves_leaf_to_next_parent_folder(
    tmp_path,
) -> None:
    first_run_dir = tmp_path / "run-a"
    first_run_dir.mkdir()
    (first_run_dir / "metadata.json").write_text(
        json.dumps({"avg_reward": 0.25}),
        encoding="utf-8",
    )
    (first_run_dir / "results.jsonl").write_text("{}\n", encoding="utf-8")

    second_run_dir = tmp_path / "run-b"
    second_run_dir.mkdir()
    (second_run_dir / "metadata.json").write_text(
        json.dumps({"avg_reward": 0.75}),
        encoding="utf-8",
    )
    (second_run_dir / "results.jsonl").write_text("{}\n", encoding="utf-8")

    first_run = RunInfo(
        env_id="demo-env",
        model="a/model",
        run_id="run-1",
        path=first_run_dir,
    )
    second_run = RunInfo(
        env_id="demo-env",
        model="b/model",
        run_id="run-2",
        path=second_run_dir,
    )

    async with VerifiersTUI(
        {"demo-env": {"a/model": [first_run], "b/model": [second_run]}}
    ).run_test() as pilot:
        await pilot.pause()

        tree = pilot.app.screen.query_one("#run-browser-tree", RunBrowserTree)

        assert tree.cursor_node is not None
        assert tree.cursor_node.data.kind == "run"
        assert tree.cursor_node.data.model == "a/model"

        await pilot.press("right")
        await pilot.pause()

        assert tree.cursor_node is not None
        assert tree.cursor_node.data.kind == "model"
        assert tree.cursor_node.data.model == "b/model"


@pytest.mark.asyncio
async def test_browse_run_tree_right_arrow_expands_collapsed_folder(tmp_path) -> None:
    first_run_dir = tmp_path / "run-a"
    first_run_dir.mkdir()
    (first_run_dir / "metadata.json").write_text(
        json.dumps({"avg_reward": 0.25}),
        encoding="utf-8",
    )
    (first_run_dir / "results.jsonl").write_text("{}\n", encoding="utf-8")

    second_run_dir = tmp_path / "run-b"
    second_run_dir.mkdir()
    (second_run_dir / "metadata.json").write_text(
        json.dumps({"avg_reward": 0.75}),
        encoding="utf-8",
    )
    (second_run_dir / "results.jsonl").write_text("{}\n", encoding="utf-8")

    first_run = RunInfo(
        env_id="alpha-env",
        model="model-a",
        run_id="run-1",
        path=first_run_dir,
    )
    second_run = RunInfo(
        env_id="beta-env",
        model="model-b",
        run_id="run-2",
        path=second_run_dir,
    )

    async with VerifiersTUI(
        {"alpha-env": {"model-a": [first_run]}, "beta-env": {"model-b": [second_run]}}
    ).run_test() as pilot:
        await pilot.pause()

        tree = pilot.app.screen.query_one("#run-browser-tree", RunBrowserTree)
        collapsed_env = tree.root.children[1]

        assert collapsed_env.data.kind == "env"
        assert collapsed_env.is_expanded is False

        tree.move_cursor(collapsed_env)
        await pilot.pause()
        await pilot.press("right")
        await pilot.pause()

        assert tree.cursor_node is collapsed_env
        assert collapsed_env.is_expanded is True


@pytest.mark.asyncio
async def test_browse_run_tree_clips_long_labels_without_horizontal_scrollbar(
    tmp_path,
) -> None:
    run_dir = tmp_path / "demo-run"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        json.dumps({"avg_reward": 0.75}),
        encoding="utf-8",
    )
    (run_dir / "results.jsonl").write_text("{}\n", encoding="utf-8")

    long_model = "openai/" + ("very-long-model-name-" * 8)
    long_run_id = "run-" + ("1234567890" * 12)
    run = RunInfo(
        env_id="demo-env",
        model=long_model,
        run_id=long_run_id,
        path=run_dir,
    )

    async with VerifiersTUI({"demo-env": {long_model: [run]}}).run_test() as pilot:
        await pilot.pause()

        tree = pilot.app.screen.query_one("#run-browser-tree", RunBrowserTree)
        model_line = tree.render_line(1).text

        assert tree.size.width > 0
        assert tree.virtual_size.width <= tree.size.width
        assert tree.styles.overflow_x == "hidden"
        assert tree.show_horizontal_scrollbar is False
        assert "…" in model_line
        assert "1 runs" in model_line


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


@pytest.mark.asyncio
async def test_view_run_screen_renders_history_sections_with_math_markdown(
    tmp_path,
) -> None:
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
                "prompt": [
                    {
                        "role": "user",
                        "content": "Solve $E = mc^2$",
                    }
                ],
                "completion": [
                    {
                        "role": "assistant",
                        "content": (
                            "Fraction $\\alpha = \\frac{1}{2}$\n\n$$\n\\sum_x p(x)\n$$"
                        ),
                    }
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

    async with ViewRunHarness(screen).run_test() as pilot:
        await pilot.pause()
        markdown_widgets = list(screen.query(MathMarkdown))
        math_widget = next(
            widget
            for widget in markdown_widgets
            if "\\alpha = \\frac{1}{2}" in widget.source
        )
        paragraphs = list(math_widget.query(MathParagraph))
        display = math_widget.query_one(MathDisplayBlock)

        assert any(
            "α" in paragraph._content.plain and "1/2" in paragraph._content.plain
            for paragraph in paragraphs
        )
        assert "sum" in display._content.plain
        assert "\\" not in display._content.plain

        await pilot.press("m")
        await pilot.pause()
        assert not list(screen.query(MathMarkdown))

        await pilot.press("m")
        await pilot.pause()
        assert list(screen.query(MathMarkdown))


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


def test_format_prompt_or_completion_includes_reasoning_traces(tmp_path) -> None:
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
            {
                "role": "assistant",
                "reasoning_content": "hidden chain",
                "content": "final answer",
            },
            {
                "role": "assistant",
                "thinking_blocks": [
                    {
                        "type": "thinking",
                        "thinking": "tool plan",
                        "signature": "sig_1",
                    },
                    {
                        "type": "redacted_thinking",
                        "data": "opaque",
                    },
                ],
                "content": [
                    {"type": "thinking", "thinking": "tool plan"},
                    {"type": "text", "text": "after thought"},
                ],
            },
        ]
    )

    assert "reasoning:\nhidden chain" in rendered.plain
    assert "final answer" in rendered.plain
    assert rendered.plain.count("tool plan") == 1
    assert "[reasoning redacted]" in rendered.plain
    assert "after thought" in rendered.plain


def test_history_section_data_includes_reasoning_traces(tmp_path) -> None:
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

    sections = screen._history_section_data(
        {
            "prompt": [{"role": "user", "content": "start"}],
            "completion": [
                {
                    "role": "assistant",
                    "reasoning_content": "step 1",
                    "content": "final answer",
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "step 2"},
                        {"type": "text", "text": "another answer"},
                    ],
                },
            ],
        }
    )

    assert sections[1].body == "final answer"
    assert sections[1].body_first is False
    assert sections[1].nested_sections[0].title == "Reasoning"
    assert sections[1].nested_sections[0].body == "step 1"
    assert sections[2].body == "another answer"
    assert sections[2].body_first is False
    assert sections[2].nested_sections[0].title == "Reasoning"
    assert sections[2].nested_sections[0].body == "step 2"
    assert (
        screen._render_history_section_copy_text(sections[1])
        == "1. assistant  final answer\n\n  Reasoning\n\n    step 1\n\n  final answer"
    )


def test_view_run_screen_sorts_pass_metrics_numerically(tmp_path) -> None:
    run_dir = tmp_path / "demo-run"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "pass_at_k": {"1": 0.1, "10": 1.0, "2": 0.2},
                "pass_all_k": {"1": 0.01, "10": 0.1, "2": 0.02},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "results.jsonl").write_text("{}\n", encoding="utf-8")

    screen = ViewRunScreen(
        RunInfo(
            env_id="demo-env",
            model="openai/gpt-5",
            run_id="run-1",
            path=run_dir,
        )
    )

    rendered = screen._build_header_metric_text().plain
    labels = [token for token in rendered.split() if token.startswith("pass")]

    assert labels == [
        "pass@1",
        "pass@2",
        "pass@10",
        "pass-all@1",
        "pass-all@2",
        "pass-all@10",
    ]


def test_view_run_screen_history_summary_omits_inline_hints(tmp_path) -> None:
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

    rendered = screen._build_history_summary_text(
        {
            "completion": [
                {"role": "assistant", "content": "first"},
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "second"},
            ]
        }
    ).plain

    assert "3 events" in rendered
    assert "1 user turns" in rendered
    assert "Enter toggles" not in rendered
    assert "PgUp/PgDn scroll" not in rendered


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
                "state_columns": ["judge_response"],
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
                "judge_response": {"winner": "assistant"},
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
    assert "judge_response" in items["details:details-info"].body
    assert '"winner": "assistant"' in items["details:details-info"].body


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
