import json
from pathlib import Path

from textual.widgets import Collapsible, OptionList, Static

from verifiers.scripts.tui import VerifiersTUI


def _write_eval_fixture(tmp_path: Path) -> tuple[Path, Path]:
    env_dir = tmp_path / "environments"
    outputs_dir = tmp_path / "outputs"
    run_dir = outputs_dir / "evals" / "tool_test--openai--gpt-4.1-mini" / "run-001"
    run_dir.mkdir(parents=True)

    metadata = {
        "num_examples": 1,
        "rollouts_per_example": 2,
        "avg_reward": 0.5,
        "date": "2026-03-08",
        "time": "09:00:00",
        "sampling_args": {"temperature": 0.2, "max_tokens": 512},
        "avg_metrics": {"num_turns": 2.5, "total_tool_calls": 1.5},
        "pass_at_k": {"1": 0.5},
        "pass_all_k": {"1": 0.0},
        "pass_threshold": 0.5,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    records = [
        {
            "prompt": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "solve it"},
            ],
            "completion": [{"role": "assistant", "content": "bad answer"}],
            "reward": 0.0,
            "info": {"judge": {"reason": "wrong"}},
            "task": "task-a",
        },
        {
            "prompt": [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "solve it"},
            ],
            "completion": [
                {"role": "assistant", "content": "thinking"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "search",
                                "arguments": '{"q":"x"}',
                            }
                        }
                    ],
                },
                {"role": "tool", "content": "tool output"},
                {"role": "assistant", "content": "final answer"},
            ],
            "reward": 1.0,
            "judge_reward": 1.0,
            "tool_reward": 1.0,
            "info": {"judge": {"reason": "correct"}},
            "task": "task-a",
        },
    ]
    with (run_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    return env_dir, outputs_dir


def _render_plain(widget: Static) -> str:
    rendered = widget.render()
    return rendered.plain if hasattr(rendered, "plain") else str(rendered)


def _focus_collapsible_title(section: Collapsible) -> None:
    next(iter(section.children)).focus()


async def _open_view_screen(pilot) -> None:
    await pilot.pause()
    await pilot.press("down", "enter", "down", "enter", "down", "enter")
    await pilot.pause()


async def test_eval_tui_completion_history_is_primary(tmp_path: Path) -> None:
    env_dir, outputs_dir = _write_eval_fixture(tmp_path)
    app = VerifiersTUI(str(env_dir), str(outputs_dir))

    async with app.run_test(size=(150, 42)) as pilot:
        await _open_view_screen(pilot)

        rollout_list = app.screen.query_one("#rollout-list", OptionList)
        history_scroll = app.screen.query_one("#completion-scroll")
        metadata_summary = app.screen.query_one("#metadata-summary", Static)
        metadata_metrics = app.screen.query_one("#metadata-metrics", Static)
        metadata_reward = app.screen.query_one("#metadata-reward", Static)
        history_summary = app.screen.query_one("#history-summary", Static)
        rollouts_panel = app.screen.query_one("#rollouts-panel")
        details_panel = app.screen.query_one("#details-panel")
        first_turn = app.screen.query_one("#turn-1", Collapsible)
        prompt_context = app.screen.query_one("#prompt-context", Collapsible)
        first_turn_body = app.screen.query_one("#turn-1-body", Static)

        assert rollout_list.option_count == 2
        assert rollouts_panel.display is True
        assert details_panel.display is True
        top_level_ids = [
            child.id
            for child in history_scroll.children
            if isinstance(child, Collapsible)
        ]
        assert top_level_ids[0] == "prompt-context"
        assert first_turn.collapsed is True
        assert prompt_context.collapsed is True
        assert "bad answer" in _render_plain(first_turn_body)
        assert "Run Summary" in _render_plain(metadata_summary)
        assert "Run ID" in _render_plain(metadata_summary)
        assert "pass@1" in _render_plain(metadata_metrics)
        reward_text = _render_plain(metadata_reward)
        assert "Current Reward" in reward_text
        assert "0.000" in reward_text
        assert "events" in _render_plain(history_summary)
        assert "PgUp/PgDn scroll" in _render_plain(history_summary)


async def test_eval_tui_rollout_navigation_uses_arrow_keys(tmp_path: Path) -> None:
    env_dir, outputs_dir = _write_eval_fixture(tmp_path)
    app = VerifiersTUI(str(env_dir), str(outputs_dir))

    async with app.run_test(size=(150, 42)) as pilot:
        await _open_view_screen(pilot)
        await pilot.press("right")
        await pilot.pause()

        rollout_list = app.screen.query_one("#rollout-list", OptionList)
        score = app.screen.query_one("#score-content", Static)
        reward = app.screen.query_one("#metadata-reward", Static)
        final_turn = app.screen.query_one("#turn-3-body", Static)
        final_turn_section = app.screen.query_one("#turn-3", Collapsible)

        assert rollout_list.highlighted == 1
        assert final_turn_section.collapsed is True
        assert "final answer" in _render_plain(final_turn)
        reward_text = _render_plain(reward)
        assert "1.000" in reward_text
        score_text = _render_plain(score)
        assert "judge_reward" in score_text
        assert "tool_reward" in score_text


async def test_eval_tui_sections_are_collapsible_and_expandable(tmp_path: Path) -> None:
    env_dir, outputs_dir = _write_eval_fixture(tmp_path)
    app = VerifiersTUI(str(env_dir), str(outputs_dir))

    async with app.run_test(size=(150, 42)) as pilot:
        await _open_view_screen(pilot)
        await pilot.press("right")
        await pilot.pause()

        prompt_context = app.screen.query_one("#prompt-context", Collapsible)
        tool_call = app.screen.query_one("#turn-2-tool-1", Collapsible)

        assert prompt_context.collapsed is True
        assert tool_call.collapsed is True

        await pilot.press("e")
        await pilot.pause()

        assert prompt_context.collapsed is False
        assert tool_call.collapsed is False

        _focus_collapsible_title(tool_call)
        await pilot.press("enter")
        await pilot.pause()
        assert tool_call.collapsed is True

        await pilot.press("enter")
        await pilot.pause()
        assert tool_call.collapsed is False

        await pilot.press("x")
        await pilot.pause()

        assert app.screen.query_one("#turn-3", Collapsible).collapsed is True
        assert prompt_context.collapsed is True


async def test_eval_tui_history_supports_paged_scrolling(tmp_path: Path) -> None:
    env_dir, outputs_dir = _write_eval_fixture(tmp_path)
    app = VerifiersTUI(str(env_dir), str(outputs_dir))

    async with app.run_test(size=(120, 18)) as pilot:
        await _open_view_screen(pilot)
        await pilot.press("right", "e")
        await pilot.pause()

        history_scroll = app.screen.query_one("#completion-scroll")
        start_scroll = history_scroll.scroll_y
        assert history_scroll.max_scroll_y > 0

        await pilot.press("pagedown")
        await pilot.pause()

        assert history_scroll.scroll_y > start_scroll


async def test_eval_tui_hides_side_panels_on_small_terminals(tmp_path: Path) -> None:
    env_dir, outputs_dir = _write_eval_fixture(tmp_path)
    app = VerifiersTUI(str(env_dir), str(outputs_dir))

    async with app.run_test(size=(120, 24)) as pilot:
        await _open_view_screen(pilot)

        assert app.screen.query_one("#rollouts-panel").display is False
        assert app.screen.query_one("#details-panel").display is False
