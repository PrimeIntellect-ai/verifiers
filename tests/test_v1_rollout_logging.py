import logging

from verifiers.v1.utils.rollout_log_utils import (
    log_rollout_finish,
    log_rollout_start,
    logger,
    rollout_timing_summary,
    rollout_tool_counts,
)


class RecordingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def capture_rollout_logs() -> RecordingHandler:
    # The verifiers logger sets propagate=False, so capture on the rollout
    # logger directly rather than relying on root propagation (e.g. caplog).
    handler = RecordingHandler()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return handler


def make_trajectory() -> list[dict]:
    return [
        {
            "completion": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "1", "name": "search", "arguments": "{}"},
                        {"id": "2", "name": "search", "arguments": "{}"},
                    ],
                }
            ]
        },
        {
            "completion": [
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "3", "name": "read", "arguments": "{}"}],
                }
            ]
        },
        {"completion": [{"role": "assistant", "content": "done"}]},
    ]


def test_rollout_tool_counts_handles_dict_and_function_shapes() -> None:
    trajectory = make_trajectory()
    trajectory.append(
        {
            "completion": [
                {"role": "assistant", "tool_calls": [{"function": {"name": "read"}}]}
            ]
        }
    )
    counts = rollout_tool_counts(trajectory)
    assert counts == {"search": 2, "read": 2}


def test_rollout_tool_counts_ignores_non_list() -> None:
    assert rollout_tool_counts(None) == {}


def test_rollout_timing_summary_matches_tui_style() -> None:
    timing = {
        "setup": {"duration": 1.2},
        "generation": {"duration": 45.0},
        "scoring": {"duration": 0.5},
        "model": {"duration": 30.0},
        "env": {"duration": 15.0},
        "overhead": 2.0,
    }
    summary = rollout_timing_summary(timing)
    assert summary == (
        "setup 1s + generation 45s (model 30s + env 15s) + scoring 500ms + overhead 2s"
    )


def test_log_rollout_start_and_finish_lines() -> None:
    handler = capture_rollout_logs()
    state = {
        "example_id": 7,
        "trajectory_id": "abc123",
        "reward": 1.5,
        "metrics": {"num_turns": 3.0, "correct": 1.0},
        "stop_condition": "program_completed",
        "is_truncated": False,
        "timing": {"generation": {"duration": 4.0}, "model": {"duration": 3.0}},
        "trajectory": make_trajectory(),
    }
    try:
        log_rollout_start(state)
        log_rollout_finish(state)
    finally:
        logger.removeHandler(handler)

    start_line, finish_line = handler.messages
    assert start_line == "Started  example_id=7 | trajectory_id=abc123"
    assert "Finished example_id=7 | trajectory_id=abc123" in finish_line
    assert "turns=" not in finish_line
    assert "tools=[search: 2, read: 1]" in finish_line
    assert "stop=program_completed" in finish_line
    assert "reward=1.5" in finish_line
    assert "metrics={num_turns: 3, correct: 1}" in finish_line
    assert "truncated" not in finish_line
    assert "error=" not in finish_line


def test_log_rollout_finish_includes_error_and_truncation() -> None:
    handler = capture_rollout_logs()
    state = {
        "example_id": 1,
        "trajectory_id": "deadbeef",
        "reward": 0.0,
        "metrics": {},
        "stop_condition": "has_error",
        "is_truncated": True,
        "timing": {},
        "trajectory": [],
        "error": {
            "error": "TimeoutError",
            "error_chain_str": "ToolError -> TimeoutError",
        },
    }
    try:
        log_rollout_finish(state)
    finally:
        logger.removeHandler(handler)

    line = handler.messages[-1]
    assert "error=ToolError -> TimeoutError" in line
    assert "truncated=True" in line
