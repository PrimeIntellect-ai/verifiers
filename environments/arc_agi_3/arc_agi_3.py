from typing import Any

from datasets import Dataset

import verifiers as vf
from verifiers.envs.experimental.arc_agi_3_env import ArcAgi3Env


def load_environment(
    game_ids: list[str] | None = None,
    history_window_turns: int = 3,
    max_turns: int = 40,
    save_recording: bool = True,
    tool_calls_log_path: str | None = "arc_tool_calls.jsonl",
    scorecard_tags: list[str] | None = None,
    operation_mode: str | None = "online",
    arc_base_url: str = "https://three.arcprize.org",
    **kwargs: Any,
) -> vf.Environment:
    """Load ARC-AGI-3 environment with dataset-driven game selection."""
    game_ids = game_ids or ["ls20"]
    rows = []
    for idx, game_id in enumerate(game_ids):
        rows.append(
            {
                "example_id": idx,
                "prompt": [{"role": "user", "content": "<prompt not used>"}],
                "task": "arc_agi_3",
                "answer": "",
                "info": {"game_id": game_id},
            }
        )

    dataset = Dataset.from_list(rows)

    def win_reward(state: vf.State, **_kwargs: Any) -> float:
        return 1.0 if state.get("arc_status") == "WIN" else 0.0

    rubric = vf.Rubric(funcs=[win_reward])
    return ArcAgi3Env(
        dataset=dataset,
        rubric=rubric,
        history_window_turns=history_window_turns,
        max_turns=max_turns,
        save_recording=save_recording,
        tool_calls_log_path=tool_calls_log_path,
        scorecard_tags=scorecard_tags,
        operation_mode=operation_mode,  # type: ignore[arg-type]
        arc_base_url=arc_base_url,
        **kwargs,
    )
