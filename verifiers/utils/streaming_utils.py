import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from verifiers.types import Info, Messages, State
from verifiers.utils.message_utils import messages_to_printable, sanitize_tool_calls


class StreamingHandler:
    def __init__(
        self,
        results_path: Path,
        total_rollouts: int,
        state_columns: list[str] | None = None,
        show_total: bool = True,
    ):
        self.results_path = results_path
        self.total_rollouts = total_rollouts
        self.state_columns = state_columns or []
        self.show_total = show_total
        self.completed_count = 0
        self.running_reward_sum = 0.0
        self.running_metrics_sum: dict[str, float] = {}
        self.sample_rollout_count: dict[int, int] = {}

        self.results_path.parent.mkdir(parents=True, exist_ok=True)

    def log_rollout(
        self,
        example_id: int,
        reward: float,
        metrics: dict[str, float],
        timing: dict[str, float],
    ) -> None:
        self.completed_count += 1
        self.running_reward_sum += reward

        for k, v in metrics.items():
            self.running_metrics_sum[k] = self.running_metrics_sum.get(k, 0.0) + v

        self.sample_rollout_count[example_id] = (
            self.sample_rollout_count.get(example_id, 0) + 1
        )

    def _write_sync(self, json_line: str) -> None:
        with open(self.results_path, "a") as f:
            f.write(json_line)

    async def write_rollout_jsonl(
        self,
        example_id: int,
        prompt: Messages,
        completion: Messages,
        answer: str,
        reward: float,
        metrics: dict[str, float],
        state: State,
        task: str,
        info: Info,
    ) -> None:
        clean_prompt = messages_to_printable(prompt)
        clean_prompt = sanitize_tool_calls(clean_prompt)
        clean_completion = messages_to_printable(completion)
        clean_completion = sanitize_tool_calls(clean_completion)

        rollout_data: dict[str, Any] = {
            "example_id": example_id,
            "prompt": clean_prompt,
            "completion": clean_completion,
            "task": task,
            "reward": reward,
            "timestamp": datetime.now().isoformat(),
        }

        if "timing" in state:
            rollout_data["generation_ms"] = state["timing"].get("generation_ms", 0.0)
            rollout_data["scoring_ms"] = state["timing"].get("scoring_ms", 0.0)
            rollout_data["total_ms"] = state["timing"].get("total_ms", 0.0)

        for k, v in metrics.items():
            rollout_data[k] = v

        if answer:
            rollout_data["answer"] = answer
        if info and info != {}:
            rollout_data["info"] = info

        if self.state_columns:
            for col in self.state_columns:
                if col in state:
                    if col == "responses":
                        rollout_data[col] = [r.model_dump() for r in state[col]]
                    else:
                        rollout_data[col] = state[col]

        json_line = json.dumps(rollout_data) + "\n"
        await asyncio.to_thread(self._write_sync, json_line)

    def get_running_stats(self) -> dict[str, float]:
        if self.completed_count == 0:
            return {}

        stats = {
            "avg_reward": self.running_reward_sum / self.completed_count,
            "completed": self.completed_count,
            "total": self.total_rollouts,
        }

        for k, v in self.running_metrics_sum.items():
            stats[f"avg_{k}"] = v / self.completed_count

        return stats
