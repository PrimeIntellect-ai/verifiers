import asyncio
import json
from pathlib import Path

from verifiers.types import State
from verifiers.utils.eval_utils import serialize_rollout


class StreamingHandler:
    def __init__(
        self,
        results_path: Path,
        total_rollouts: int,
        state_columns: list[str] | None = None,
    ):
        self.results_path = results_path
        self.total_rollouts = total_rollouts
        self.state_columns = state_columns or []
        self.completed_count = 0
        self.running_reward_sum = 0.0
        self.running_metrics_sum: dict[str, float] = {}

        self.results_path.parent.mkdir(parents=True, exist_ok=True)

    def log_rollout(self, state: State) -> None:
        self.completed_count += 1
        reward = state.get("reward", 0.0)
        self.running_reward_sum += reward

        metrics = state.get("metrics", {})
        for k, v in metrics.items():
            self.running_metrics_sum[k] = self.running_metrics_sum.get(k, 0.0) + v

    def _write_sync(self, json_line: str) -> None:
        with open(self.results_path, "a") as f:
            f.write(json_line)

    async def write_rollout_jsonl(self, state: State) -> None:
        rollout_data = serialize_rollout(
            state,
            state_columns=self.state_columns,
            include_timestamp=True,
        )
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
