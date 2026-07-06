from pathlib import Path
from typing import Any

from verifiers.v1 import legacy
from verifiers.v1.configs.eval import EvalConfig


class GroupOnlyEnv:
    requires_group_rollouts = True

    def __init__(self) -> None:
        self.group_calls: list[dict[str, Any]] = []

    def get_eval_dataset(self) -> list[dict[str, Any]]:
        return [
            {
                "prompt": [{"role": "user", "content": "repeat"}],
                "answer": "repeat",
                "info": {},
            }
        ]

    async def run_rollout(self, **kwargs) -> dict[str, Any]:
        raise AssertionError("group-scored envs must use run_group")

    async def run_group(self, **kwargs) -> list[dict[str, Any]]:
        self.group_calls.append(kwargs)
        outputs = []
        for idx, group_input in enumerate(kwargs["group_inputs"]):
            reward = float(idx + 1)
            outputs.append(
                {
                    "example_id": group_input.get("example_id", 0),
                    "model": kwargs["model"],
                    "prompt": group_input["prompt"],
                    "answer": group_input["answer"],
                    "reward": reward,
                    "metrics": {"group_reward": reward},
                    "is_completed": True,
                    "is_truncated": False,
                    "trajectory": [],
                    "timing": {},
                }
            )
        return outputs


async def test_legacy_eval_uses_group_rollouts_for_group_reward_env(
    monkeypatch, tmp_path: Path
) -> None:
    env = GroupOnlyEnv()

    import verifiers

    monkeypatch.setattr(verifiers, "load_environment", lambda *args, **kwargs: env)
    monkeypatch.setattr(legacy, "_eval_client", lambda *args, **kwargs: object())

    traces = await legacy.run_legacy_eval(
        EvalConfig(
            id="group-only-v0",
            num_tasks=1,
            num_rollouts=2,
            max_concurrent=1,
            model="test-model",
            output_dir=tmp_path,
        )
    )

    assert len(env.group_calls) == 1
    assert len(env.group_calls[0]["group_inputs"]) == 2
    assert env.group_calls[0]["state_columns"] == ["trajectory"]
    assert [trace.reward for trace in traces] == [1.0, 2.0]
    assert (tmp_path / "results.jsonl").read_text().count("\n") == 2
