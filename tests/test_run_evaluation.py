from unittest.mock import AsyncMock, patch

import pytest

from verifiers.types import ClientConfig, EvalConfig
from verifiers.utils.eval_utils import run_evaluation


@pytest.mark.asyncio
async def test_run_evaluation_builds_dataset_before_starting_env_server():
    order: list[tuple[str, int]] = []

    class FakeEnv:
        def set_kwargs(self, **kwargs):
            return None

        def get_eval_dataset(self, n: int = -1, seed=None):
            order.append(("get_eval_dataset", n))
            raise RuntimeError("dataset unavailable")

        start_server = AsyncMock()
        stop_server = AsyncMock()

    fake_env = FakeEnv()
    config = EvalConfig(
        env_id="hle",
        env_args={},
        env_dir_path="./environments",
        model="openai/gpt-4.1-mini",
        client_config=ClientConfig(),
        sampling_args={},
        num_examples=10,
        rollouts_per_example=3,
        max_concurrent=1,
    )

    with patch("verifiers.utils.eval_utils.vf.load_environment", return_value=fake_env):
        with pytest.raises(RuntimeError, match="dataset unavailable"):
            await run_evaluation(config)

    assert order == [("get_eval_dataset", 1)]
    fake_env.start_server.assert_not_awaited()
    fake_env.stop_server.assert_not_awaited()
