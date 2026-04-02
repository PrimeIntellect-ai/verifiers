"""Tests for verifiers.utils.eval_utils.

Covers:
- print_results indexing with multiple rollouts per example
"""

from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

import verifiers.utils.eval_utils as eval_utils
from verifiers.types import ClientConfig, EvalConfig, GenerateOutputs
from verifiers.utils.save_utils import states_to_outputs


def test_print_results_rollout_indexing(capsys, make_metadata, make_state, make_input):
    """Test that print_results correctly groups results by rollout when sorted by example_id.

    Results are sorted by example_id, giving order: [ex0_r0, ex0_r1, ex1_r0, ex1_r1, ...]
    The indexing should correctly extract:
    - r1: all first rollouts (indices 0, 2, 4, ...)
    - r2: all second rollouts (indices 1, 3, 5, ...)
    """
    from verifiers.utils.eval_utils import print_results

    num_examples = 3
    rollouts_per_example = 2

    # Simulate results sorted by example_id (as generate() now does)
    # Order: [ex0_r0, ex0_r1, ex1_r0, ex1_r1, ex2_r0, ex2_r1]
    # Rewards are designed so we can verify correct grouping:
    # - All r0 rewards: 0.1, 0.3, 0.5 (for examples 0, 1, 2)
    # - All r1 rewards: 0.2, 0.4, 0.6 (for examples 0, 1, 2)
    rewards = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    example_ids = [0, 0, 1, 1, 2, 2]

    # Metric follows same pattern
    metric_values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    metadata = make_metadata(
        num_examples=num_examples, rollouts_per_example=rollouts_per_example
    )
    inputs = [make_input(example_id=example_id) for example_id in example_ids]
    states = [
        make_state(**input, reward=reward, metrics={"test_metric": metric_value})
        for input, reward, metric_value in zip(inputs, rewards, metric_values)
    ]
    rollout_outputs = states_to_outputs(states)

    results = GenerateOutputs(outputs=rollout_outputs, metadata=metadata)
    print_results(results)
    captured = capsys.readouterr()

    # Verify rollout groupings are correct
    # r1 should have rewards [0.1, 0.3, 0.5] (first rollout of each example)
    assert "r1: [0.1, 0.3, 0.5]" in captured.out
    # r2 should have rewards [0.2, 0.4, 0.6] (second rollout of each example)
    assert "r2: [0.2, 0.4, 0.6]" in captured.out

    # Same for metrics
    assert "r1: [1.0, 3.0, 5.0]" in captured.out
    assert "r2: [2.0, 4.0, 6.0]" in captured.out


def test_print_results_single_rollout(capsys, make_metadata, make_state, make_input):
    """Test print_results with single rollout per example (edge case)."""
    from verifiers.utils.eval_utils import print_results

    num_examples = 3
    rollouts_per_example = 1

    rewards = [0.1, 0.2, 0.3]
    example_ids = [0, 1, 2]

    metadata = make_metadata(
        num_examples=num_examples, rollouts_per_example=rollouts_per_example
    )
    states = [
        make_state(**make_input(example_id=example_id), reward=reward)
        for example_id, reward in zip(example_ids, rewards)
    ]
    rollout_outputs = states_to_outputs(states)

    results = GenerateOutputs(outputs=rollout_outputs, metadata=metadata)

    print_results(results)
    captured = capsys.readouterr()

    # With single rollout, r1 should have all rewards
    assert "r1: [0.1, 0.2, 0.3]" in captured.out


def test_print_results_three_rollouts(capsys, make_metadata, make_state, make_input):
    """Test print_results with three rollouts per example."""
    from verifiers.utils.eval_utils import print_results

    num_examples = 2
    rollouts_per_example = 3

    # Order: [ex0_r0, ex0_r1, ex0_r2, ex1_r0, ex1_r1, ex1_r2]
    rewards = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    example_ids = [0, 0, 0, 1, 1, 1]

    inputs = [make_input(example_id=example_id) for example_id in example_ids]
    states = [
        make_state(**input, reward=reward) for input, reward in zip(inputs, rewards)
    ]
    rollout_outputs = states_to_outputs(states)
    metadata = make_metadata(
        num_examples=num_examples, rollouts_per_example=rollouts_per_example
    )

    results = GenerateOutputs(outputs=rollout_outputs, metadata=metadata)

    print_results(results)
    captured = capsys.readouterr()

    # r1 should have [0.1, 0.4] (first rollout of each example)
    assert "r1: [0.1, 0.4]" in captured.out
    # r2 should have [0.2, 0.5] (second rollout of each example)
    assert "r2: [0.2, 0.5]" in captured.out
    # r3 should have [0.3, 0.6] (third rollout of each example)
    assert "r3: [0.3, 0.6]" in captured.out


def test_print_results_includes_usage(capsys, make_metadata, make_output):
    from verifiers.utils.eval_utils import print_results

    outputs = [
        make_output(example_id=0, reward=1.0, metrics={"test_metric": 1.0}),
        make_output(example_id=1, reward=0.0, metrics={"test_metric": 2.0}),
    ]
    outputs[0]["token_usage"] = {"input_tokens": 10.0, "output_tokens": 4.0}
    outputs[1]["token_usage"] = {"input_tokens": 6.0, "output_tokens": 2.0}
    metadata = make_metadata(num_examples=2, rollouts_per_example=1, usage=None)

    results = GenerateOutputs(outputs=outputs, metadata=metadata)
    print_results(results)
    captured = capsys.readouterr()

    assert "Usage:" in captured.out
    assert "input_tokens (avg): 8.000" in captured.out
    assert "output_tokens (avg): 3.000" in captured.out


@pytest.mark.asyncio
async def test_run_evaluation_offline_prepared_routes_without_env_server(
    monkeypatch, tmp_path, make_metadata
):
    expected_outputs: GenerateOutputs = {
        "outputs": [],
        "metadata": make_metadata(path_to_save=tmp_path / "results"),
    }

    fake_env = Mock()
    fake_env.set_kwargs = Mock()
    fake_env.start_server = AsyncMock()
    fake_env.stop_server = AsyncMock()
    fake_env.evaluate = AsyncMock()
    fake_env._evaluate_offline = AsyncMock(return_value=expected_outputs)

    monkeypatch.setattr(eval_utils.vf, "load_environment", lambda **kwargs: fake_env)
    monkeypatch.setattr(
        eval_utils,
        "load_prepared_outputs",
        lambda path: ([{"example_id": 0, "completion": "ok"}], {"model": "prepared"}),
    )

    config = EvalConfig(
        env_id="dummy-env",
        env_args={},
        env_dir_path=str(tmp_path),
        output_dir=str(tmp_path),
        model="prepared-model",
        client_config=ClientConfig(
            api_base_url="offline://local",
            api_key_var="OFFLINE_UNUSED",
        ),
        sampling_args={},
        num_examples=1,
        rollouts_per_example=1,
        max_concurrent=4,
        offline_mode="prepared_completions",
        prepared_completions_path=Path("/tmp/prepared.jsonl"),
        disable_env_server=False,
    )

    outputs = await eval_utils.run_evaluation(config)

    assert outputs == expected_outputs
    fake_env.set_kwargs.assert_not_called()
    fake_env.start_server.assert_not_awaited()
    fake_env.stop_server.assert_not_awaited()
    fake_env.evaluate.assert_not_awaited()
    fake_env._evaluate_offline.assert_awaited_once()
    assert fake_env._evaluate_offline.await_args.kwargs["prepared_outputs"] == [
        {"example_id": 0, "completion": "ok"}
    ]
    assert (
        fake_env._evaluate_offline.await_args.kwargs["use_ground_truth_as_completion"]
        is False
    )


@pytest.mark.asyncio
async def test_run_evaluation_offline_ground_truth_skips_prepared_loading(
    monkeypatch, tmp_path, make_metadata
):
    expected_outputs: GenerateOutputs = {
        "outputs": [],
        "metadata": make_metadata(path_to_save=tmp_path / "results"),
    }

    fake_env = Mock()
    fake_env.set_kwargs = Mock()
    fake_env.start_server = AsyncMock()
    fake_env.stop_server = AsyncMock()
    fake_env.evaluate = AsyncMock()
    fake_env._evaluate_offline = AsyncMock(return_value=expected_outputs)

    monkeypatch.setattr(eval_utils.vf, "load_environment", lambda **kwargs: fake_env)

    def fail_load_prepared_outputs(path):
        raise AssertionError("load_prepared_outputs should not be called")

    monkeypatch.setattr(eval_utils, "load_prepared_outputs", fail_load_prepared_outputs)

    config = EvalConfig(
        env_id="dummy-env",
        env_args={},
        env_dir_path=str(tmp_path),
        output_dir=str(tmp_path),
        model="offline/ground-truth",
        client_config=ClientConfig(
            api_base_url="offline://local",
            api_key_var="OFFLINE_UNUSED",
        ),
        sampling_args={},
        num_examples=2,
        rollouts_per_example=1,
        max_concurrent=4,
        offline_mode="ground_truth",
        disable_env_server=False,
    )

    outputs = await eval_utils.run_evaluation(config)

    assert outputs == expected_outputs
    fake_env.start_server.assert_not_awaited()
    fake_env.stop_server.assert_not_awaited()
    fake_env.evaluate.assert_not_awaited()
    fake_env._evaluate_offline.assert_awaited_once()
    assert fake_env._evaluate_offline.await_args.kwargs["prepared_outputs"] is None
    assert (
        fake_env._evaluate_offline.await_args.kwargs["use_ground_truth_as_completion"]
        is True
    )
