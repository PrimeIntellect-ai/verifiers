from verifiers.types import ClientConfig, EvalConfig
from verifiers.utils.eval_display import EvalDisplay


def make_config(
    *,
    max_concurrent: int,
    rollouts_per_example: int = 1,
    independent_scoring: bool = False,
) -> EvalConfig:
    return EvalConfig(
        env_id="dummy-env",
        env_args={},
        env_dir_path="./environments",
        model="gpt-4.1-mini",
        client_config=ClientConfig(),
        sampling_args={},
        num_examples=5,
        rollouts_per_example=rollouts_per_example,
        max_concurrent=max_concurrent,
        independent_scoring=independent_scoring,
    )


def test_display_max_concurrent_caps_to_total_rollouts() -> None:
    config = make_config(max_concurrent=32)

    assert EvalDisplay._display_max_concurrent(config, total_rollouts=8) == 8


def test_display_max_concurrent_uses_rollout_level_concurrency() -> None:
    config = make_config(max_concurrent=9, rollouts_per_example=4)

    # Grouped scoring semaphore is applied at group level, but UI shows
    # rollout-level concurrency by rounding up to whole groups, then capping
    # to available total rollouts.
    assert EvalDisplay._display_max_concurrent(config, total_rollouts=10) == 10


def test_display_max_concurrent_does_not_scale_independent_scoring() -> None:
    config = make_config(
        max_concurrent=9, rollouts_per_example=4, independent_scoring=True
    )

    assert EvalDisplay._display_max_concurrent(config, total_rollouts=10) == 9
