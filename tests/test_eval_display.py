from rich.console import Console

from verifiers.types import ClientConfig, EvalConfig
from verifiers.utils.eval_display import EvalDisplay


def make_config(
    *,
    max_concurrent: int,
    rollouts_per_example: int = 1,
    independent_scoring: bool = False,
    endpoint_id: str | None = None,
    client_config: ClientConfig | None = None,
) -> EvalConfig:
    return EvalConfig(
        env_id="dummy-env",
        env_args={},
        env_dir_path="./environments",
        endpoint_id=endpoint_id,
        model="gpt-4.1-mini",
        client_config=client_config or ClientConfig(),
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


def test_format_client_target_uses_endpoint_id_summary_for_multi_endpoint() -> None:
    config = make_config(
        max_concurrent=1,
        endpoint_id="gpt-5-mini",
        client_config=ClientConfig(
            api_base_url="http://localhost:8000/v1",
            endpoint_configs=[
                ClientConfig(api_base_url="http://localhost:8000/v1"),
                ClientConfig(api_base_url="http://localhost:8001/v1"),
            ],
        ),
    )

    assert (
        EvalDisplay._format_client_target(config)
        == "endpoint_id=gpt-5-mini (2 endpoints)"
    )


def test_format_client_target_uses_single_resolved_base_url() -> None:
    config = make_config(
        max_concurrent=1,
        endpoint_id="gpt-5-mini",
        client_config=ClientConfig(
            api_base_url="http://localhost:8000/v1",
            endpoint_configs=[ClientConfig(api_base_url="http://localhost:8001/v1")],
        ),
    )

    assert EvalDisplay._format_client_target(config) == "http://localhost:8001/v1"


def render_plain(renderable) -> str:
    console = Console(width=100, record=True)
    console.print(renderable)
    return console.export_text()


def test_tokens_row_omits_cost_when_unavailable() -> None:
    display = EvalDisplay([make_config(max_concurrent=1)])

    rendered = render_plain(
        display._make_tokens_row({"input_tokens": 12.0, "output_tokens": 7.0})
    )

    assert "input 12" in rendered
    assert "output 7" in rendered
    assert "cost" not in rendered


def test_tokens_row_includes_cost_when_available() -> None:
    display = EvalDisplay([make_config(max_concurrent=1)])

    rendered = render_plain(
        display._make_tokens_row(
            {
                "input_tokens": 12.0,
                "output_tokens": 7.0,
                "final_input_tokens": 10.0,
                "final_output_tokens": 5.0,
            },
            {"input_usd": 0.005, "output_usd": 0.0073, "total_usd": 0.0123},
        )
    )

    assert "input 12" in rendered
    assert "output 7" in rendered
    assert "final_input 10" in rendered
    assert "final_output 5" in rendered
    assert "cost (all) $0.0123" in rendered
    assert rendered.index("final_output 5") < rendered.index("cost (all) $0.0123")
