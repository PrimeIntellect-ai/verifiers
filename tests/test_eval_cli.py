import argparse
from pathlib import Path
from types import SimpleNamespace

import verifiers.scripts.eval as vf_eval
from verifiers.types import GenerateMetadata, GenerateOutputs


def _make_metadata(config) -> GenerateMetadata:
    return GenerateMetadata(
        env_id=config.env_id,
        env_args=config.env_args,
        model=config.model,
        base_url=config.client_config.api_base_url,
        num_examples=config.num_examples,
        rollouts_per_example=config.rollouts_per_example,
        sampling_args=config.sampling_args,
        date="1970-01-01",
        time_ms=0.0,
        avg_reward=0.0,
        avg_metrics={},
        state_columns=config.state_columns or [],
        path_to_save=Path("test.jsonl"),
    )


def _run_cli(monkeypatch, overrides):
    base_args = {
        "env_id": "dummy-env",
        "env_args": {},
        "env_dir_path": "./environments",
        "endpoints_path": "./configs/endpoints.py",
        "model": "gpt-4.1-mini",
        "api_key_var": "OPENAI_API_KEY",
        "api_base_url": "https://api.openai.com/v1",
        "header": None,
        "num_examples": 1,
        "rollouts_per_example": 1,
        "max_concurrent": 1,
        "max_concurrent_generation": None,
        "max_concurrent_scoring": None,
        "max_tokens": 42,
        "temperature": 0.9,
        "sampling_args": None,
        "verbose": False,
        "print_results": False,
        "no_interleave_scoring": False,
        "state_columns": [],
        "save_results": False,
        "save_every": -1,
        "save_to_hf_hub": False,
        "hf_hub_dataset_name": "",
    }
    base_args.update(overrides)
    args_namespace = SimpleNamespace(**base_args)

    captured: dict[str, dict] = {}

    monkeypatch.setattr(
        argparse.ArgumentParser,
        "parse_args",
        lambda self: args_namespace,
    )
    monkeypatch.setattr(vf_eval, "setup_logging", lambda *_, **__: None)
    monkeypatch.setattr(vf_eval, "load_endpoints", lambda *_: {})

    async def fake_run_evaluation(config):
        captured["sampling_args"] = dict(config.sampling_args)
        metadata = _make_metadata(config)
        return GenerateOutputs(
            prompt=[[{"role": "user", "content": "p"}]],
            completion=[[{"role": "assistant", "content": "c"}]],
            answer=[""],
            state=[
                {
                    "timing": {
                        "generation_ms": 0.0,
                        "scoring_ms": 0.0,
                        "total_ms": 0.0,
                    }
                }
            ],
            task=["default"],
            info=[{}],
            example_id=[0],
            reward=[1.0],
            metrics={},
            metadata=metadata,
        )

    monkeypatch.setattr(vf_eval, "run_evaluation", fake_run_evaluation)

    vf_eval.main()
    return captured


def test_cli_sampling_args_precedence_over_flags(monkeypatch):
    captured = _run_cli(
        monkeypatch,
        {
            "sampling_args": {
                "enable_thinking": False,
                "max_tokens": 77,
                "temperature": 0.1,
            },
        },
    )

    sa = captured["sampling_args"]
    assert sa["max_tokens"] == 77
    assert sa["temperature"] == 0.1
    assert sa["enable_thinking"] is False


def test_cli_sampling_args_fill_from_flags_when_missing(monkeypatch):
    captured = _run_cli(
        monkeypatch,
        {
            "sampling_args": {"enable_thinking": True},
            "max_tokens": 55,
            "temperature": 0.8,
        },
    )

    sa = captured["sampling_args"]
    assert sa["max_tokens"] == 55
    assert sa["temperature"] == 0.8
    assert sa["enable_thinking"] is True
