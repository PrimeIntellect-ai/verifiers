from verifiers.scripts.eval import build_eval_config


def test_resolved_model_lands_in_extra_env_kwargs(monkeypatch):
    raw = {
        "env_id": "math-python",
        "model": "openai/gpt-4.1-mini",
        "api_base_url": "https://example.test/v1",
        "api_key_var": "OPENAI_API_KEY",
    }
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    cfg = build_eval_config(raw)

    assert cfg.model == "openai/gpt-4.1-mini"
    assert cfg.extra_env_kwargs.get("model") == "openai/gpt-4.1-mini"


def test_env_args_model_overrides_for_env_but_not_client(monkeypatch):
    raw = {
        "env_id": "math-python",
        "model": "openai/gpt-4.1-mini",
        "env_args": {"model": "qwen/qwen3-14b"},
        "api_base_url": "https://example.test/v1",
        "api_key_var": "OPENAI_API_KEY",
    }
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    cfg = build_eval_config(raw)

    assert cfg.model == "openai/gpt-4.1-mini"
    assert cfg.env_args.get("model") == "qwen/qwen3-14b"
    assert cfg.extra_env_kwargs.get("model") is None
