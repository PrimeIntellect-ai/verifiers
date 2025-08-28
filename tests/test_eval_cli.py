import verifiers.scripts.eval as vf_eval


def _make_fake_env(captured):
    class FakeEnv:
        def evaluate(
            self,
            client,
            model,
            sampling_args=None,
            num_examples=-1,
            rollouts_per_example=1,
            **kwargs,
        ):
            captured["sampling_args"] = dict(sampling_args or {})

            class Result:
                prompt = ["p"]
                completion = ["c"]
                reward = [1.0]
                info = [{}]
                task = ["default"]
                answer = [""]
                metrics = {}

            return Result()

    return FakeEnv()


def test_cli_sampling_args_precedence_over_flags(monkeypatch):
    captured = {}

    # Patch environment loader to return our fake env
    monkeypatch.setattr(
        vf_eval.vf,
        "load_environment",
        lambda env_id, **env_args: _make_fake_env(captured),
    )

    # Patch OpenAI client used by the CLI to a simple dummy
    class DummyOpenAI:
        def __init__(self, api_key=None, base_url=None):
            self.api_key = api_key
            self.base_url = base_url

    monkeypatch.setattr(vf_eval, "OpenAI", DummyOpenAI)

    # Run evaluation with JSON sampling args overriding flags
    vf_eval.eval_environment(
        env="dummy-env",
        env_args={},
        env_dir_path="./environments",
        endpoints_path="./configs/endpoints.py",
        model="gpt-4.1-mini",
        api_key_var="OPENAI_API_KEY",
        api_base_url="https://api.openai.com/v1",
        num_examples=1,
        rollouts_per_example=1,
        max_concurrent_requests=1,
        max_tokens=42,
        temperature=0.9,
        sampling_args={
            "enable_thinking": False,
            "max_tokens": 77,
            "temperature": 0.1,
        },
        verbose=False,
        save_dataset=False,
        save_to_hf_hub=False,
        hf_hub_dataset_name="",
    )

    sa = captured["sampling_args"]
    assert sa["max_tokens"] == 77
    assert sa["temperature"] == 0.1
    assert sa["enable_thinking"] is False


def test_cli_sampling_args_fill_from_flags_when_missing(monkeypatch):
    captured = {}

    # Patch environment loader to return our fake env
    monkeypatch.setattr(
        vf_eval.vf,
        "load_environment",
        lambda env_id, **env_args: _make_fake_env(captured),
    )

    # Patch OpenAI client used by the CLI to a simple dummy
    class DummyOpenAI:
        def __init__(self, api_key=None, base_url=None):
            self.api_key = api_key
            self.base_url = base_url

    monkeypatch.setattr(vf_eval, "OpenAI", DummyOpenAI)

    # Run evaluation with JSON lacking max_tokens/temperature
    vf_eval.eval_environment(
        env="dummy-env",
        env_args={},
        env_dir_path="./environments",
        endpoints_path="./configs/endpoints.py",
        model="gpt-4.1-mini",
        api_key_var="OPENAI_API_KEY",
        api_base_url="https://api.openai.com/v1",
        num_examples=1,
        rollouts_per_example=1,
        max_concurrent_requests=1,
        max_tokens=55,
        temperature=0.8,
        sampling_args={
            "enable_thinking": True,
        },
        verbose=False,
        save_dataset=False,
        save_to_hf_hub=False,
        hf_hub_dataset_name="",
    )

    sa = captured["sampling_args"]
    assert sa["max_tokens"] == 55
    assert sa["temperature"] == 0.8
    assert sa["enable_thinking"] is True
