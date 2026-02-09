import pytest

import verifiers.cli.commands.eval as eval_command


def test_main_delegates_to_vf_eval_when_not_hosted(monkeypatch):
    captured: dict[str, list[str]] = {}

    def fake_run_vf_eval(argv: list[str]) -> None:
        captured["argv"] = argv

    monkeypatch.setattr(eval_command, "_run_vf_eval", fake_run_vf_eval)

    eval_command.main(["my-env", "-n", "4"])

    assert captured["argv"] == ["my-env", "-n", "4"]


def test_main_rejects_hosted_only_flags_without_hosted():
    with pytest.raises(SystemExit) as exc_info:
        eval_command.main(["my-env", "--follow"])

    assert exc_info.value.code == 2


def test_main_hosted_creates_expected_payload(monkeypatch):
    monkeypatch.setattr(
        eval_command,
        "_load_prime_config",
        lambda: {
            "base_url": "https://api.primeintellect.ai",
            "frontend_url": "https://app.primeintellect.ai",
            "api_key": "test-api-key",
        },
    )

    calls: list[dict[str, object]] = []

    def fake_request_json(
        method: str,
        base_url: str,
        endpoint: str,
        api_key: str,
        *,
        json_payload=None,
        timeout: float = 30.0,
    ):
        calls.append(
            {
                "method": method,
                "base_url": base_url,
                "endpoint": endpoint,
                "api_key": api_key,
                "payload": json_payload,
                "timeout": timeout,
            }
        )
        if endpoint.startswith("/environmentshub/"):
            return {"data": {"id": "env-123"}}
        if endpoint == "/hosted-evaluations":
            return {
                "evaluation_id": "eval-abc",
                "viewer_url": "https://viewer/eval-abc",
            }
        raise AssertionError(f"unexpected endpoint: {endpoint}")

    monkeypatch.setattr(eval_command, "_request_json", fake_request_json)

    eval_command.main(
        [
            "primeintellect/gsm8k",
            "--hosted",
            "-m",
            "openai/gpt-4.1-mini",
            "-n",
            "10",
            "-r",
            "2",
            "-a",
            '{"difficulty":"hard"}',
            "--timeout-minutes",
            "120",
            "--allow-sandbox-access",
            "--allow-instances-access",
            "--custom-secrets",
            '{"API_KEY":"secret"}',
            "--eval-name",
            "nightly-gsm8k",
        ]
    )

    assert len(calls) == 2
    assert calls[0]["endpoint"] == "/environmentshub/primeintellect/gsm8k/@latest"

    payload = calls[1]["payload"]
    assert payload == {
        "environment_ids": ["env-123"],
        "inference_model": "openai/gpt-4.1-mini",
        "eval_config": {
            "num_examples": 10,
            "rollouts_per_example": 2,
            "allow_sandbox_access": True,
            "allow_instances_access": True,
            "env_args": {"difficulty": "hard"},
            "timeout_minutes": 120,
            "custom_secrets": {"API_KEY": "secret"},
        },
        "name": "nightly-gsm8k",
    }


@pytest.mark.parametrize(
    ("display_header", "expected_version"),
    [
        ("primeintellect/wordle", "latest"),
        ("wordle (local - ahead of primeintellect/wordle)", "latest"),
        ("primeintellect/wordle@2.0.0", "2.0.0"),
    ],
)
def test_main_hosted_resolves_slug_from_display_header(
    monkeypatch, display_header: str, expected_version: str
):
    monkeypatch.setattr(
        eval_command,
        "_load_prime_config",
        lambda: {
            "base_url": "https://api.primeintellect.ai",
            "frontend_url": "https://app.primeintellect.ai",
            "api_key": "test-api-key",
        },
    )

    requested_endpoints: list[str] = []

    def fake_request_json(
        method: str,
        base_url: str,
        endpoint: str,
        api_key: str,
        *,
        json_payload=None,
        timeout: float = 30.0,
    ):
        requested_endpoints.append(endpoint)
        if endpoint.startswith("/environmentshub/"):
            return {"data": {"id": "env-456"}}
        if endpoint == "/hosted-evaluations":
            return {"evaluation_id": "eval-456"}
        raise AssertionError(f"unexpected endpoint: {endpoint}")

    monkeypatch.setattr(eval_command, "_request_json", fake_request_json)

    eval_command.main(
        [
            "wordle",
            "--hosted",
            "--header",
            f"X-Prime-Eval-Env-Display: {display_header}",
        ]
    )

    assert (
        requested_endpoints[0]
        == f"/environmentshub/primeintellect/wordle/@{expected_version}"
    )


def test_main_hosted_supports_toml_config(monkeypatch, tmp_path):
    config_path = tmp_path / "evals.toml"
    config_path.write_text("[[eval]]\nenv_id='placeholder'\n", encoding="utf-8")

    monkeypatch.setattr(
        eval_command,
        "_load_prime_config",
        lambda: {
            "base_url": "https://api.primeintellect.ai",
            "frontend_url": "https://app.primeintellect.ai",
            "api_key": "test-api-key",
        },
    )
    monkeypatch.setattr(
        eval_command,
        "load_toml_config",
        lambda _path: [
            {
                "env_id": "primeintellect/gsm8k",
                "model": "openai/gpt-4.1-mini",
                "num_examples": 10,
                "rollouts_per_example": 2,
                "env_args": {"difficulty": "hard"},
            },
            {
                "env_id": "primeintellect/wordle@2.0.0",
            },
        ],
    )
    monkeypatch.setattr(
        eval_command.vf_eval,
        "get_env_eval_defaults",
        lambda env_id: (
            {"num_examples": 11, "rollouts_per_example": 5}
            if env_id == "primeintellect/wordle@2.0.0"
            else {}
        ),
    )

    calls: list[dict[str, object]] = []

    def fake_request_json(
        method: str,
        base_url: str,
        endpoint: str,
        api_key: str,
        *,
        json_payload=None,
        timeout: float = 30.0,
    ):
        calls.append(
            {
                "method": method,
                "base_url": base_url,
                "endpoint": endpoint,
                "api_key": api_key,
                "payload": json_payload,
                "timeout": timeout,
            }
        )
        if endpoint == "/environmentshub/primeintellect/gsm8k/@latest":
            return {"data": {"id": "env-1"}}
        if endpoint == "/environmentshub/primeintellect/wordle/@2.0.0":
            return {"data": {"id": "env-2"}}
        if endpoint == "/hosted-evaluations":
            created_count = len([call for call in calls if call["method"] == "POST"])
            return {"evaluation_id": f"eval-{created_count}"}
        raise AssertionError(f"unexpected endpoint: {endpoint}")

    monkeypatch.setattr(eval_command, "_request_json", fake_request_json)

    eval_command.main(
        [
            str(config_path),
            "--hosted",
            "--custom-secrets",
            '{"API_KEY":"secret"}',
        ]
    )

    post_payloads = [call["payload"] for call in calls if call["method"] == "POST"]
    assert len(post_payloads) == 2
    assert calls[0]["endpoint"] == "/environmentshub/primeintellect/gsm8k/@latest"
    assert calls[2]["endpoint"] == "/environmentshub/primeintellect/wordle/@2.0.0"
    assert post_payloads[0] == {
        "environment_ids": ["env-1"],
        "inference_model": "openai/gpt-4.1-mini",
        "eval_config": {
            "num_examples": 10,
            "rollouts_per_example": 2,
            "allow_sandbox_access": False,
            "allow_instances_access": False,
            "env_args": {"difficulty": "hard"},
            "custom_secrets": {"API_KEY": "secret"},
        },
    }
    assert post_payloads[1] == {
        "environment_ids": ["env-2"],
        "inference_model": eval_command.vf_eval.DEFAULT_MODEL,
        "eval_config": {
            "num_examples": 11,
            "rollouts_per_example": 5,
            "allow_sandbox_access": False,
            "allow_instances_access": False,
            "custom_secrets": {"API_KEY": "secret"},
        },
    }
