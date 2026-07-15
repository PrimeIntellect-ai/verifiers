from unittest.mock import MagicMock, patch

from verifiers.gepa.adapter import make_reflection_lm
from verifiers.types import ClientConfig


@patch("verifiers.gepa.adapter.OpenAI")
def test_reflection_lm_uses_resolved_prime_credentials_and_headers(mock_openai, monkeypatch):
    monkeypatch.setenv("PRIME_API_KEY", "prime-key")
    monkeypatch.setenv("PRIME_TEAM_ID", "team-id")
    response = MagicMock()
    response.choices[0].message.content = "proposal"
    mock_openai.return_value.chat.completions.create.return_value = response

    reflect = make_reflection_lm(
        ClientConfig(
            api_key_var="PRIME_API_KEY",
            api_base_url="https://api.pinference.ai/api/v1",
            extra_headers={"X-Test": "yes"},
        ),
        "openai/gpt-5.5",
    )

    assert reflect("improve this") == "proposal"
    mock_openai.assert_called_once_with(
        api_key="prime-key",
        base_url="https://api.pinference.ai/api/v1",
        timeout=3600.0,
        max_retries=10,
        default_headers={"X-Test": "yes", "X-Prime-Team-ID": "team-id"},
    )
