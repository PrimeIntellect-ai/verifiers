"""Tests for JudgeRubric team header propagation."""

import os
from unittest.mock import patch

from openai import AsyncOpenAI

from verifiers.rubrics.judge_rubric import JudgeRubric
from verifiers.utils.client_utils import build_prime_headers

CLEAN_ENV = {
    k: v for k, v in os.environ.items() if k not in ("PRIME_API_KEY", "PRIME_TEAM_ID")
}


class TestBuildPrimeHeaders:
    """Test build_prime_headers — same logic as _build_headers_and_api_key but
    usable without a ClientConfig (for judge clients, GEPA, etc.)."""

    def test_non_prime_key_returns_empty_headers(self):
        headers, api_key = build_prime_headers("SOME_OTHER_KEY")
        assert headers == {}
        assert api_key is None

    @patch.dict(
        os.environ,
        {"PRIME_API_KEY": "pit_test123", "PRIME_TEAM_ID": "team_abc"},
        clear=False,
    )
    def test_prime_key_with_team_id(self):
        headers, api_key = build_prime_headers("PRIME_API_KEY")
        assert headers == {"X-Prime-Team-ID": "team_abc"}
        assert api_key == "pit_test123"

    def test_prime_key_without_team_id(self):
        env = {**CLEAN_ENV, "PRIME_API_KEY": "pit_test123"}
        with patch.dict(os.environ, env, clear=True):
            with patch(
                "verifiers.utils.client_utils.load_prime_config", return_value={}
            ):
                headers, api_key = build_prime_headers("PRIME_API_KEY")
                assert "X-Prime-Team-ID" not in headers
                assert api_key == "pit_test123"

    def test_prime_key_missing_falls_back_to_config(self):
        with patch.dict(
            os.environ, {**CLEAN_ENV, "PRIME_TEAM_ID": "team_abc"}, clear=True
        ):
            with patch(
                "verifiers.utils.client_utils.load_prime_config",
                return_value={"api_key": "pit_from_config"},
            ):
                headers, api_key = build_prime_headers("PRIME_API_KEY")
                assert api_key == "pit_from_config"
                assert headers == {"X-Prime-Team-ID": "team_abc"}

    def test_team_id_from_config_file(self):
        with patch.dict(
            os.environ, {**CLEAN_ENV, "PRIME_API_KEY": "pit_test"}, clear=True
        ):
            with patch(
                "verifiers.utils.client_utils.load_prime_config",
                return_value={"team_id": "team_from_file"},
            ):
                headers, _ = build_prime_headers("PRIME_API_KEY")
                assert headers == {"X-Prime-Team-ID": "team_from_file"}


class TestJudgeRubricTeamHeaders:
    """Test that JudgeRubric propagates team headers when no explicit client is given."""

    def test_default_client_gets_team_header(self):
        with patch.dict(
            os.environ,
            {**CLEAN_ENV, "PRIME_API_KEY": "pit_test", "PRIME_TEAM_ID": "team_xyz"},
            clear=True,
        ):
            with patch(
                "verifiers.utils.client_utils.load_prime_config", return_value={}
            ):
                rubric = JudgeRubric()
                assert (
                    rubric.judge_client._custom_headers.get("X-Prime-Team-ID")
                    == "team_xyz"
                )

    def test_explicit_client_not_overridden(self):
        custom_client = AsyncOpenAI(
            api_key="sk-custom", base_url="http://localhost:1234/v1"
        )
        rubric = JudgeRubric(judge_client=custom_client)
        assert rubric.judge_client is custom_client

    def test_no_team_id_no_header(self):
        with patch.dict(os.environ, {**CLEAN_ENV}, clear=True):
            with patch(
                "verifiers.utils.client_utils.load_prime_config", return_value={}
            ):
                rubric = JudgeRubric()
                assert rubric.judge_client is not None
                assert "X-Prime-Team-ID" not in (
                    rubric.judge_client._custom_headers or {}
                )
