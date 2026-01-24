"""Tests for BrowserEnv integration (browser_env, dom_mode, cua_mode).

These tests verify the BrowserEnv class and its mode implementations
without requiring external services (Browserbase, CUA server).
"""

import os
import pytest
from unittest.mock import MagicMock, patch
from datasets import Dataset


# ============================================================================
# BrowserEnv Validation Tests
# ============================================================================


class TestBrowserEnvValidation:
    """Tests for environment variable validation in BrowserEnv."""

    def test_dom_mode_missing_browserbase_api_key_raises(self):
        """Test that DOM mode raises when BROWSERBASE_API_KEY is missing."""
        from verifiers.envs.integrations.browser_env.browser_env import BrowserEnv

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="BROWSERBASE_API_KEY"):
                BrowserEnv(
                    mode="dom",
                    dataset=Dataset.from_dict(
                        {"question": ["test"], "answer": ["test"]}
                    ),
                )

    def test_dom_mode_missing_browserbase_project_id_raises(self):
        """Test that DOM mode raises when BROWSERBASE_PROJECT_ID is missing."""
        from verifiers.envs.integrations.browser_env.browser_env import BrowserEnv

        with patch.dict(os.environ, {"BROWSERBASE_API_KEY": "test-key"}, clear=True):
            with pytest.raises(ValueError, match="BROWSERBASE_PROJECT_ID"):
                BrowserEnv(
                    mode="dom",
                    dataset=Dataset.from_dict(
                        {"question": ["test"], "answer": ["test"]}
                    ),
                )

    def test_dom_mode_missing_model_api_key_raises(self):
        """Test that DOM mode raises when MODEL_API_KEY is missing."""
        from verifiers.envs.integrations.browser_env.browser_env import BrowserEnv

        with patch.dict(
            os.environ,
            {"BROWSERBASE_API_KEY": "test-key", "BROWSERBASE_PROJECT_ID": "test-proj"},
            clear=True,
        ):
            with pytest.raises(ValueError, match="MODEL_API_KEY"):
                BrowserEnv(
                    mode="dom",
                    dataset=Dataset.from_dict(
                        {"question": ["test"], "answer": ["test"]}
                    ),
                )

    def test_cua_local_mode_no_browserbase_required(self):
        """Test that CUA LOCAL mode doesn't require Browserbase credentials."""
        from verifiers.envs.integrations.browser_env.browser_env import BrowserEnv

        # CUA LOCAL mode should not raise for missing Browserbase keys
        # but will fail at server health check - we just test validation passes
        with patch.dict(os.environ, {}, clear=True):
            # Mock the server health check to avoid network call
            with patch(
                "verifiers.envs.integrations.browser_env.modes.cua_mode.CUAMode.verify_server_connection"
            ):
                # This should NOT raise ValueError about missing env vars
                # use_sandbox=False to use manual CUA mode
                try:
                    env = BrowserEnv(
                        mode="cua",
                        env="LOCAL",
                        use_sandbox=False,
                        dataset=Dataset.from_dict(
                            {"question": ["test"], "answer": ["test"]}
                        ),
                    )
                    assert env.mode == "cua"
                except ValueError as e:
                    # Should not be about missing env vars for LOCAL mode
                    assert "BROWSERBASE" not in str(e)

    def test_cua_browserbase_mode_requires_credentials(self):
        """Test that CUA BROWSERBASE mode requires Browserbase credentials."""
        from verifiers.envs.integrations.browser_env.browser_env import BrowserEnv

        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "verifiers.envs.integrations.browser_env.modes.cua_mode.CUAMode.verify_server_connection"
            ):
                # Both sandbox and manual mode require credentials for BROWSERBASE
                with pytest.raises(ValueError, match="BROWSERBASE_API_KEY"):
                    BrowserEnv(
                        mode="cua",
                        env="BROWSERBASE",
                        use_sandbox=False,
                        dataset=Dataset.from_dict(
                            {"question": ["test"], "answer": ["test"]}
                        ),
                    )

    def test_invalid_mode_raises(self):
        """Test that an invalid mode raises ValueError."""
        from verifiers.envs.integrations.browser_env.browser_env import BrowserEnv

        with patch.dict(
            os.environ,
            {
                "BROWSERBASE_API_KEY": "test",
                "BROWSERBASE_PROJECT_ID": "test",
                "MODEL_API_KEY": "test",
            },
            clear=True,
        ):
            with pytest.raises(ValueError, match="Unknown mode"):
                BrowserEnv(
                    mode="invalid",
                    dataset=Dataset.from_dict(
                        {"question": ["test"], "answer": ["test"]}
                    ),
                )


class TestBrowserEnvSystemPrompt:
    """Tests for default system prompt selection in BrowserEnv."""

    def test_default_system_prompt_dom(self):
        """Test that DOM mode uses DOM_DEFAULT_PROMPT by default."""
        from verifiers.envs.integrations.browser_env.browser_env import (
            BrowserEnv,
            DOM_DEFAULT_PROMPT,
        )

        with patch.dict(
            os.environ,
            {
                "BROWSERBASE_API_KEY": "test",
                "BROWSERBASE_PROJECT_ID": "test",
                "MODEL_API_KEY": "test",
            },
            clear=True,
        ):
            env = BrowserEnv(
                mode="dom",
                dataset=Dataset.from_dict({"question": ["test"], "answer": ["test"]}),
            )
            assert env.system_prompt == DOM_DEFAULT_PROMPT

    def test_default_system_prompt_cua(self):
        """Test that CUA mode uses CUA_DEFAULT_PROMPT by default."""
        from verifiers.envs.integrations.browser_env.browser_env import (
            BrowserEnv,
            CUA_DEFAULT_PROMPT,
        )

        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "verifiers.envs.integrations.browser_env.modes.cua_mode.CUAMode.verify_server_connection"
            ):
                env = BrowserEnv(
                    mode="cua",
                    env="LOCAL",
                    use_sandbox=False,
                    dataset=Dataset.from_dict(
                        {"question": ["test"], "answer": ["test"]}
                    ),
                )
                assert env.system_prompt == CUA_DEFAULT_PROMPT

    def test_custom_system_prompt_preserved(self):
        """Test that custom system prompt overrides default."""
        from verifiers.envs.integrations.browser_env.browser_env import BrowserEnv

        custom_prompt = "You are a custom browser agent."

        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "verifiers.envs.integrations.browser_env.modes.cua_mode.CUAMode.verify_server_connection"
            ):
                env = BrowserEnv(
                    mode="cua",
                    env="LOCAL",
                    use_sandbox=False,
                    dataset=Dataset.from_dict(
                        {"question": ["test"], "answer": ["test"]}
                    ),
                    system_prompt=custom_prompt,
                )
                assert env.system_prompt == custom_prompt


# ============================================================================
# CUASandboxMode Tests
# ============================================================================


class TestCUASandboxModeInit:
    """Tests for CUASandboxMode initialization."""

    def test_sandbox_mode_requires_prime_sandboxes(self):
        """Test that CUASandboxMode requires prime-sandboxes package."""
        from verifiers.envs.integrations.browser_env.modes.cua_sandbox_mode import (
            SANDBOX_AVAILABLE,
        )

        # This test just verifies the import guard exists
        assert isinstance(SANDBOX_AVAILABLE, bool)

    def test_use_sandbox_true_creates_sandbox_mode(self):
        """Test that use_sandbox=True creates CUASandboxMode."""
        from verifiers.envs.integrations.browser_env.browser_env import BrowserEnv
        from verifiers.envs.integrations.browser_env.modes.cua_sandbox_mode import (
            CUASandboxMode,
        )

        with patch.dict(
            os.environ,
            {"BROWSERBASE_API_KEY": "test", "BROWSERBASE_PROJECT_ID": "test"},
            clear=True,
        ):
            env = BrowserEnv(
                mode="cua",
                use_sandbox=True,
                env="BROWSERBASE",
                dataset=Dataset.from_dict({"question": ["test"], "answer": ["test"]}),
            )
            assert isinstance(env._mode_impl, CUASandboxMode)

    def test_use_sandbox_false_creates_cua_mode(self):
        """Test that use_sandbox=False creates regular CUAMode."""
        from verifiers.envs.integrations.browser_env.browser_env import BrowserEnv
        from verifiers.envs.integrations.browser_env.modes.cua_mode import CUAMode
        from verifiers.envs.integrations.browser_env.modes.cua_sandbox_mode import (
            CUASandboxMode,
        )

        with patch.dict(os.environ, {}, clear=True):
            with patch(
                "verifiers.envs.integrations.browser_env.modes.cua_mode.CUAMode.verify_server_connection"
            ):
                env = BrowserEnv(
                    mode="cua",
                    use_sandbox=False,
                    env="LOCAL",
                    dataset=Dataset.from_dict(
                        {"question": ["test"], "answer": ["test"]}
                    ),
                )
                assert isinstance(env._mode_impl, CUAMode)
                assert not isinstance(env._mode_impl, CUASandboxMode)


class TestCUASandboxModeScreenshotFilter:
    """Tests for screenshot filtering in CUASandboxMode."""

    def test_filter_screenshots_keeps_recent(self):
        """Test that filter keeps the N most recent screenshots."""
        from verifiers.envs.integrations.browser_env.modes.cua_sandbox_mode import (
            CUASandboxMode,
            SANDBOX_AVAILABLE,
        )

        if not SANDBOX_AVAILABLE:
            pytest.skip("prime-sandboxes not installed")

        mode = CUASandboxMode(keep_recent_screenshots=2)

        messages = [
            {"role": "user", "content": [{"type": "text", "text": "msg1"}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "msg2"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,img1"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "msg3"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,img2"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "msg4"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,img3"},
                    },
                ],
            },
        ]

        filtered = mode.filter_screenshots_in_messages(messages)

        # First screenshot (msg2) should be replaced with placeholder
        assert filtered[1]["content"][1]["type"] == "text"
        assert "removed" in filtered[1]["content"][1]["text"].lower()

        # Last two screenshots (msg3, msg4) should be preserved
        assert filtered[2]["content"][1]["type"] == "image_url"
        assert filtered[3]["content"][1]["type"] == "image_url"


# ============================================================================
# CUAMode Tests
# ============================================================================


class TestCUAModeScreenshotFilter:
    """Tests for screenshot filtering in CUAMode."""

    def test_filter_screenshots_keeps_recent(self):
        """Test that filter keeps the N most recent screenshots."""
        from verifiers.envs.integrations.browser_env.modes.cua_mode import CUAMode

        mode = CUAMode(keep_recent_screenshots=2)

        messages = [
            {"role": "user", "content": [{"type": "text", "text": "msg1"}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "msg2"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,img1"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "msg3"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,img2"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "msg4"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,img3"},
                    },
                ],
            },
        ]

        filtered = mode.filter_screenshots_in_messages(messages)

        # First screenshot (msg2) should be replaced with placeholder
        assert filtered[1]["content"][1]["type"] == "text"
        assert "removed" in filtered[1]["content"][1]["text"].lower()

        # Last two screenshots (msg3, msg4) should be preserved
        assert filtered[2]["content"][1]["type"] == "image_url"
        assert filtered[3]["content"][1]["type"] == "image_url"

    def test_filter_screenshots_none_keeps_all(self):
        """Test that keep_recent_screenshots=None keeps all screenshots."""
        from verifiers.envs.integrations.browser_env.modes.cua_mode import CUAMode

        mode = CUAMode(keep_recent_screenshots=None)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,img1"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,img2"},
                    },
                ],
            },
        ]

        filtered = mode.filter_screenshots_in_messages(messages)

        # All screenshots should be preserved
        assert filtered[0]["content"][0]["type"] == "image_url"
        assert filtered[1]["content"][0]["type"] == "image_url"

    def test_filter_screenshots_fewer_than_limit(self):
        """Test that filtering doesn't change messages when fewer than N screenshots."""
        from verifiers.envs.integrations.browser_env.modes.cua_mode import CUAMode

        mode = CUAMode(keep_recent_screenshots=5)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,img1"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,img2"},
                    },
                ],
            },
        ]

        filtered = mode.filter_screenshots_in_messages(messages)

        # Both screenshots should be preserved (2 < 5)
        assert filtered[0]["content"][0]["type"] == "image_url"
        assert filtered[1]["content"][0]["type"] == "image_url"


class TestCUAModeResponseFormat:
    """Tests for response formatting in CUAMode."""

    def test_format_response_success(self):
        """Test formatting a successful response."""
        from verifiers.envs.integrations.browser_env.modes.cua_mode import CUAMode

        mode = CUAMode(save_screenshots=False)

        response = {
            "success": True,
            "state": {
                "url": "https://example.com",
                "viewport": {"width": 1024, "height": 768},
            },
        }

        formatted = mode._format_response(response)

        assert len(formatted) == 1
        assert formatted[0]["type"] == "text"
        assert "Success" in formatted[0]["text"]
        assert "https://example.com" in formatted[0]["text"]

    def test_format_response_failure(self):
        """Test formatting a failed response with error."""
        from verifiers.envs.integrations.browser_env.modes.cua_mode import CUAMode

        mode = CUAMode(save_screenshots=False)

        response = {
            "success": False,
            "error": "Element not found",
            "state": {"url": "https://example.com", "viewport": {}},
        }

        formatted = mode._format_response(response)

        assert len(formatted) == 1
        assert formatted[0]["type"] == "text"
        assert "Failed" in formatted[0]["text"]
        assert "Element not found" in formatted[0]["text"]

    def test_format_response_with_screenshot(self):
        """Test formatting includes image_url when screenshot present."""
        from verifiers.envs.integrations.browser_env.modes.cua_mode import CUAMode

        mode = CUAMode(save_screenshots=False)

        response = {
            "success": True,
            "state": {
                "url": "https://example.com",
                "screenshot": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                "viewport": {"width": 1024, "height": 768},
            },
        }

        formatted = mode._format_response(response)

        assert len(formatted) == 2
        assert formatted[0]["type"] == "text"
        assert formatted[1]["type"] == "image_url"
        assert "data:image/png;base64," in formatted[1]["image_url"]["url"]

    def test_format_response_no_screenshot(self):
        """Test formatting handles missing screenshot gracefully."""
        from verifiers.envs.integrations.browser_env.modes.cua_mode import CUAMode

        mode = CUAMode(save_screenshots=False)

        response = {
            "success": True,
            "state": {"url": "https://example.com", "viewport": {}},
        }

        formatted = mode._format_response(response)

        assert len(formatted) == 1
        assert formatted[0]["type"] == "text"


# ============================================================================
# DOMMode Tests
# ============================================================================


class TestDOMModeLLMConfig:
    """Tests for LLM config extraction in DOMMode."""

    def test_get_llm_config_basic(self):
        """Test basic LLM config extraction."""
        from verifiers.envs.integrations.browser_env.modes.dom_mode import DOMMode

        mode = DOMMode()

        mock_client = MagicMock()
        mock_client.base_url = "https://api.openai.com/v1"
        mock_client.api_key = "test-key"

        state = {"client": mock_client, "model": "gpt-4o"}

        config = mode._get_llm_config(state)

        assert config is not None
        assert config["modelName"] == "gpt-4o"
        # OpenAI URL should not be included (default)
        assert "baseURL" not in config

    def test_get_llm_config_non_openai(self):
        """Test LLM config includes baseURL for non-OpenAI endpoints."""
        from verifiers.envs.integrations.browser_env.modes.dom_mode import DOMMode

        mode = DOMMode()

        mock_client = MagicMock()
        mock_client.base_url = "https://custom-api.example.com/v1"
        mock_client.api_key = "test-key"

        state = {"client": mock_client, "model": "custom-model"}

        config = mode._get_llm_config(state)

        assert config is not None
        assert config["modelName"] == "custom-model"
        assert config["baseURL"] == "https://custom-api.example.com/v1"
        assert config["apiKey"] == "test-key"

    def test_get_llm_config_no_client(self):
        """Test LLM config returns None when no client available."""
        from verifiers.envs.integrations.browser_env.modes.dom_mode import DOMMode

        mode = DOMMode()
        state = {}

        config = mode._get_llm_config(state)

        assert config is None

    def test_get_llm_config_no_model(self):
        """Test LLM config returns None when no model specified."""
        from verifiers.envs.integrations.browser_env.modes.dom_mode import DOMMode

        mode = DOMMode()

        mock_client = MagicMock()
        state = {"client": mock_client}

        config = mode._get_llm_config(state)

        assert config is None


# ============================================================================
# Example Environment Tests
# ============================================================================


class TestExampleDatasets:
    """Tests for example environment datasets."""

    def test_dom_example_dataset_structure(self):
        """Test DOM example dataset has correct structure."""
        from environments.browser_dom_example.browser_dom_example import (
            create_example_dataset,
        )

        dataset = create_example_dataset()

        assert "question" in dataset.column_names
        assert "answer" in dataset.column_names
        assert "start_url" in dataset.column_names
        assert "task_id" in dataset.column_names
        assert len(dataset) >= 1

    def test_cua_example_dataset_structure(self):
        """Test CUA example dataset has correct structure."""
        from environments.browser_cua_example.browser_cua_example import (
            create_example_dataset,
        )

        dataset = create_example_dataset()

        assert "question" in dataset.column_names
        assert "answer" in dataset.column_names
        assert "start_url" in dataset.column_names
        assert "task_id" in dataset.column_names
        assert len(dataset) >= 1


class TestJudgeAnswer:
    """Tests for judge_answer reward function."""

    @pytest.mark.asyncio
    async def test_judge_answer_returns_1_for_yes(self):
        """Test that judge_answer returns 1.0 when judge says yes."""
        from environments.browser_dom_example.browser_dom_example import judge_answer

        async def mock_judge(prompt, completion, answer, state):
            return "yes, the answer is correct"

        result = await judge_answer(
            judge=mock_judge,
            prompt="What is 2+2?",
            completion=[{"role": "assistant", "content": "The answer is 4"}],
            answer="4",
            state={},
        )
        assert result == 1.0

    @pytest.mark.asyncio
    async def test_judge_answer_returns_0_for_no(self):
        """Test that judge_answer returns 0.0 when judge says no."""
        from environments.browser_dom_example.browser_dom_example import judge_answer

        async def mock_judge(prompt, completion, answer, state):
            return "no, the answer is incorrect"

        result = await judge_answer(
            judge=mock_judge,
            prompt="What is 2+2?",
            completion=[{"role": "assistant", "content": "The answer is 5"}],
            answer="4",
            state={},
        )
        assert result == 0.0

    @pytest.mark.asyncio
    async def test_judge_answer_case_insensitive(self):
        """Test that judge response check is case insensitive."""
        from environments.browser_dom_example.browser_dom_example import judge_answer

        async def mock_judge(prompt, completion, answer, state):
            return "YES"

        result = await judge_answer(
            judge=mock_judge,
            prompt="test",
            completion="test",
            answer="test",
            state={},
        )
        assert result == 1.0


# ============================================================================
# Constants Tests
# ============================================================================


class TestBrowserEnvConstants:
    """Tests for browser environment constants."""

    def test_dom_default_prompt_exists(self):
        """Test that DOM_DEFAULT_PROMPT is defined and non-empty."""
        from verifiers.envs.integrations.browser_env.browser_env import (
            DOM_DEFAULT_PROMPT,
        )

        assert DOM_DEFAULT_PROMPT
        assert len(DOM_DEFAULT_PROMPT) > 50
        assert "Stagehand" in DOM_DEFAULT_PROMPT

    def test_cua_default_prompt_exists(self):
        """Test that CUA_DEFAULT_PROMPT is defined and non-empty."""
        from verifiers.envs.integrations.browser_env.browser_env import (
            CUA_DEFAULT_PROMPT,
        )

        assert CUA_DEFAULT_PROMPT
        assert len(CUA_DEFAULT_PROMPT) > 50
        assert "click" in CUA_DEFAULT_PROMPT

    def test_mode_type_literal(self):
        """Test that ModeType includes expected values."""
        from verifiers.envs.integrations.browser_env.browser_env import ModeType
        from typing import get_args

        args = get_args(ModeType)
        assert "dom" in args
        assert "cua" in args
