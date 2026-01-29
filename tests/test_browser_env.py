"""Tests for BrowserEnv integration (browser_env, dom_mode, cua_mode).

These tests verify the BrowserEnv class and its mode implementations
without requiring external services (Browserbase, CUA server).
"""

import os
import pytest
from unittest.mock import MagicMock, patch
from datasets import Dataset

# Skip all tests in this module if browser dependencies are not installed
pytest.importorskip("stagehand", reason="verifiers[browser] extra not installed")


# ============================================================================
# BrowserEnv Validation Tests
# ============================================================================


class TestBrowserEnvValidation:
    """Tests for BrowserEnv validation."""

    def test_invalid_mode_raises(self):
        """Test that an invalid mode raises ValueError."""
        from verifiers.envs.integrations.browser_env.browser_env import BrowserEnv

        with patch.dict(
            os.environ,
            {"BROWSERBASE_API_KEY": "test", "MODEL_API_KEY": "test"},
            clear=True,
        ):
            with pytest.raises(ValueError, match="Unknown mode"):
                BrowserEnv(
                    mode="invalid",
                    project_id="test",
                    dataset=Dataset.from_dict(
                        {"question": ["test"], "answer": ["test"]}
                    ),
                )


# ============================================================================
# CUAMode Tests (Unified)
# ============================================================================


class TestCUAModeInit:
    """Tests for CUAMode initialization with execution_mode parameter."""

    def test_sandbox_mode_requires_prime_sandboxes(self):
        """Test that CUAMode sandbox mode requires prime-sandboxes package."""
        from verifiers.envs.integrations.browser_env.modes.cua_mode import (
            SANDBOX_AVAILABLE,
        )

        # This test just verifies the import guard exists
        assert isinstance(SANDBOX_AVAILABLE, bool)

    def test_use_sandbox_true_creates_sandbox_execution_mode(self):
        """Test that use_sandbox=True creates CUAMode with execution_mode='sandbox'."""
        from verifiers.envs.integrations.browser_env.browser_env import BrowserEnv
        from verifiers.envs.integrations.browser_env.modes.cua_mode import CUAMode

        with patch.dict(
            os.environ,
            {"BROWSERBASE_API_KEY": "test"},
            clear=True,
        ):
            env = BrowserEnv(
                mode="cua",
                project_id="test",
                use_sandbox=True,
                env="BROWSERBASE",
                dataset=Dataset.from_dict({"question": ["test"], "answer": ["test"]}),
            )
            assert isinstance(env._mode_impl, CUAMode)
            assert env._mode_impl._execution_mode == "sandbox"

    def test_use_sandbox_false_creates_local_execution_mode(self):
        """Test that use_sandbox=False creates CUAMode with execution_mode='local'."""
        from verifiers.envs.integrations.browser_env.browser_env import BrowserEnv
        from verifiers.envs.integrations.browser_env.modes.cua_mode import CUAMode

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
                assert env._mode_impl._execution_mode == "local"


class TestCUASandboxModeBackwardsCompat:
    """Tests for backwards compatibility with CUASandboxMode."""

    def test_deprecated_cua_sandbox_mode_import(self):
        """Test that CUASandboxMode import still works with deprecation warning."""
        import warnings
        from verifiers.envs.integrations.browser_env.modes import CUASandboxMode
        from verifiers.envs.integrations.browser_env.modes.cua_mode import (
            SANDBOX_AVAILABLE,
        )

        if not SANDBOX_AVAILABLE:
            pytest.skip("prime-sandboxes not installed")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mode = CUASandboxMode(keep_recent_screenshots=2)
            # Check that deprecation warning was issued
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()
            # Check that it's actually a CUAMode with sandbox execution
            assert mode._execution_mode == "sandbox"


class TestCUAModeScreenshotFilter:
    """Tests for screenshot filtering in CUAMode."""

    def test_filter_screenshots_keeps_recent(self):
        """Test that filter keeps the N most recent screenshots."""
        from verifiers.envs.integrations.browser_env.modes.cua_mode import CUAMode

        mode = CUAMode(execution_mode="local", keep_recent_screenshots=2)

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

    def test_filter_screenshots_sandbox_mode(self):
        """Test that filter works in sandbox mode too."""
        from verifiers.envs.integrations.browser_env.modes.cua_mode import (
            CUAMode,
            SANDBOX_AVAILABLE,
        )

        if not SANDBOX_AVAILABLE:
            pytest.skip("prime-sandboxes not installed")

        mode = CUAMode(execution_mode="sandbox", keep_recent_screenshots=2)

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

        mode = CUAMode(execution_mode="local", keep_recent_screenshots=None)

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

        mode = CUAMode(execution_mode="local", keep_recent_screenshots=5)

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

        mode = CUAMode(execution_mode="local", save_screenshots=False)

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

        mode = CUAMode(execution_mode="local", save_screenshots=False)

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

        mode = CUAMode(execution_mode="local", save_screenshots=False)

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

        mode = CUAMode(execution_mode="local", save_screenshots=False)

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

    def test_mode_type_literal(self):
        """Test that ModeType includes expected values."""
        from verifiers.envs.integrations.browser_env.browser_env import ModeType
        from typing import get_args

        args = get_args(ModeType)
        assert "dom" in args
        assert "cua" in args
