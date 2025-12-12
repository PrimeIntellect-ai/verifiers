"""Tests for the RLMEnv class."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from datasets import Dataset

from verifiers.envs.rlm_env import RLMEnv
from verifiers.rubrics.rubric import Rubric


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_sandbox_client():
    """Create a mock AsyncSandboxClient."""
    client = MagicMock()
    client.create = AsyncMock(return_value=MagicMock(id="sandbox_123"))
    client.delete = AsyncMock()
    client.bulk_delete = AsyncMock()
    client.wait_for_creation = AsyncMock()
    client.execute_command = AsyncMock(return_value=MagicMock(stdout="", stderr=""))
    return client


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for RLMEnv."""
    return Dataset.from_dict(
        {
            "question": ["What is 2+2?"],
            "answer": ["4"],
            "info": [{}],
        }
    )


@pytest.fixture
def rlm_env(mock_sandbox_client, mock_dataset):
    """Create an RLMEnv instance with mocked dependencies."""
    with (
        patch("verifiers.envs.sandbox_env.AsyncSandboxClient") as mock_client_cls,
        patch("verifiers.envs.sandbox_env.CreateSandboxRequest"),
    ):
        mock_client_cls.return_value = mock_sandbox_client
        env = RLMEnv(
            dataset=mock_dataset,
            max_iterations=10,
            max_output_length=1000,
        )
        env.sandbox_client = mock_sandbox_client
        yield env
        # Clean up to prevent teardown logging errors
        env.active_sandboxes.clear()


@pytest.fixture
def rlm_env_with_sub_tools(mock_sandbox_client, mock_dataset):
    """Create an RLMEnv instance with sub_tools configured."""

    def sample_tool(x: int, y: int) -> int:
        """Add two numbers together."""
        return x + y

    def another_tool(text: str) -> str:
        """Reverse a string."""
        return text[::-1]

    with (
        patch("verifiers.envs.sandbox_env.AsyncSandboxClient") as mock_client_cls,
        patch("verifiers.envs.sandbox_env.CreateSandboxRequest"),
    ):
        mock_client_cls.return_value = mock_sandbox_client
        env = RLMEnv(
            dataset=mock_dataset,
            sub_tools=[sample_tool, another_tool],
            sub_tool_max_turns=3,
        )
        env.sandbox_client = mock_sandbox_client
        yield env
        # Clean up to prevent teardown logging errors
        env.active_sandboxes.clear()


# =============================================================================
# 1. Pure Utility Functions
# =============================================================================


class TestFormatExecutionOutput:
    """Tests for _format_execution_output method."""

    def test_format_with_stdout(self, rlm_env):
        """Format successful execution with stdout."""
        result = {
            "status": "ok",
            "stdout": "Hello, world!",
            "stderr": "",
            "result": None,
            "execution_count": 1,
        }
        output = rlm_env._format_execution_output(result)
        assert output == "Hello, world!"

    def test_format_with_stderr(self, rlm_env):
        """Format execution with stderr."""
        result = {
            "status": "ok",
            "stdout": "output",
            "stderr": "warning message",
            "result": None,
            "execution_count": 1,
        }
        output = rlm_env._format_execution_output(result)
        assert "output" in output
        assert "stderr:" in output
        assert "warning message" in output

    def test_format_with_result_value(self, rlm_env):
        """Format execution with result value."""
        result = {
            "status": "ok",
            "stdout": "",
            "stderr": "",
            "result": "42",
            "execution_count": 3,
        }
        output = rlm_env._format_execution_output(result)
        assert "Out[3]: 42" in output

    def test_format_error_status(self, rlm_env):
        """Format error status with traceback."""
        result = {
            "status": "error",
            "stdout": "",
            "stderr": "",
            "result": "Traceback (most recent call last):\n  NameError: name 'x' is not defined",
            "execution_count": 1,
        }
        output = rlm_env._format_execution_output(result)
        assert "Traceback" in output
        assert "NameError" in output

    def test_truncate_long_output(self, rlm_env):
        """Truncate output exceeding max_output_length."""
        long_output = "x" * 2000
        result = {
            "status": "ok",
            "stdout": long_output,
            "stderr": "",
            "result": None,
            "execution_count": 1,
        }
        output = rlm_env._format_execution_output(result)
        assert (
            len(output) <= rlm_env.max_output_length + 50
        )  # Allow for truncation message
        assert "[output truncated]" in output

    def test_empty_output(self, rlm_env):
        """Handle empty output."""
        result = {
            "status": "ok",
            "stdout": "",
            "stderr": "",
            "result": None,
            "execution_count": 1,
        }
        output = rlm_env._format_execution_output(result)
        assert output == "(no output)"


class TestBuildContextDict:
    """Tests for _build_context_dict method."""

    def test_build_context_with_string(self, rlm_env):
        """Build context dict with string data."""
        context = rlm_env._build_context_dict("hello world")
        assert context["input_data"] == "hello world"
        assert context["input_data_metadata"]["type"] == "<class 'str'>"
        assert context["input_data_metadata"]["size"] == 11

    def test_build_context_with_list(self, rlm_env):
        """Build context dict with list data."""
        context = rlm_env._build_context_dict([1, 2, 3, 4, 5])
        assert context["input_data"] == [1, 2, 3, 4, 5]
        assert context["input_data_metadata"]["type"] == "<class 'list'>"
        assert context["input_data_metadata"]["size"] == 5

    def test_build_context_with_dict(self, rlm_env):
        """Build context dict with dict data."""
        data = {"a": 1, "b": 2}
        context = rlm_env._build_context_dict(data)
        assert context["input_data"] == data
        assert context["input_data_metadata"]["type"] == "<class 'dict'>"
        assert context["input_data_metadata"]["size"] == 2

    def test_build_context_with_none(self, rlm_env):
        """Build context dict with None (no context)."""
        context = rlm_env._build_context_dict(None)
        assert context["input_data"] is None
        assert context["input_data_metadata"]["type"] == "<class 'NoneType'>"
        assert context["input_data_metadata"]["size"] == 0


class TestBuildContextMetadata:
    """Tests for _build_context_metadata method."""

    def test_metadata_for_string(self, rlm_env):
        """Metadata includes correct type and size for string."""
        metadata = rlm_env._build_context_metadata("test string")
        assert metadata["type"] == "<class 'str'>"
        assert metadata["size"] == 11

    def test_metadata_for_list(self, rlm_env):
        """Metadata includes correct type and size for list."""
        metadata = rlm_env._build_context_metadata([1, 2, 3])
        assert metadata["type"] == "<class 'list'>"
        assert metadata["size"] == 3

    def test_metadata_for_object_without_len(self, rlm_env):
        """Metadata handles objects without __len__."""
        metadata = rlm_env._build_context_metadata(42)
        assert metadata["type"] == "<class 'int'>"
        assert metadata["size"] == "unknown"


class TestGenerateSubToolsDocumentation:
    """Tests for _generate_sub_tools_documentation method."""

    def test_empty_when_no_sub_tools(self, rlm_env):
        """Generate empty string when no sub_tools."""
        docs = rlm_env._generate_sub_tools_documentation()
        assert docs == ""

    def test_generate_docs_for_tools(self, rlm_env_with_sub_tools):
        """Generate proper markdown documentation for tools."""
        docs = rlm_env_with_sub_tools._generate_sub_tools_documentation()
        assert "Sub-Agent Tools" in docs
        assert "sample_tool" in docs
        assert "another_tool" in docs
        assert "Add two numbers" in docs
        assert "Reverse a string" in docs

    def test_docs_include_parameters(self, rlm_env_with_sub_tools):
        """Documentation includes parameter information."""
        docs = rlm_env_with_sub_tools._generate_sub_tools_documentation()
        assert "Parameters" in docs
        assert "`x`" in docs or "x" in docs
        assert "`y`" in docs or "y" in docs


class TestExtractTunnelUrlFromLine:
    """Tests for extract_tunnel_url_from_line function."""

    def test_extract_valid_url(self):
        """Extract valid trycloudflare.com URL."""
        from verifiers.utils.tunnel import extract_tunnel_url_from_line

        line = (
            "2024-01-01 12:00:00 INF https://random-words.trycloudflare.com registered"
        )
        url = extract_tunnel_url_from_line(line)
        assert url == "https://random-words.trycloudflare.com"

    def test_return_none_for_no_url(self):
        """Return None for lines without tunnel URL."""
        from verifiers.utils.tunnel import extract_tunnel_url_from_line

        line = "Starting cloudflared tunnel..."
        url = extract_tunnel_url_from_line(line)
        assert url is None

    def test_handle_trailing_characters(self):
        """Handle URLs with trailing characters."""
        from verifiers.utils.tunnel import extract_tunnel_url_from_line

        line = "https://test-tunnel.trycloudflare.com/path?query=1 some text"
        url = extract_tunnel_url_from_line(line)
        assert url is not None
        assert url.startswith("https://")
        assert ".trycloudflare.com" in url

    def test_no_https_prefix(self):
        """Return None when line has domain but no https://."""
        from verifiers.utils.tunnel import extract_tunnel_url_from_line

        line = "something.trycloudflare.com without https"
        url = extract_tunnel_url_from_line(line)
        assert url is None


# =============================================================================
# 2. Initialization and Configuration
# =============================================================================


class TestRLMEnvInitialization:
    """Tests for RLMEnv initialization."""

    def test_default_initialization(self, mock_sandbox_client, mock_dataset):
        """Default initialization with minimal args."""
        with (
            patch("verifiers.envs.sandbox_env.AsyncSandboxClient") as mock_client_cls,
            patch("verifiers.envs.sandbox_env.CreateSandboxRequest"),
        ):
            mock_client_cls.return_value = mock_sandbox_client
            env = RLMEnv(dataset=mock_dataset)

            assert env.sub_model is None
            assert env.sub_tools == []
            assert env.max_iterations == 50
            assert env.max_output_length == 8192
            assert env.max_sub_llm_parallelism == 5
            assert env.context_key == "context"

    def test_custom_configuration(self, mock_sandbox_client, mock_dataset):
        """Custom sub_model, sub_tools, max_iterations, max_output_length."""

        def dummy_tool(x: int) -> int:
            return x * 2

        with (
            patch("verifiers.envs.sandbox_env.AsyncSandboxClient") as mock_client_cls,
            patch("verifiers.envs.sandbox_env.CreateSandboxRequest"),
        ):
            mock_client_cls.return_value = mock_sandbox_client
            env = RLMEnv(
                dataset=mock_dataset,
                sub_model="gpt-4",
                sub_tools=[dummy_tool],
                max_iterations=20,
                max_output_length=4096,
                max_sub_llm_parallelism=10,
                context_key="custom_context",
            )

            assert env.sub_model == "gpt-4"
            assert len(env.sub_tools) == 1
            assert env.max_iterations == 20
            assert env.max_output_length == 4096
            assert env.max_sub_llm_parallelism == 10
            assert env.context_key == "custom_context"

    def test_system_prompt_customization(self, mock_sandbox_client, mock_dataset):
        """System prompt customization."""
        custom_prompt = "You are a custom RLM assistant."
        with (
            patch("verifiers.envs.sandbox_env.AsyncSandboxClient") as mock_client_cls,
            patch("verifiers.envs.sandbox_env.CreateSandboxRequest"),
        ):
            mock_client_cls.return_value = mock_sandbox_client
            env = RLMEnv(
                dataset=mock_dataset,
                system_prompt=custom_prompt,
            )

            assert env.custom_system_prompt == custom_prompt

    def test_bash_tool_removed(self, rlm_env):
        """Verify bash tool is removed from parent class."""
        # RLMEnv should not have bash in its tool_map
        assert "bash" not in rlm_env.tool_map


# =============================================================================
# 3. State Management
# =============================================================================


class TestSetupState:
    """Tests for setup_state method."""

    @pytest.mark.asyncio
    async def test_creates_rollout_id(self, rlm_env):
        """Creates rollout_id and registers in active_rollouts."""
        # Mock the interception server and tunnel pool
        rlm_env._ensure_interception_server = AsyncMock()
        rlm_env._tunnel_pool.get_tunnel_url = AsyncMock(
            return_value="https://test.trycloudflare.com"
        )
        rlm_env._write_json_to_sandbox = AsyncMock()
        rlm_env._wait_for_worker_ready = AsyncMock()

        state = {
            "info": {},
            "model": "test-model",
            "client": MagicMock(),
        }

        result = await rlm_env.setup_state(state)

        assert "rollout_id" in result
        assert result["rollout_id"].startswith("rlm_")
        assert result["rollout_id"] in rlm_env.active_rollouts

    @pytest.mark.asyncio
    async def test_sets_up_interception_url(self, rlm_env):
        """Sets up interception_url in state."""
        rlm_env._ensure_interception_server = AsyncMock()
        rlm_env._tunnel_pool.get_tunnel_url = AsyncMock(
            return_value="https://test.trycloudflare.com"
        )
        rlm_env._write_json_to_sandbox = AsyncMock()
        rlm_env._wait_for_worker_ready = AsyncMock()

        state = {
            "info": {},
            "model": "test-model",
            "client": MagicMock(),
        }

        result = await rlm_env.setup_state(state)

        assert "interception_url" in result
        assert "trycloudflare.com" in result["interception_url"]

    @pytest.mark.asyncio
    async def test_stores_rlm_context(self, rlm_env):
        """Stores rlm_context in state."""
        rlm_env._ensure_interception_server = AsyncMock()
        rlm_env._tunnel_pool.get_tunnel_url = AsyncMock(
            return_value="https://test.trycloudflare.com"
        )
        rlm_env._write_json_to_sandbox = AsyncMock()
        rlm_env._wait_for_worker_ready = AsyncMock()

        context_data = {"key": "value"}
        state = {
            "info": {"context": context_data},
            "model": "test-model",
            "client": MagicMock(),
        }

        result = await rlm_env.setup_state(state)

        assert "rlm_context" in result
        assert result["rlm_context"]["input_data"] == context_data


class TestCleanupRLMState:
    """Tests for cleanup_rlm_state method."""

    @pytest.mark.asyncio
    async def test_removes_rollout_from_active(self, rlm_env):
        """Removes rollout from active_rollouts."""
        rollout_id = "rlm_test123"
        rlm_env.active_rollouts[rollout_id] = {"client": MagicMock()}

        state = {"rollout_id": rollout_id}
        await rlm_env.cleanup_rlm_state(state)

        assert rollout_id not in rlm_env.active_rollouts

    @pytest.mark.asyncio
    async def test_handles_missing_rollout_id(self, rlm_env):
        """Handles missing rollout_id gracefully."""
        state = {}  # No rollout_id
        # Should not raise
        await rlm_env.cleanup_rlm_state(state)

    @pytest.mark.asyncio
    async def test_handles_unknown_rollout_id(self, rlm_env):
        """Handles unknown rollout_id gracefully."""
        state = {"rollout_id": "nonexistent"}
        # Should not raise
        await rlm_env.cleanup_rlm_state(state)


# =============================================================================
# 4. Environment Response Flow
# =============================================================================


class TestGetPromptMessages:
    """Tests for get_prompt_messages method."""

    @pytest.mark.asyncio
    async def test_adds_system_prompt_on_first_turn(self, rlm_env):
        """Adds system prompt on first turn (empty trajectory)."""
        state = {
            "trajectory": [],
            "prompt": [{"role": "user", "content": "What is 2+2?"}],
        }

        messages = await rlm_env.get_prompt_messages(state)

        assert messages[0]["role"] == "system"
        assert "RLM" in messages[0]["content"] or "REPL" in messages[0]["content"]

    @pytest.mark.asyncio
    async def test_appends_sub_tools_docs(self, rlm_env_with_sub_tools):
        """Appends sub-tools documentation to system prompt."""
        state = {
            "trajectory": [],
            "prompt": [{"role": "user", "content": "Test"}],
        }

        messages = await rlm_env_with_sub_tools.get_prompt_messages(state)

        assert "Sub-Agent Tools" in messages[0]["content"]
        assert "sample_tool" in messages[0]["content"]

    @pytest.mark.asyncio
    async def test_string_prompt_converted_to_messages(self, rlm_env):
        """String prompt is converted to message format."""
        state = {
            "trajectory": [],
            "prompt": "What is 2+2?",
        }

        messages = await rlm_env.get_prompt_messages(state)

        # Should have system + user message
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What is 2+2?"


# =============================================================================
# 5. Stop Conditions
# =============================================================================


class TestStopConditions:
    """Tests for stop conditions."""

    @pytest.mark.asyncio
    async def test_answer_ready_true(self, rlm_env):
        """answer_ready returns True when final_answer in state."""
        state = {"final_answer": "42"}
        result = await rlm_env.answer_ready(state)
        assert result is True

    @pytest.mark.asyncio
    async def test_answer_ready_false(self, rlm_env):
        """answer_ready returns False otherwise."""
        state = {}
        result = await rlm_env.answer_ready(state)
        assert result is False

    @pytest.mark.asyncio
    async def test_no_tools_called_always_false(self, rlm_env):
        """no_tools_called always returns False."""
        state = {"trajectory": []}
        result = await rlm_env.no_tools_called(state)
        assert result is False


# =============================================================================
# 6. Sub-LLM Tool Infrastructure
# =============================================================================


class TestCallSubTool:
    """Tests for _call_sub_tool method."""

    @pytest.mark.asyncio
    async def test_executes_tool_successfully(self, rlm_env_with_sub_tools):
        """Executes tool and returns result message."""
        result = await rlm_env_with_sub_tools._call_sub_tool(
            "sample_tool", {"x": 2, "y": 3}, "call_123"
        )

        assert result["role"] == "tool"
        assert result["content"] == "5"  # 2 + 3
        assert result["tool_call_id"] == "call_123"

    @pytest.mark.asyncio
    async def test_handles_tool_error(self, rlm_env_with_sub_tools):
        """Handles tool execution errors gracefully."""
        # Call with wrong arguments
        result = await rlm_env_with_sub_tools._call_sub_tool(
            "sample_tool", {"x": "not_an_int", "y": 3}, "call_456"
        )

        assert result["role"] == "tool"
        assert "Error" in result["content"]
        assert result["tool_call_id"] == "call_456"


class TestRunSubLLMWithTools:
    """Tests for _run_sub_llm_with_tools method."""

    @pytest.mark.asyncio
    async def test_completes_without_tool_calls(self, rlm_env_with_sub_tools):
        """Completes when no tool calls in response."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.tool_calls = None
        mock_message.content = "Final answer"
        mock_response.choices = [MagicMock(message=mock_message)]
        mock_response.model_dump = MagicMock(
            return_value={"choices": [{"message": {"content": "Final answer"}}]}
        )
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        messages = [{"role": "user", "content": "Test"}]
        (
            result,
            prompt_tokens,
            completion_tokens,
            tool_call_count,
            num_turns,
            max_turns_reached,
        ) = await rlm_env_with_sub_tools._run_sub_llm_with_tools(
            mock_client, "gpt-4", messages
        )

        assert "choices" in result
        assert tool_call_count == 0
        assert num_turns == 1
        assert max_turns_reached is False

    @pytest.mark.asyncio
    async def test_executes_tool_calls(self, rlm_env_with_sub_tools):
        """Executes tool calls and continues conversation."""
        mock_client = MagicMock()

        # First response with tool call
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function.name = "sample_tool"
        mock_tool_call.function.arguments = '{"x": 2, "y": 3}'

        mock_message1 = MagicMock()
        mock_message1.tool_calls = [mock_tool_call]
        mock_message1.content = None
        mock_message1.model_dump = MagicMock(
            return_value={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "sample_tool",
                            "arguments": '{"x": 2, "y": 3}',
                        },
                    }
                ],
            }
        )

        # Second response without tool calls
        mock_message2 = MagicMock()
        mock_message2.tool_calls = None
        mock_message2.content = "The result is 5"

        mock_response1 = MagicMock()
        mock_response1.choices = [MagicMock(message=mock_message1)]

        mock_response2 = MagicMock()
        mock_response2.choices = [MagicMock(message=mock_message2)]
        mock_response2.model_dump = MagicMock(
            return_value={"choices": [{"message": {"content": "The result is 5"}}]}
        )

        mock_client.chat.completions.create = AsyncMock(
            side_effect=[mock_response1, mock_response2]
        )

        messages = [{"role": "user", "content": "Add 2 and 3"}]
        result = await rlm_env_with_sub_tools._run_sub_llm_with_tools(
            mock_client, "gpt-4", messages
        )

        assert mock_client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_respects_max_turns_limit(self, rlm_env_with_sub_tools):
        """Respects sub_tool_max_turns limit."""
        mock_client = MagicMock()

        # Always return tool calls
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function.name = "sample_tool"
        mock_tool_call.function.arguments = '{"x": 1, "y": 1}'

        mock_message = MagicMock()
        mock_message.tool_calls = [mock_tool_call]
        mock_message.content = None
        mock_message.model_dump = MagicMock(
            return_value={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "sample_tool",
                            "arguments": '{"x": 1, "y": 1}',
                        },
                    }
                ],
            }
        )

        mock_response_with_tools = MagicMock()
        mock_response_with_tools.choices = [MagicMock(message=mock_message)]

        # Final response without tools
        mock_final_message = MagicMock()
        mock_final_message.tool_calls = None
        mock_final_message.content = "Done"

        mock_final_response = MagicMock()
        mock_final_response.choices = [MagicMock(message=mock_final_message)]
        mock_final_response.model_dump = MagicMock(
            return_value={"choices": [{"message": {"content": "Done"}}]}
        )

        # Return tool calls for max_turns, then final response
        responses = [
            mock_response_with_tools
        ] * rlm_env_with_sub_tools.sub_tool_max_turns
        responses.append(mock_final_response)
        mock_client.chat.completions.create = AsyncMock(side_effect=responses)

        messages = [{"role": "user", "content": "Test"}]
        await rlm_env_with_sub_tools._run_sub_llm_with_tools(
            mock_client, "gpt-4", messages
        )

        # Should be max_turns + 1 (final call without tools)
        assert (
            mock_client.chat.completions.create.call_count
            == rlm_env_with_sub_tools.sub_tool_max_turns + 1
        )


# =============================================================================
# 7. Interception Server
# =============================================================================


class TestHandleSubLLMRequest:
    """Tests for _handle_sub_llm_request method."""

    @pytest.mark.asyncio
    async def test_returns_404_for_unknown_rollout(self, rlm_env):
        """Returns 404 for unknown rollout_id."""
        mock_request = MagicMock()
        mock_request.match_info = {"rollout_id": "unknown_id"}

        response = await rlm_env._handle_sub_llm_request(mock_request)

        assert response.status == 404

    @pytest.mark.asyncio
    async def test_returns_400_for_invalid_json(self, rlm_env):
        """Returns 400 for invalid JSON."""
        rollout_id = "rlm_test123"
        rlm_env.active_rollouts[rollout_id] = {
            "client": MagicMock(),
            "model": "test-model",
        }

        mock_request = MagicMock()
        mock_request.match_info = {"rollout_id": rollout_id}
        mock_request.json = AsyncMock(
            side_effect=json.JSONDecodeError("test", "doc", 0)
        )

        response = await rlm_env._handle_sub_llm_request(mock_request)

        assert response.status == 400

    @pytest.mark.asyncio
    async def test_routes_to_correct_model(self, rlm_env):
        """Routes to correct sub-model via client."""
        rollout_id = "rlm_test123"
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.model_dump = MagicMock(
            return_value={"choices": [{"message": {"content": "response"}}]}
        )
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        rlm_env.active_rollouts[rollout_id] = {
            "client": mock_client,
            "model": "test-model",
            "sub_model": "gpt-4",
        }

        mock_request = MagicMock()
        mock_request.match_info = {"rollout_id": rollout_id}
        mock_request.json = AsyncMock(
            return_value={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "test"}],
            }
        )

        response = await rlm_env._handle_sub_llm_request(mock_request)

        mock_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_uses_tool_loop_when_configured(self, rlm_env_with_sub_tools):
        """Uses tool-calling loop when sub_tools configured."""
        rollout_id = "rlm_test123"
        mock_client = MagicMock()

        # Response without tool calls
        mock_message = MagicMock()
        mock_message.tool_calls = None
        mock_message.content = "response"
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_message)]
        mock_response.model_dump = MagicMock(
            return_value={"choices": [{"message": {"content": "response"}}]}
        )
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        rlm_env_with_sub_tools.active_rollouts[rollout_id] = {
            "client": mock_client,
            "model": "test-model",
            "sub_model": "gpt-4",
        }

        mock_request = MagicMock()
        mock_request.match_info = {"rollout_id": rollout_id}
        mock_request.json = AsyncMock(
            return_value={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "test"}],
            }
        )

        response = await rlm_env_with_sub_tools._handle_sub_llm_request(mock_request)

        # Should have called with tools parameter
        call_kwargs = mock_client.chat.completions.create.call_args
        assert "tools" in call_kwargs.kwargs


# =============================================================================
# Post Rollout
# =============================================================================


class TestPostRollout:
    """Tests for post_rollout method."""

    @pytest.mark.asyncio
    async def test_skips_if_final_answer_exists(self, rlm_env):
        """Skips reading if final_answer already in state."""
        state = {"final_answer": "already set", "sandbox_id": "sandbox_123"}

        await rlm_env.post_rollout(state)

        # Should not have tried to read from sandbox
        rlm_env.sandbox_client.execute_command.assert_not_called()

    @pytest.mark.asyncio
    async def test_reads_answer_from_sandbox(self, rlm_env):
        """Reads answer from sandbox if not set."""
        rlm_env.sandbox_client.execute_command = AsyncMock(
            return_value=MagicMock(
                stdout='{"content": "read from sandbox", "ready": true}'
            )
        )
        state = {"sandbox_id": "sandbox_123"}

        await rlm_env.post_rollout(state)

        assert state["final_answer"] == "read from sandbox"

    @pytest.mark.asyncio
    async def test_handles_missing_sandbox(self, rlm_env):
        """Handles missing sandbox_id."""
        state = {}

        await rlm_env.post_rollout(state)

        assert state["final_answer"] == ""

    @pytest.mark.asyncio
    async def test_handles_read_error(self, rlm_env):
        """Handles errors when reading from sandbox."""
        rlm_env.sandbox_client.execute_command = AsyncMock(
            return_value=MagicMock(stdout="invalid json")
        )
        state = {"sandbox_id": "sandbox_123"}

        await rlm_env.post_rollout(state)

        assert state["final_answer"] == ""


# =============================================================================
# 8. Sub-LLM Metrics Tracking
# =============================================================================


class TestSubLLMMetricsInitialization:
    """Tests for sub-LLM metrics initialization."""

    @pytest.mark.asyncio
    async def test_initializes_metrics_in_active_rollouts(self, rlm_env):
        """Initializes sub-LLM metrics tracking in active_rollouts."""
        rlm_env._ensure_interception_server = AsyncMock()
        rlm_env._tunnel_pool.get_tunnel_url = AsyncMock(
            return_value="https://test.trycloudflare.com"
        )
        rlm_env._write_json_to_sandbox = AsyncMock()
        rlm_env._wait_for_worker_ready = AsyncMock()

        state = {"info": {}, "model": "test-model", "client": MagicMock()}
        result = await rlm_env.setup_state(state)

        rollout_context = rlm_env.active_rollouts[result["rollout_id"]]
        assert rollout_context["sub_llm_call_count"] == 0
        assert rollout_context["sub_llm_prompt_tokens"] == 0
        assert rollout_context["sub_llm_completion_tokens"] == 0


class TestSubLLMMetricsTracking:
    """Tests for sub-LLM metrics tracking in _handle_sub_llm_request."""

    @pytest.mark.asyncio
    async def test_appends_call_to_list(self, rlm_env):
        """Appends call data to sub_llm_calls list for each sub-LLM request."""
        rollout_id = "rlm_test123"
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.model_dump = MagicMock(
            return_value={
                "choices": [{"message": {"content": "response"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            }
        )
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        rlm_env.active_rollouts[rollout_id] = {
            "client": mock_client,
            "model": "test-model",
            "sub_model": "gpt-4",
        }

        mock_request = MagicMock()
        mock_request.match_info = {"rollout_id": rollout_id}
        mock_request.json = AsyncMock(
            return_value={
                "messages": [{"role": "user", "content": "test"}],
            }
        )

        await rlm_env._handle_sub_llm_request(mock_request)

        context = rlm_env.active_rollouts[rollout_id]
        sub_llm_calls = context.get("sub_llm_calls", [])
        assert len(sub_llm_calls) == 1
        assert sub_llm_calls[0]["metadata"]["prompt_tokens"] == 10
        assert sub_llm_calls[0]["metadata"]["completion_tokens"] == 20

    @pytest.mark.asyncio
    async def test_accumulates_across_requests(self, rlm_env):
        """Accumulates call data across multiple sub-LLM requests."""
        rollout_id = "rlm_test123"
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.model_dump = MagicMock(
            return_value={
                "choices": [{"message": {"content": "response"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            }
        )
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Pre-populate with 5 existing calls
        existing_calls = [
            {"metadata": {"prompt_tokens": 20, "completion_tokens": 40}}
            for _ in range(5)
        ]
        rlm_env.active_rollouts[rollout_id] = {
            "client": mock_client,
            "model": "test-model",
            "sub_model": "gpt-4",
            "sub_llm_calls": existing_calls,
        }

        mock_request = MagicMock()
        mock_request.match_info = {"rollout_id": rollout_id}
        mock_request.json = AsyncMock(
            return_value={
                "messages": [{"role": "user", "content": "test"}],
            }
        )

        await rlm_env._handle_sub_llm_request(mock_request)

        context = rlm_env.active_rollouts[rollout_id]
        sub_llm_calls = context.get("sub_llm_calls", [])
        assert len(sub_llm_calls) == 6
        # New call should have the new tokens
        assert sub_llm_calls[-1]["metadata"]["prompt_tokens"] == 10
        assert sub_llm_calls[-1]["metadata"]["completion_tokens"] == 20

    @pytest.mark.asyncio
    async def test_handles_missing_usage(self, rlm_env):
        """Handles responses without usage data gracefully."""
        rollout_id = "rlm_test123"
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.model_dump = MagicMock(
            return_value={
                "choices": [{"message": {"content": "response"}}],
                # No "usage" field
            }
        )
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        rlm_env.active_rollouts[rollout_id] = {
            "client": mock_client,
            "model": "test-model",
            "sub_model": "gpt-4",
        }

        mock_request = MagicMock()
        mock_request.match_info = {"rollout_id": rollout_id}
        mock_request.json = AsyncMock(
            return_value={
                "messages": [{"role": "user", "content": "test"}],
            }
        )

        await rlm_env._handle_sub_llm_request(mock_request)

        context = rlm_env.active_rollouts[rollout_id]
        sub_llm_calls = context.get("sub_llm_calls", [])
        assert len(sub_llm_calls) == 1
        assert sub_llm_calls[0]["metadata"]["prompt_tokens"] == 0
        assert sub_llm_calls[0]["metadata"]["completion_tokens"] == 0


class TestSubLLMMetricsWithTools:
    """Tests for sub-LLM metrics with tool-calling loop."""

    @pytest.mark.asyncio
    async def test_accumulates_tokens_across_tool_turns(self, rlm_env_with_sub_tools):
        """Accumulates tokens across multiple tool-calling turns."""
        mock_client = MagicMock()

        # First response with tool call
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function.name = "sample_tool"
        mock_tool_call.function.arguments = '{"x": 2, "y": 3}'

        mock_message1 = MagicMock()
        mock_message1.tool_calls = [mock_tool_call]
        mock_message1.content = None
        mock_message1.model_dump = MagicMock(
            return_value={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "sample_tool",
                            "arguments": '{"x": 2, "y": 3}',
                        },
                    }
                ],
            }
        )

        mock_response1 = MagicMock()
        mock_response1.choices = [MagicMock(message=mock_message1)]
        mock_response1.usage = MagicMock(prompt_tokens=50, completion_tokens=30)

        # Second response without tool calls
        mock_message2 = MagicMock()
        mock_message2.tool_calls = None
        mock_message2.content = "The result is 5"

        mock_response2 = MagicMock()
        mock_response2.choices = [MagicMock(message=mock_message2)]
        mock_response2.usage = MagicMock(prompt_tokens=100, completion_tokens=20)
        mock_response2.model_dump = MagicMock(
            return_value={
                "choices": [{"message": {"content": "The result is 5"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            }
        )

        mock_client.chat.completions.create = AsyncMock(
            side_effect=[mock_response1, mock_response2]
        )

        messages = [{"role": "user", "content": "Add 2 and 3"}]
        (
            response_dict,
            prompt_tokens,
            completion_tokens,
            tool_call_count,
            num_turns,
            max_turns_reached,
        ) = await rlm_env_with_sub_tools._run_sub_llm_with_tools(
            mock_client, "gpt-4", messages
        )

        # Should accumulate tokens from both calls
        assert prompt_tokens == 150  # 50 + 100
        assert completion_tokens == 50  # 30 + 20
        assert tool_call_count == 1  # One tool call was made
        assert num_turns == 2  # Two LLM calls: one with tool call, one final
        assert max_turns_reached is False


class TestSubLLMMetricsCleanup:
    """Tests for metrics cleanup."""

    @pytest.mark.asyncio
    async def test_derives_metrics_from_calls_list(self, rlm_env):
        """Derives sub-LLM metrics from sub_llm_calls list during cleanup."""
        rollout_id = "rlm_test123"
        # Populate sub_llm_calls list with call data
        sub_llm_calls = [
            {
                "metadata": {
                    "prompt_tokens": 30,
                    "completion_tokens": 60,
                    "tool_call_count": 1,
                    "num_turns": 2,
                }
            },
            {
                "metadata": {
                    "prompt_tokens": 50,
                    "completion_tokens": 100,
                    "tool_call_count": 2,
                    "num_turns": 3,
                }
            },
            {
                "metadata": {
                    "prompt_tokens": 70,
                    "completion_tokens": 140,
                    "tool_call_count": 0,
                    "num_turns": 1,
                }
            },
        ]
        rlm_env.active_rollouts[rollout_id] = {
            "client": MagicMock(),
            "sub_llm_calls": sub_llm_calls,
        }

        state = {"rollout_id": rollout_id}
        await rlm_env.cleanup_rlm_state(state)

        # Metrics should be derived from the calls list
        assert state["sub_llm_call_count"] == 3
        assert state["sub_llm_prompt_tokens"] == 150  # 30 + 50 + 70
        assert state["sub_llm_completion_tokens"] == 300  # 60 + 100 + 140
        assert state["sub_llm_total_tool_calls"] == 3  # 1 + 2 + 0
        assert state["sub_llm_total_turns"] == 6  # 2 + 3 + 1

    @pytest.mark.asyncio
    async def test_handles_missing_calls_list_gracefully(self, rlm_env):
        """Handles missing sub_llm_calls list in active_rollouts gracefully."""
        rollout_id = "rlm_test123"
        rlm_env.active_rollouts[rollout_id] = {
            "client": MagicMock(),
            # No sub_llm_calls field
        }

        state = {"rollout_id": rollout_id}
        await rlm_env.cleanup_rlm_state(state)

        assert state["sub_llm_call_count"] == 0
        assert state["sub_llm_prompt_tokens"] == 0
        assert state["sub_llm_completion_tokens"] == 0


class TestSubLLMMetricsRubric:
    """Tests for automatic sub-LLM metrics rubric."""

    def test_rubric_includes_metric_functions(self, rlm_env):
        """Rubric includes 0-weighted metric functions."""
        func_names = rlm_env.rubric._get_reward_func_names()
        assert "sub_llm_calls" in func_names
        assert "sub_llm_prompt_tokens" in func_names
        assert "sub_llm_completion_tokens" in func_names

    def test_metric_weights_are_zero(self, rlm_env):
        """Metric functions have weight 0."""
        names = rlm_env.rubric._get_reward_func_names()
        weights = rlm_env.rubric._get_reward_weights()
        name_to_weight = dict(zip(names, weights))
        assert name_to_weight["sub_llm_calls"] == 0.0
        assert name_to_weight["sub_llm_prompt_tokens"] == 0.0
        assert name_to_weight["sub_llm_completion_tokens"] == 0.0

    def test_user_rubric_combined(self, mock_sandbox_client, mock_dataset):
        """User rubric is combined with internal metrics rubric."""

        def custom_reward(state) -> float:
            return 1.0

        user_rubric = Rubric(funcs=[custom_reward], weights=[1.0])

        with (
            patch("verifiers.envs.sandbox_env.AsyncSandboxClient") as mock_client_cls,
            patch("verifiers.envs.sandbox_env.CreateSandboxRequest"),
        ):
            mock_client_cls.return_value = mock_sandbox_client
            env = RLMEnv(dataset=mock_dataset, rubric=user_rubric)

            func_names = env.rubric._get_reward_func_names()
            # Should have both internal metrics and user reward
            assert "sub_llm_calls" in func_names
            assert "sub_llm_prompt_tokens" in func_names
            assert "sub_llm_completion_tokens" in func_names
            assert "custom_reward" in func_names
