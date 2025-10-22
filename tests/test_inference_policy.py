# ABOUTME: Unit tests for inference policy abstractions
# ABOUTME: Tests APIPolicy, VLLMPolicy, and factory methods
import pytest
from unittest.mock import AsyncMock, Mock

from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletion
from openai.types import Completion

from verifiers.inference.policy import APIPolicy, InferencePolicy


class TestInferencePolicy:
    """Test base InferencePolicy abstraction."""

    def test_cannot_instantiate_abstract_class(self):
        """Base class cannot be instantiated directly."""
        with pytest.raises(TypeError):
            InferencePolicy()  # type: ignore

    def test_from_client_with_async_client(self):
        """Factory method creates APIPolicy from AsyncOpenAI client."""
        client = AsyncOpenAI(api_key="test", base_url="http://test")
        policy = InferencePolicy.from_client(client, "test-model")

        assert isinstance(policy, APIPolicy)
        assert policy.model == "test-model"
        assert policy.client == client

    def test_from_client_with_sync_client(self):
        """Factory method creates APIPolicy from sync OpenAI client."""
        client = OpenAI(api_key="test", base_url="http://test")
        policy = InferencePolicy.from_client(client, "test-model")

        assert isinstance(policy, APIPolicy)
        assert policy.model == "test-model"
        # Should wrap sync client in async client
        assert isinstance(policy.client, AsyncOpenAI)


class TestAPIPolicy:
    """Test APIPolicy implementation."""

    @pytest.fixture
    def mock_async_client(self):
        """Create mock AsyncOpenAI client."""
        client = Mock(spec=AsyncOpenAI)
        client.api_key = "test-key"
        client.base_url = "http://test"
        return client

    @pytest.fixture
    def api_policy(self, mock_async_client):
        """Create APIPolicy instance."""
        return APIPolicy(client=mock_async_client, model="test-model")

    def test_initialization_with_async_client(self, mock_async_client):
        """APIPolicy initializes correctly with AsyncOpenAI client."""
        policy = APIPolicy(client=mock_async_client, model="test-model")
        assert policy.client == mock_async_client
        assert policy.model == "test-model"

    def test_initialization_with_sync_client(self):
        """APIPolicy wraps sync OpenAI client in AsyncOpenAI."""
        sync_client = Mock(spec=OpenAI)
        sync_client.api_key = "test-key"
        sync_client.base_url = "http://test"

        policy = APIPolicy(client=sync_client, model="test-model")

        assert isinstance(policy.client, AsyncOpenAI)
        assert policy.model == "test-model"

    @pytest.mark.asyncio
    async def test_generate_chat_format(self, api_policy, mock_async_client):
        """Generate works with chat completion format."""
        # Setup mock response
        mock_response = Mock(spec=ChatCompletion)
        mock_async_client.chat = Mock()
        mock_async_client.chat.completions = Mock()
        mock_async_client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Test chat format
        prompt = [{"role": "user", "content": "Hello"}]
        result = await api_policy.generate(
            prompt=prompt,
            sampling_args={"temperature": 0.7}
        )

        # Verify call was made
        mock_async_client.chat.completions.create.assert_called_once_with(
            model="test-model",
            messages=prompt,
            temperature=0.7
        )
        assert result == mock_response

    @pytest.mark.asyncio
    async def test_generate_completion_format(self, api_policy, mock_async_client):
        """Generate works with completion format."""
        # Setup mock response
        mock_response = Mock(spec=Completion)
        mock_async_client.completions = Mock()
        mock_async_client.completions.create = AsyncMock(return_value=mock_response)

        # Test completion format
        prompt = "Hello, world"
        result = await api_policy.generate(
            prompt=prompt,
            sampling_args={"temperature": 0.7}
        )

        # Verify call was made
        mock_async_client.completions.create.assert_called_once_with(
            model="test-model",
            prompt=prompt,
            temperature=0.7
        )
        assert result == mock_response

    @pytest.mark.asyncio
    async def test_generate_with_no_sampling_args(self, api_policy, mock_async_client):
        """Generate works without sampling args."""
        mock_response = Mock(spec=ChatCompletion)
        mock_async_client.chat = Mock()
        mock_async_client.chat.completions = Mock()
        mock_async_client.chat.completions.create = AsyncMock(return_value=mock_response)

        prompt = [{"role": "user", "content": "Hello"}]
        await api_policy.generate(prompt=prompt)

        # Should still call with model
        mock_async_client.chat.completions.create.assert_called_once()
        call_kwargs = mock_async_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "test-model"
        assert call_kwargs["messages"] == prompt


class TestVLLMPolicy:
    """Test VLLMPolicy implementation."""

    def test_initialization_requires_vllm(self):
        """VLLMPolicy requires vLLM to be installed."""
        # This will fail if vLLM is not installed, which is expected
        # In CI, we should skip this test or mock the import
        try:
            from verifiers.inference.backends.vllm_policy import VLLMPolicy
            # If import succeeds, test basic initialization
            # Note: This will try to connect to a server, so we can't fully test without mocking
            assert VLLMPolicy is not None
        except ImportError:
            pytest.skip("vLLM not installed")

    def test_vllm_policy_structure(self):
        """VLLMPolicy has expected methods and attributes."""
        try:
            from verifiers.inference.backends.vllm_policy import VLLMPolicy

            # Check class has required methods
            assert hasattr(VLLMPolicy, 'generate')
            assert hasattr(VLLMPolicy, 'enable_weight_sync')
            assert hasattr(VLLMPolicy, 'sync_weights')
            assert hasattr(VLLMPolicy, 'from_client')
        except ImportError:
            pytest.skip("vLLM not installed")


class TestBackwardsCompatibility:
    """Test backwards compatibility patterns."""

    def test_policy_can_wrap_existing_client_code(self):
        """Existing code using clients can be wrapped in policy."""
        # Simulate existing code pattern
        client = OpenAI(api_key="test", base_url="http://test")
        model = "gpt-4"

        # New code can wrap this
        policy = InferencePolicy.from_client(client, model)

        assert isinstance(policy, APIPolicy)
        assert policy.model == model

    @pytest.mark.asyncio
    async def test_policy_interface_matches_client_usage(self):
        """Policy interface is similar to direct client usage."""
        mock_client = Mock(spec=AsyncOpenAI)
        mock_client.chat = Mock()
        mock_client.chat.completions = Mock()
        mock_client.chat.completions.create = AsyncMock(return_value=Mock())

        policy = APIPolicy(client=mock_client, model="test")

        # Both interfaces should work similarly
        prompt = [{"role": "user", "content": "test"}]
        sampling_args = {"temperature": 0.7}

        # Policy interface
        await policy.generate(prompt=prompt, sampling_args=sampling_args)

        # Verify underlying client was called
        assert mock_client.chat.completions.create.called
