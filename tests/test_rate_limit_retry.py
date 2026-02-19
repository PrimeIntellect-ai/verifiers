"""Tests for rate limit error handling and retry mechanism."""

import httpx
import pytest
from openai import RateLimitError as OpenAIRateLimitError

from verifiers.errors import RateLimitError as VFRateLimitError
from verifiers.types import EvalConfig
from verifiers.utils.async_utils import maybe_retry


def _make_rate_limit_error() -> OpenAIRateLimitError:
    response = httpx.Response(
        status_code=429,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        json={"error": {"message": "Too many requests", "type": "rate_limit_error"}},
    )
    return OpenAIRateLimitError("Rate limit exceeded", response=response, body=None)


@pytest.mark.asyncio
async def test_rate_limit_error_retries_with_config():
    """Test that RateLimitError triggers retry when configured."""
    call_count = 0

    async def failing_func():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return {"error": VFRateLimitError("Rate limited")}
        return {"result": "success"}

    wrapped = maybe_retry(failing_func, max_retries=3, initial=0.01)
    result = await wrapped()

    assert call_count == 3
    assert result["result"] == "success"


@pytest.mark.asyncio
async def test_rate_limit_error_exhaustion_returns_error():
    """Test that exhausted retries return error in state."""
    async def always_failing_func():
        return {"error": VFRateLimitError("Always rate limited")}

    wrapped = maybe_retry(always_failing_func, max_retries=2, initial=0.01)
    result = await wrapped()

    assert "error" in result
    assert isinstance(result["error"], VFRateLimitError)


@pytest.mark.asyncio
async def test_no_retry_when_max_retries_zero():
    """Test that max_retries=0 disables retry."""
    call_count = 0

    async def failing_func():
        nonlocal call_count
        call_count += 1
        return {"error": VFRateLimitError("Rate limited")}

    wrapped = maybe_retry(failing_func, max_retries=0)
    result = await wrapped()

    assert call_count == 1  # Only called once, no retry
    assert "error" in result


@pytest.mark.asyncio
async def test_jitter_configuration():
    """Test that jitter can be disabled."""
    async def failing_func():
        return {"error": VFRateLimitError("Rate limited")}

    # Should not raise with jitter enabled (default)
    wrapped_with_jitter = maybe_retry(failing_func, max_retries=1, initial=0.01, jitter=True)
    result = await wrapped_with_jitter()
    assert "error" in result

    # Should not raise with jitter disabled
    wrapped_no_jitter = maybe_retry(failing_func, max_retries=1, initial=0.01, jitter=False)
    result = await wrapped_no_jitter()
    assert "error" in result


@pytest.mark.asyncio
async def test_multiple_error_types_in_retry():
    """Test that multiple error types can be retried."""
    from verifiers.errors import InfraError

    call_count = 0

    async def multi_error_func():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"error": VFRateLimitError("Rate limited")}
        elif call_count == 2:
            return {"error": InfraError("Infra error")}
        else:
            return {"result": "success"}

    wrapped = maybe_retry(
        multi_error_func,
        max_retries=3,
        initial=0.01,
        error_types=(VFRateLimitError, InfraError)
    )
    result = await wrapped()

    assert result["result"] == "success"
    assert call_count == 3


@pytest.mark.asyncio
async def test_retry_configuration_values_are_used():
    """Test that EvalConfig accepts and stores retry timing parameters."""
    from verifiers.types import ClientConfig

    config = EvalConfig(
        env_id="test_env",
        env_args={},
        env_dir_path="/tmp/test",
        model="gpt-4",
        client_config=ClientConfig(api_key_var="TEST_KEY"),
        sampling_args={},
        num_examples=1,
        rollouts_per_example=1,
        max_concurrent=1,
        retry_base_delay=2.0,
        retry_max_backoff=30.0,
        retry_jitter=False,
    )

    assert config.retry_base_delay == 2.0
    assert config.retry_max_backoff == 30.0
    assert config.retry_jitter is False


@pytest.mark.asyncio
async def test_retry_configuration_defaults():
    """Test that EvalConfig has correct default values for retry timing."""
    from verifiers.types import ClientConfig

    config = EvalConfig(
        env_id="test_env",
        env_args={},
        env_dir_path="/tmp/test",
        model="gpt-4",
        client_config=ClientConfig(api_key_var="TEST_KEY"),
        sampling_args={},
        num_examples=1,
        rollouts_per_example=1,
        max_concurrent=1,
    )

    # Verify defaults match maybe_retry defaults
    assert config.retry_base_delay == 1.0
    assert config.retry_max_backoff == 60.0
    assert config.retry_jitter is True
