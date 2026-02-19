"""Tests for rate limit error handling and retry mechanism."""

import httpx
import pytest
from openai import RateLimitError as OpenAIRateLimitError

from verifiers.errors import RateLimitError as VFRateLimitError
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
    error_sequence = [
        VFRateLimitError("Rate limited"),
        InfraError("Infra error"),
        {"result": "success"}
    ]

    async def multi_error_func():
        nonlocal call_count
        result = error_sequence[min(call_count, len(error_sequence) - 1)]
        call_count += 1
        return result

    wrapped = maybe_retry(
        multi_error_func,
        max_retries=3,
        initial=0.01,
        error_types=(VFRateLimitError, InfraError)
    )
    result = await wrapped()

    assert result["result"] == "success"
    assert call_count == 3
