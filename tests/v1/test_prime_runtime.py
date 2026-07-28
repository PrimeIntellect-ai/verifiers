import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import prime_sandboxes
import pytest
from pydantic import ValidationError

from verifiers.v1.runtimes.prime import PrimeConfig, PrimeRuntime


def test_prime_creation_wait_attempts_default_and_validation() -> None:
    assert PrimeConfig().wait_for_creation_max_attempts == 60
    with pytest.raises(ValidationError):
        PrimeConfig(wait_for_creation_max_attempts=0)


def test_prime_creation_wait_attempts_reach_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = SimpleNamespace(
        create=AsyncMock(
            return_value=SimpleNamespace(
                id="sandbox-1",
                pending_image_build_id=None,
            )
        ),
        wait_for_creation=AsyncMock(),
        execute_command=AsyncMock(),
    )
    monkeypatch.setattr(prime_sandboxes, "AsyncSandboxClient", lambda: client)
    monkeypatch.setattr(
        prime_sandboxes,
        "CreateSandboxRequest",
        lambda **values: values,
    )

    runtime = PrimeRuntime(
        PrimeConfig(
            image="example.invalid/image:latest",
            wait_for_creation_max_attempts=600,
        )
    )
    asyncio.run(runtime.start())

    client.wait_for_creation.assert_awaited_once_with(
        "sandbox-1",
        max_attempts=600,
    )
