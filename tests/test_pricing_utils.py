import asyncio
import json

import pytest

from verifiers.utils import pricing_utils
from verifiers.utils.pricing_utils import (
    compute_cost_usd,
    compute_eval_cost,
    fetch_prime_pricing,
)


class FakeProcess:
    def __init__(self, returncode: int, stdout: bytes, stderr: bytes = b"") -> None:
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr


def test_compute_cost_usd_zero_usage() -> None:
    usage = {"input_tokens": 0.0, "output_tokens": 0.0}
    pricing = {"input_usd_per_mtok": 1.0, "output_usd_per_mtok": 5.0}

    assert compute_cost_usd(usage, pricing) == 0.0
    assert compute_eval_cost(usage, pricing) == {
        "input_usd": 0.0,
        "output_usd": 0.0,
        "total_usd": 0.0,
    }


def test_compute_cost_usd_missing_pricing() -> None:
    usage = {"input_tokens": 1_000.0, "output_tokens": 2_000.0}

    assert compute_cost_usd(usage, None) is None


def test_compute_cost_usd_normal_case() -> None:
    usage = {"input_tokens": 1_000.0, "output_tokens": 2_000.0}
    pricing = {"input_usd_per_mtok": 1.0, "output_usd_per_mtok": 5.0}

    assert compute_cost_usd(usage, pricing) == pytest.approx(0.011)
    assert compute_eval_cost(usage, pricing) == {
        "input_usd": pytest.approx(0.001),
        "output_usd": pytest.approx(0.010),
        "total_usd": pytest.approx(0.011),
    }


def test_compute_cost_usd_only_input_tokens() -> None:
    usage = {"input_tokens": 2_000.0, "output_tokens": 0.0}
    pricing = {"input_usd_per_mtok": 1.0, "output_usd_per_mtok": 5.0}

    assert compute_cost_usd(usage, pricing) == pytest.approx(0.002)


def test_compute_cost_usd_only_output_tokens() -> None:
    usage = {"input_tokens": 0.0, "output_tokens": 2_000.0}
    pricing = {"input_usd_per_mtok": 1.0, "output_usd_per_mtok": 5.0}

    assert compute_cost_usd(usage, pricing) == pytest.approx(0.010)


async def test_fetch_prime_pricing_parses_json_fixture(monkeypatch) -> None:
    pricing_utils._clear_prime_pricing_cache()
    payload = {
        "object": "list",
        "data": [
            {
                "id": "anthropic/claude-haiku-4.5",
                "pricing": {
                    "input_usd_per_mtok": 1.0,
                    "output_usd_per_mtok": 5.0,
                    "effective_at": "2025-10-21T07:14:20.826034+00:00",
                },
            },
            {"id": "unpriced/model"},
        ],
    }

    async def fake_create_subprocess_exec(*cmd, stdout, stderr):
        assert cmd == (
            "prime",
            "inference",
            "models",
            "--output",
            "json",
            "--plain",
        )
        assert stdout is asyncio.subprocess.PIPE
        assert stderr is asyncio.subprocess.PIPE
        return FakeProcess(0, json.dumps(payload).encode())

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    assert await fetch_prime_pricing() == {
        "anthropic/claude-haiku-4.5": {
            "input_usd_per_mtok": 1.0,
            "output_usd_per_mtok": 5.0,
        }
    }


async def test_fetch_prime_pricing_binary_missing_returns_empty(monkeypatch) -> None:
    pricing_utils._clear_prime_pricing_cache()

    async def fake_create_subprocess_exec(*cmd, stdout, stderr):
        raise FileNotFoundError("prime")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    assert await fetch_prime_pricing() == {}


async def test_fetch_prime_pricing_bad_json_returns_empty(monkeypatch) -> None:
    pricing_utils._clear_prime_pricing_cache()

    async def fake_create_subprocess_exec(*cmd, stdout, stderr):
        return FakeProcess(0, b"{not json")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    assert await fetch_prime_pricing() == {}
