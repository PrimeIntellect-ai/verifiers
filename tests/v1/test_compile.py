"""Tests for verifiers.v1.utils.compile."""

from types import SimpleNamespace

from verifiers.v1.runtimes.modal import ModalConfig
from verifiers.v1.runtimes.prime import PrimeConfig
from verifiers.v1.runtimes.subprocess import SubprocessConfig
from verifiers.v1.utils.compile import cap_remote_harness_timeout


def _task(idx: str = "t0") -> SimpleNamespace:
    return SimpleNamespace(data=SimpleNamespace(idx=idx))


def test_cap_remote_harness_timeout_caps_at_runtime_timeout():
    """A harness timeout longer than the runtime's hard timeout is capped there."""
    config = PrimeConfig(timeout=120, idle_timeout=60)
    capped = cap_remote_harness_timeout(3600, config, _task())
    assert capped == 120


def test_cap_remote_harness_timeout_passes_through_when_under_cap():
    """A harness timeout under the runtime timeout passes through unchanged."""
    config = PrimeConfig(timeout=3600)
    assert cap_remote_harness_timeout(120, config, _task()) == 120


def test_cap_remote_harness_timeout_passes_through_for_local_runtime():
    """Local runtimes are never capped."""
    config = SubprocessConfig()
    assert cap_remote_harness_timeout(999999, config, _task()) == 999999


def test_cap_remote_harness_timeout_truncates_fractional_lifetime():
    """Modal truncates timeout to int(); the cap must match so the harness limit
    never exceeds the sandbox's actual (truncated) lifetime."""
    config = ModalConfig(timeout=120.5)
    capped = cap_remote_harness_timeout(121, config, _task())
    assert capped == 120  # int(120.5) = 120, not 120.5


def test_cap_remote_harness_timeout_none_passes_through():
    """A None harness timeout passes through unchanged."""
    config = PrimeConfig(timeout=120, idle_timeout=60)
    assert cap_remote_harness_timeout(None, config, _task()) is None
