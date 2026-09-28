"""A provider fault the ACP agent reports over the wire becomes a ProviderError, not a HarnessError."""

import py_compile
import tempfile
from pathlib import Path

from verifiers.v1.acp import ACP_SOURCE, _turn_failure
from verifiers.v1.errors import ProviderError


def test_turn_failure_maps_provider_kind_to_provider_error():
    err = _turn_failure(
        "rlm",
        {"ok": False, "error": "Connection error", "error_kind": "provider"},
        "ACP stderr tail",
    )
    assert isinstance(err, ProviderError)
    assert "rlm" in str(err) and "Connection error" in str(err) and "stderr" in str(err)


def test_turn_failure_without_kind_is_runtime_error():
    err = _turn_failure("rlm", {"ok": False, "error": "engine bug"}, "")
    assert type(err) is RuntimeError
    assert not isinstance(err, ProviderError)


def test_runner_source_carries_error_kind_and_compiles():
    # runner.py runs standalone in the sandbox (imports the `acp` package, absent on the host),
    # so guard its wiring through the shipped source rather than importing it.
    assert "class SessionTurnError" in ACP_SOURCE
    assert 'packet["error_kind"] = kind' in ACP_SOURCE
    path = Path(tempfile.mkdtemp()) / "runner.py"
    path.write_text(ACP_SOURCE)
    py_compile.compile(str(path), doraise=True)
