import sys
from unittest.mock import AsyncMock

import pytest

from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.loaders import harness_class, harness_config_type, import_harness
from verifiers.v1.runtimes.prime import PrimeConfig, PrimeRuntime


def test_external_hyphenated_harness_id_loads_from_flat_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "verifiers.v1.harnesses.external_harness_v1", None)

    module = import_harness("external-harness-v1")

    assert module.__name__ == "external_harness_v1"
    assert harness_class("external-harness-v1").__name__ == "ExternalHarness"
    config_type = harness_config_type("external-harness-v1")
    assert config_type.__name__ == "ExternalHarnessConfig"
    assert (
        config_type.model_validate(
            {"id": "external-harness-v1", "custom_flag": True}
        ).custom_flag
        is True
    )


def test_external_harness_config_fields_are_forwarded_from_eval_config():
    config = EvalConfig.model_validate(
        {
            "taskset": {"id": "echo-v1"},
            "harness": {"id": "external-harness-v1", "custom_flag": True},
        }
    )

    assert type(config.harness).__name__ == "ExternalHarnessConfig"
    assert config.harness.id == "external-harness-v1"
    assert config.harness.custom_flag is True


def test_external_harness_internal_import_errors_are_not_hidden(tmp_path, monkeypatch):
    (tmp_path / "broken_harness_v1.py").write_text(
        "import missing_dependency_for_loader_test\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ModuleNotFoundError) as exc_info:
        harness_class("broken-harness-v1")

    assert exc_info.value.name == "missing_dependency_for_loader_test"


def test_prime_runtime_preserves_explicit_labels_exactly():
    config = PrimeConfig(labels=["programbench"])

    assert config.labels == ["programbench"]


@pytest.mark.asyncio
async def test_prime_runtime_poll_fallback_accepts_running_status():
    class Sandbox:
        status = "RUNNING"

    runtime = PrimeRuntime(PrimeConfig())
    runtime._sandbox_id = "sandbox-id"
    runtime._client = AsyncMock()
    runtime._client.wait_for_creation.side_effect = TimeoutError("timed out")
    runtime._client.get.return_value = Sandbox()

    await runtime._wait_until_running()

    runtime._client.get.assert_awaited_once_with("sandbox-id")
