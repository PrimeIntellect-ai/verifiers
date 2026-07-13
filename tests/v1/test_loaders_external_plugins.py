import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from verifiers.v1.loaders import harness_class, import_harness


FIXTURE_PACKAGE = Path(__file__).parent / "fixtures" / "external_plugin_package"


def test_installed_external_package_constructs_full_environment(tmp_path):
    site_packages = tmp_path / "site-packages"
    install = subprocess.run(
        [
            shutil.which("uv") or "uv",
            "pip",
            "install",
            "--target",
            str(site_packages),
            "--no-deps",
            str(FIXTURE_PACKAGE),
        ],
        capture_output=True,
        text=True,
    )
    assert install.returncode == 0, install.stderr

    script = r"""
import sys

sys.path.insert(0, sys.argv[1])

from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.env import Environment

config = EvalConfig.model_validate(
    {
        "taskset": {"id": "external-plugin-v1", "custom_taskset_flag": True},
        "harness": {"id": "external-plugin-v1", "custom_harness_flag": True},
    }
)
assert type(config.taskset).__name__ == "ExternalTasksetConfig"
assert config.taskset.custom_taskset_flag is True
assert type(config.harness).__name__ == "ExternalHarnessConfig"
assert config.harness.custom_harness_flag is True

environment = Environment(config)
assert type(environment.taskset).__name__ == "ExternalTaskset"
assert type(environment.harness).__name__ == "ExternalHarness"
assert environment.harness.config is config.harness
assert len(environment.taskset.select()) == 1
"""
    construct = subprocess.run(
        [sys.executable, "-I", "-c", script, str(site_packages)],
        capture_output=True,
        text=True,
    )
    assert construct.returncode == 0, construct.stderr


def test_external_harness_internal_import_errors_are_not_hidden(tmp_path, monkeypatch):
    (tmp_path / "broken_harness_v1.py").write_text(
        "import missing_dependency_for_loader_test\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(ModuleNotFoundError) as exc_info:
        harness_class("broken-harness-v1")

    assert exc_info.value.name == "missing_dependency_for_loader_test"


def test_missing_external_harness_reports_both_import_candidates():
    with pytest.raises(ModuleNotFoundError) as exc_info:
        import_harness("missing-external-harness-v1")

    message = str(exc_info.value)
    assert "normalized module 'missing_external_harness_v1'" in message
    assert "'verifiers.v1.harnesses.missing_external_harness_v1'" in message
    assert "'missing_external_harness_v1'" in message
    assert exc_info.value.__cause__.name == "missing_external_harness_v1"
