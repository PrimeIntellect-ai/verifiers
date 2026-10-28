"""A harness's in-box install is not bound by what the task image sets for its solver.

Abundant ProgramBench images (2026-10-28) keep the solver off npm (ENV
NPM_CONFIG_OFFLINE=true, NPM_CONFIG_REGISTRY=http://127.0.0.1:9) and one deletes
Python's `decimal` (the solver writes decimal arithmetic itself): the OpenCode install
failed with ENOTCACHED, and the ACP runner's `import pydantic` failed on the image's
Python, on every draw."""

import os
import subprocess
import sys

import pytest

from verifiers.v1.harnesses.claude_code import harness as claude_code
from verifiers.v1.harnesses.codex import harness as codex
from verifiers.v1.harnesses.deepseek import harness as deepseek
from verifiers.v1.harnesses.kimi_code import harness as kimi_code
from verifiers.v1.harnesses.node import NPM_ENV
from verifiers.v1.harnesses.opencode import harness as opencode
from verifiers.v1.harnesses.pi import harness as pi
from verifiers.v1.runtimes.runtime import _SOUND_PYTHON


@pytest.mark.parametrize(
    "script",
    [
        opencode.INSTALL,
        pi.INSTALL,
        kimi_code.INSTALL,
        deepseek.INSTALL,
        codex.INSTALL,
        claude_code.ACP_INSTALL,
    ],
)
def test_every_npm_install_runs_free_of_the_task_npm_config(script):
    assert NPM_ENV in script
    assert script.index(NPM_ENV) < script.index("npm install")


def test_the_install_drops_the_image_npm_settings_for_its_own():
    image = {
        "NPM_CONFIG_OFFLINE": "true",
        "NPM_CONFIG_REGISTRY": "http://127.0.0.1:9",
        "npm_config_cache_min": "9999999",
        "Npm_Config_Only": "x",
    }
    shown = subprocess.run(
        ["sh", "-ec", NPM_ENV + "env"],
        env={"PATH": os.environ["PATH"], "HOME": "/nonexistent", **image},
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    npm = sorted(line for line in shown if line.lower().startswith("npm_config_"))
    assert npm == [
        "npm_config_cache=/var/tmp/vf-npm/cache",
        "npm_config_globalconfig=/var/tmp/vf-npm/no-global-npmrc",
        "npm_config_offline=false",
        "npm_config_prefer_offline=false",
        "npm_config_registry=https://registry.npmjs.org/",
        "npm_config_userconfig=/var/tmp/vf-npm/no-user-npmrc",
    ]


@pytest.mark.parametrize("module", ["decimal", "ctypes", "ssl", "unicodedata"])
def test_an_interpreter_missing_a_stdlib_module_its_scripts_need_is_unsound(module):
    """Such an interpreter fails the check, so the script runs on uv's own Python."""
    check = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            f"import sys; sys.modules[{module!r}] = None; exec({_SOUND_PYTHON!r})",
        ],
        capture_output=True,
        check=False,
    )
    assert check.returncode != 0
    subprocess.run([sys.executable, "-I", "-c", _SOUND_PYTHON], check=True)
