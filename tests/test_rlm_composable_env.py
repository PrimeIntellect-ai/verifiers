"""Tests for RLM harness integration with ComposableEnv.

Validates that rlm_harness() produces a Harness with the correct metrics
fields and that the install script is generated correctly.
"""

import importlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock, call

import pytest

import verifiers as vf
from verifiers.envs.experimental.composable import (
    ComposableEnv,
    Harness,
    SandboxSpec,
    SandboxTaskSet,
)
from verifiers.envs.experimental.composable.harnesses import rlm as rlm_module
from verifiers.envs.experimental.composable.harnesses.rlm import (
    build_install_script,
    rlm_harness,
    resolve_local_checkout,
)


class MockSandboxRubric(vf.Rubric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_reward_func(self.solved)

    async def solved(self, state, **kwargs) -> float:
        return 1.0 if state.get("test_output") == "PASS" else 0.0


class MockSandboxTaskSet(SandboxTaskSet):
    def get_instruction(self, info):
        return f"Fix bug #{info.get('id', 0)}"

    def get_sandbox_spec(self, info):
        return SandboxSpec(image="python:3.11-slim", cpu_cores=2, memory_gb=2)

    def get_rubric(self):
        return MockSandboxRubric()

    def get_workdir(self, info):
        return "/testbed"


class MockSandboxTaskSetWithSkills(MockSandboxTaskSet):
    """Skills auto-discovered via get_skills_dir() — module monkeypatched in tests."""

    pass


def _make_dataset(n=3):
    from datasets import Dataset

    return Dataset.from_dict(
        {
            "info": [{"id": i, "question": f"q{i}"} for i in range(n)],
            "answer": ["" for _ in range(n)],
        }
    )


def _make_temp_taskset_package(tmp_path, monkeypatch, *, with_skills: bool):
    package_name = f"rlm_fixture_{tmp_path.name.replace('-', '_')}"
    pkg_dir = tmp_path / package_name
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "taskset_mod.py").write_text("MARKER = 1\n")

    if with_skills:
        skill_dir = pkg_dir / "skills" / "demo"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: demo\n---\n")
        (skill_dir / "pyproject.toml").write_text(
            "[project]\nname = 'rlm-skill-demo'\nversion = '0.0.0'\n"
        )

    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    mod = importlib.import_module(f"{package_name}.taskset_mod")
    monkeypatch.setattr(MockSandboxTaskSetWithSkills, "__module__", mod.__name__)
    return mod


def _make_git_checkout(target: Path) -> Path:
    checkout = target
    checkout.mkdir()
    (checkout / "install.sh").write_text("#!/usr/bin/env bash\n")
    (checkout / "pyproject.toml").write_text("[project]\nname='rlm'\nversion='0.0.0'\n")
    subprocess.run(["git", "init", "-b", "main"], cwd=checkout, check=True)
    subprocess.run(
        ["git", "add", "install.sh", "pyproject.toml"], cwd=checkout, check=True
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Codex",
            "-c",
            "user.email=codex@example.com",
            "commit",
            "-m",
            "init",
        ],
        cwd=checkout,
        check=True,
    )
    return checkout


# ── RLM harness ──────────────────────────────────────────────────────────


def test_rlm_harness_install_script_requires_uploaded_checkout():
    script = build_install_script()
    assert 'test -f "$RLM_CHECKOUT_PATH/install.sh"' in script
    assert "git clone" not in script
    assert 'bash "$RLM_CHECKOUT_PATH/install.sh"' in script


def test_rlm_harness_sets_metrics_fields(tmp_path):
    harness = rlm_harness(local_checkout=_make_git_checkout(tmp_path / "rlm"))
    assert harness.metrics_path == "{workdir}/.rlm/sessions/*/meta.json"
    assert harness.metrics_key == "metrics"
    assert harness.metrics_prefix == "rlm_"


def test_rlm_harness_sets_skills_path(tmp_path):
    harness = rlm_harness(local_checkout=_make_git_checkout(tmp_path / "rlm"))
    assert harness.skills_path == "/task/rlm-skills"


def test_resolve_local_checkout_validates_explicit_path(tmp_path):
    checkout = _make_git_checkout(tmp_path / "rlm")

    resolved = resolve_local_checkout(checkout)

    assert resolved == checkout.resolve()


def test_rlm_harness_uploads_explicit_local_checkout(tmp_path):
    checkout = _make_git_checkout(tmp_path / "rlm")

    harness = rlm_harness(local_checkout=checkout)

    assert harness.get_upload_dirs is not None
    assert harness.get_upload_dirs() == {"rlm_checkout": checkout.resolve()}
    assert harness.upload_dir_mapping == {"rlm_checkout": "/tmp/rlm-checkout"}


def test_resolve_local_checkout_materializes_host_cache(tmp_path):
    source_checkout = _make_git_checkout(tmp_path / "rlm-source")
    checkout_dir = tmp_path / "checkout-root" / "rlm"

    resolved = resolve_local_checkout(
        local_checkout=checkout_dir,
        rlm_repo_url=str(source_checkout),
        rlm_branch="main",
    )

    assert resolved == checkout_dir.resolve()
    assert (checkout_dir / ".git").is_dir()
    assert (checkout_dir / "install.sh").is_file()
    assert (checkout_dir / "pyproject.toml").is_file()


def test_rlm_harness_uses_default_host_cache_when_local_checkout_unspecified(
    tmp_path, monkeypatch
):
    source_checkout = _make_git_checkout(tmp_path / "rlm-source")
    monkeypatch.setattr(
        rlm_module,
        "DEFAULT_RLM_LOCAL_CHECKOUT_CACHE_ROOT",
        tmp_path / "cache-root",
    )

    harness = rlm_harness(
        rlm_repo_url=str(source_checkout),
        rlm_branch="main",
    )

    assert harness.get_upload_dirs is not None
    upload_checkout = harness.get_upload_dirs()["rlm_checkout"]
    assert isinstance(upload_checkout, Path)
    assert upload_checkout.is_dir()
    assert upload_checkout.name.startswith("rlm-source-main-")
    assert harness.upload_dir_mapping == {"rlm_checkout": "/tmp/rlm-checkout"}


def test_rlm_harness_always_uploads_checkout(tmp_path, monkeypatch):
    source_checkout = _make_git_checkout(tmp_path / "rlm-source")
    monkeypatch.setattr(
        rlm_module,
        "DEFAULT_RLM_LOCAL_CHECKOUT_CACHE_ROOT",
        tmp_path / "cache-root",
    )

    harness = rlm_harness(
        rlm_repo_url=str(source_checkout),
        rlm_branch="main",
    )

    assert harness.get_upload_dirs is not None
    assert harness.upload_dir_mapping is not None


def test_resolve_local_checkout_redacts_gh_token_on_clone_failure(tmp_path, monkeypatch):
    failing_checkout = tmp_path / "checkout-root" / "rlm"
    token = "super/secret token"
    quoted_token = "super%2Fsecret%20token"

    def _raise_clone_error(*args, **kwargs):
        raise subprocess.CalledProcessError(
            128,
            args[0],
            stderr=(
                "fatal: could not read from "
                f"https://{quoted_token}@github.com/PrimeIntellect-ai/rlm.git"
            ),
        )

    monkeypatch.setattr(rlm_module.subprocess, "run", _raise_clone_error)

    with pytest.raises(RuntimeError) as exc_info:
        resolve_local_checkout(
            local_checkout=failing_checkout,
            rlm_repo_url="github.com/PrimeIntellect-ai/rlm.git",
            rlm_branch="main",
            gh_token=token,
        )

    message = str(exc_info.value)
    assert token not in message
    assert "<redacted>" in message


# ── install_env ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rlm_install_runs_without_skills(tmp_path, monkeypatch):
    _make_temp_taskset_package(tmp_path, monkeypatch, with_skills=False)
    taskset = MockSandboxTaskSetWithSkills(dataset=_make_dataset(), name="test")
    env = ComposableEnv(
        taskset=taskset,
        harness=Harness(
            run_command="true",
            install_script="install-rlm",
            instruction_path="/tmp/with space/prompt.txt",
            system_prompt="system",
            system_prompt_path="/tmp/other path/system.txt",
            skills_path="/task/rlm-skills",
        ),
        install_env={"GH_TOKEN": "secret"},
    )
    env.sandbox_client = SimpleNamespace(
        execute_command=AsyncMock(
            return_value=SimpleNamespace(stdout="", stderr="", exit_code=0)
        ),
        teardown=lambda: None,
    )
    env.taskset.setup = AsyncMock()
    env.upload_content = AsyncMock()
    env.upload_file = AsyncMock()

    await env.post_sandbox_setup({"sandbox_id": "sbx", "info": {"id": 0}})

    assert env.upload_file.await_count == 0
    assert env.sandbox_client.execute_command.await_args_list == [
        call(
            "sbx",
            "mkdir -p '/tmp/other path' '/tmp/with space'",
            timeout=10,
        ),
        call(
            "sbx",
            "install-rlm",
            timeout=300,
            env={"GH_TOKEN": "secret"},
        ),
    ]


# ── Skills upload ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rlm_uploads_skills_before_install(tmp_path, monkeypatch):
    _make_temp_taskset_package(tmp_path, monkeypatch, with_skills=True)
    taskset = MockSandboxTaskSetWithSkills(dataset=_make_dataset(), name="test")
    env = ComposableEnv(
        taskset=taskset,
        harness=Harness(
            run_command="true",
            install_script="install-rlm",
            skills_path="/task/rlm-skills",
        ),
    )
    env.sandbox_client = SimpleNamespace(
        execute_command=AsyncMock(
            return_value=SimpleNamespace(stdout="", stderr="", exit_code=0)
        ),
        teardown=lambda: None,
    )
    env.taskset.setup = AsyncMock()
    env.upload_content = AsyncMock()
    env.upload_file = AsyncMock()

    await env.post_sandbox_setup({"sandbox_id": "sbx", "info": {"id": 0}})

    env.upload_file.assert_awaited_once()
    upload_call = env.upload_file.await_args
    assert upload_call.args[0] == "sbx"
    assert upload_call.args[1] == "/tmp/_upload_task_rlm-skills.tar.gz"

    install_call = env.sandbox_client.execute_command.await_args_list[-1]
    assert install_call == call("sbx", "install-rlm", timeout=300)
    extract_call = env.sandbox_client.execute_command.await_args_list[1]
    assert extract_call == call(
        "sbx",
        "mkdir -p /task && tar -xzf /tmp/_upload_task_rlm-skills.tar.gz -C / && rm -f /tmp/_upload_task_rlm-skills.tar.gz",
        timeout=60,
    )


# ── RLM metrics via harness fields ──────────────────────────────────────


@pytest.mark.asyncio
async def test_rlm_collects_logs_and_metrics(tmp_path):
    taskset = MockSandboxTaskSet(dataset=_make_dataset(), name="test")
    metrics = {
        "turns": 3,
        "stop_reason": "done",
        "prompt_tokens": 100,
        "completion_tokens": 25,
    }
    harness = rlm_harness(local_checkout=_make_git_checkout(tmp_path / "rlm"))
    env = ComposableEnv(
        taskset=taskset,
        harness=Harness(
            run_command=harness.run_command,
            log_path="/tmp/log dir/agent.log",
            metrics_path=harness.metrics_path,
            metrics_key=harness.metrics_key,
            metrics_prefix=harness.metrics_prefix,
        ),
    )
    env.sandbox_client = SimpleNamespace(
        execute_command=AsyncMock(
            side_effect=[
                SimpleNamespace(stdout="agent log\n", stderr="", exit_code=0),
                SimpleNamespace(
                    stdout=json.dumps({"metrics": metrics}),
                    stderr="",
                    exit_code=0,
                ),
            ]
        ),
        teardown=lambda: None,
    )

    state = {
        "sandbox_id": "sbx",
        "info": {"id": 0},
        "timing": {"total_ms": 0},
        "trajectory": [],
    }

    await env.post_rollout(state)

    assert env.sandbox_client.execute_command.await_args_list == [
        call(
            "sbx",
            "cat '/tmp/log dir/agent.log' 2>/dev/null || echo '<no logs>'",
            working_dir=None,
        ),
        call(
            "sbx",
            'f=$(ls /testbed/.rlm/sessions/*/meta.json 2>/dev/null | head -1) && cat "$f" || echo "{}"',
            working_dir=None,
        ),
    ]
    assert state["agent_logs"] == "agent log"
    assert state["rlm_turns"] == 3
    assert state["rlm_stop_reason"] == "done"
    assert state["rlm_prompt_tokens"] == 100
    assert state["rlm_completion_tokens"] == 25
