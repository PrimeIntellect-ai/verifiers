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
    composable_env as composable_env_module,
)
from verifiers.envs.experimental.composable import (
    ComposableEnv,
    Harness,
    SandboxSpec,
    SandboxTaskSet,
)
from verifiers.envs.experimental.composable.harnesses import rlm as rlm_module
from verifiers.envs.experimental.utils import (
    git_checkout_cache as git_checkout_cache_module,
)
from verifiers.envs.experimental.utils.file_locks import (
    exclusive_path_lock,
    shared_path_lock,
)
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


def _git_head_commit(checkout: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().lower()


def _commit_file(checkout: Path, name: str, content: str) -> str:
    (checkout / name).write_text(content)
    subprocess.run(["git", "add", name], cwd=checkout, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Codex",
            "-c",
            "user.email=codex@example.com",
            "commit",
            "-m",
            f"update {name}",
        ],
        cwd=checkout,
        check=True,
    )
    return _git_head_commit(checkout)


# ── RLM harness ──────────────────────────────────────────────────────────


def test_rlm_harness_install_script_requires_uploaded_checkout():
    script = build_install_script()
    assert 'test -f "$RLM_CHECKOUT_PATH/install.sh"' in script
    assert "git clone" not in script
    assert 'bash "$RLM_CHECKOUT_PATH/install.sh"' in script
    # Shim setup is a post-install step now, NOT part of install_script.
    assert "/usr/local/bin/" not in script


def test_rlm_harness_blocks_git_by_default(tmp_path):
    """By default RLM uploads a sh shim that refuses on any invocation
    and exits 1. The shim is staged at /tmp and moved to
    $HOME/.local/bin/git by the post-install script — that dir is on
    the agent's PATH (via the run_command's ``export PATH=...``) but
    NOT on the container's default PATH, so scoring's
    ``execute_command`` calls (e.g. ``git apply`` in eval scripts) keep
    resolving to the real ``/usr/bin/git``."""
    checkout = _make_git_checkout(tmp_path / "rlm")
    harness = rlm_harness(local_checkout=checkout)

    assert harness.post_install_uploads is not None
    assert set(harness.post_install_uploads.keys()) == {"/tmp/__rlm_git_shim"}

    shim = harness.post_install_uploads["/tmp/__rlm_git_shim"]
    assert shim.startswith("#!/bin/sh\n")
    assert "Bash command 'git' is not allowed." in shim
    assert "exit 1" in shim

    script = harness.post_install_script
    assert script is not None
    # Shim must end up at $HOME/.local/bin/git, executable, with parent
    # dir created if missing.
    assert 'mkdir -p "$HOME/.local/bin"' in script
    assert 'mv /tmp/__rlm_git_shim "$HOME/.local/bin/git"' in script
    assert 'chmod +x "$HOME/.local/bin/git"' in script
    # The shim must NOT live anywhere on the container's default PATH —
    # /usr/local/bin/git would shadow the real git for scoring's
    # execute_command shells (which don't get $HOME/.local/bin
    # prepended) and break ``git apply`` of test_patches in eval.sh.
    assert "/usr/local/bin/git" not in script


def test_rlm_harness_allow_git_uploads_nothing(tmp_path):
    """``allow_git=True`` skips the shim entirely — no uploads, no
    post-install script. Matches opencode's ``allow_git=True`` default:
    the caller has opted in, so don't get in their way."""
    checkout = _make_git_checkout(tmp_path / "rlm")
    harness = rlm_harness(local_checkout=checkout, allow_git=True)

    assert harness.post_install_uploads is None
    assert harness.post_install_script is None


def test_rlm_harness_uses_explicit_local_checkout(tmp_path):
    checkout = _make_git_checkout(tmp_path / "rlm")
    harness = rlm_harness(local_checkout=checkout)

    assert harness.get_upload_dirs() == {"rlm_checkout": checkout.resolve()}
    assert harness.upload_dir_mapping == {"rlm_checkout": "/tmp/rlm-checkout"}
    assert harness.metrics_path == "{workdir}/.rlm/sessions/*/meta.json"
    assert harness.metrics_key == "metrics"
    assert harness.metrics_prefix == "rlm_"
    assert harness.skills_path == "/task/rlm-skills"


def test_resolve_local_checkout_rejects_missing_explicit_path(tmp_path):
    missing_checkout = tmp_path / "missing-rlm"

    with pytest.raises(ValueError, match="not a directory"):
        resolve_local_checkout(missing_checkout)


def test_resolve_local_checkout_materializes_host_cache_for_named_ref(
    tmp_path, monkeypatch
):
    source_checkout = _make_git_checkout(tmp_path / "rlm-source")
    source_commit = _git_head_commit(source_checkout)
    monkeypatch.setattr(
        rlm_module,
        "DEFAULT_RLM_LOCAL_CHECKOUT_CACHE_ROOT",
        tmp_path / "cache-root",
    )

    resolved = resolve_local_checkout(
        rlm_repo_url=str(source_checkout),
        rlm_ref="main",
    )

    assert resolved.name == source_commit
    assert resolved.is_dir()
    assert (resolved / ".git").exists()
    assert (resolved / "install.sh").is_file()
    assert (resolved / "pyproject.toml").is_file()


def test_rlm_harness_uses_default_host_cache_when_local_checkout_unspecified(
    tmp_path, monkeypatch
):
    source_checkout = _make_git_checkout(tmp_path / "rlm-source")
    source_commit = _git_head_commit(source_checkout)
    monkeypatch.setattr(
        rlm_module,
        "DEFAULT_RLM_LOCAL_CHECKOUT_CACHE_ROOT",
        tmp_path / "cache-root",
    )

    harness = rlm_harness(
        rlm_repo_url=str(source_checkout),
        rlm_ref="main",
    )

    assert harness.get_upload_dirs is not None
    upload_checkout = harness.get_upload_dirs()["rlm_checkout"]
    assert isinstance(upload_checkout, Path)
    assert upload_checkout.is_dir()
    assert upload_checkout.name == source_commit
    assert harness.upload_dir_mapping == {"rlm_checkout": "/tmp/rlm-checkout"}


def test_rlm_harness_memoizes_resolved_checkout_per_instance(tmp_path, monkeypatch):
    source_checkout = _make_git_checkout(tmp_path / "rlm-source")
    second_commit = _commit_file(source_checkout, "README.md", "updated\n")
    monkeypatch.setattr(
        rlm_module,
        "DEFAULT_RLM_LOCAL_CHECKOUT_CACHE_ROOT",
        tmp_path / "cache-root",
    )

    harness = rlm_harness(
        rlm_repo_url=str(source_checkout),
        rlm_ref="main",
    )
    first_upload_checkout = harness.get_upload_dirs()["rlm_checkout"]

    _commit_file(source_checkout, "extra.txt", "third\n")
    second_upload_checkout = harness.get_upload_dirs()["rlm_checkout"]

    assert isinstance(first_upload_checkout, Path)
    assert first_upload_checkout.name == second_commit
    assert second_upload_checkout == first_upload_checkout


def test_rlm_harness_resolves_new_commit_for_new_instance(tmp_path, monkeypatch):
    source_checkout = _make_git_checkout(tmp_path / "rlm-source")
    first_commit = _git_head_commit(source_checkout)
    monkeypatch.setattr(
        rlm_module,
        "DEFAULT_RLM_LOCAL_CHECKOUT_CACHE_ROOT",
        tmp_path / "cache-root",
    )

    first_harness = rlm_harness(
        rlm_repo_url=str(source_checkout),
        rlm_ref="main",
    )
    first_checkout = first_harness.get_upload_dirs()["rlm_checkout"]
    assert isinstance(first_checkout, Path)
    assert first_checkout.name == first_commit

    second_commit = _commit_file(source_checkout, "README.md", "updated\n")

    second_harness = rlm_harness(
        rlm_repo_url=str(source_checkout),
        rlm_ref="main",
    )
    second_checkout = second_harness.get_upload_dirs()["rlm_checkout"]
    assert isinstance(second_checkout, Path)
    assert second_checkout.name == second_commit
    assert second_checkout != first_checkout
    assert not first_checkout.exists()


def test_rlm_harness_skips_pruning_locked_stale_checkout(tmp_path, monkeypatch):
    source_checkout = _make_git_checkout(tmp_path / "rlm-source")
    first_commit = _git_head_commit(source_checkout)
    monkeypatch.setattr(
        rlm_module,
        "DEFAULT_RLM_LOCAL_CHECKOUT_CACHE_ROOT",
        tmp_path / "cache-root",
    )

    first_harness = rlm_harness(
        rlm_repo_url=str(source_checkout),
        rlm_ref="main",
    )
    first_checkout = first_harness.get_upload_dirs()["rlm_checkout"]
    assert isinstance(first_checkout, Path)
    assert first_checkout.name == first_commit

    second_commit = _commit_file(source_checkout, "README.md", "updated\n")

    with shared_path_lock(first_checkout, suffix=".upload.lock"):
        second_harness = rlm_harness(
            rlm_repo_url=str(source_checkout),
            rlm_ref="main",
        )
        second_checkout = second_harness.get_upload_dirs()["rlm_checkout"]
        assert isinstance(second_checkout, Path)
        assert second_checkout.name == second_commit
        assert first_checkout.exists()

    third_harness = rlm_harness(
        rlm_repo_url=str(source_checkout),
        rlm_ref="main",
    )
    third_checkout = third_harness.get_upload_dirs()["rlm_checkout"]
    assert third_checkout == second_checkout
    assert not first_checkout.exists()


def test_resolve_local_checkout_redacts_gh_token_on_clone_failure(
    tmp_path, monkeypatch
):
    token = "super/secret token"
    quoted_token = "super%2Fsecret%20token"
    monkeypatch.setattr(
        rlm_module,
        "DEFAULT_RLM_LOCAL_CHECKOUT_CACHE_ROOT",
        tmp_path / "cache-root",
    )

    def _raise_clone_error(*args, **kwargs):
        raise subprocess.CalledProcessError(
            128,
            args[0],
            stderr=(
                "fatal: could not read from "
                f"https://{quoted_token}@github.com/PrimeIntellect-ai/rlm.git"
            ),
        )

    monkeypatch.setattr(git_checkout_cache_module.subprocess, "run", _raise_clone_error)

    with pytest.raises(RuntimeError) as exc_info:
        resolve_local_checkout(
            rlm_repo_url="github.com/PrimeIntellect-ai/rlm.git",
            rlm_ref="main",
            gh_token=token,
        )

    message = str(exc_info.value)
    assert token not in message
    assert "<redacted>" in message


def test_resolve_local_checkout_materializes_host_cache_for_exact_commit(
    tmp_path, monkeypatch
):
    source_checkout = _make_git_checkout(tmp_path / "rlm-source")
    second_commit = _commit_file(source_checkout, "README.md", "updated\n")
    monkeypatch.setattr(
        rlm_module,
        "DEFAULT_RLM_LOCAL_CHECKOUT_CACHE_ROOT",
        tmp_path / "cache-root",
    )

    resolved = resolve_local_checkout(
        rlm_repo_url=str(source_checkout),
        rlm_ref=second_commit,
    )

    assert resolved.name == second_commit
    assert resolved.is_dir()


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


@pytest.mark.asyncio
async def test_post_install_uploads_and_script_run_after_install():
    """Post-install hooks: files upload AFTER install_script finishes,
    then post_install_script runs. Failure on either path is fatal."""
    taskset = MockSandboxTaskSet(dataset=_make_dataset(), name="test")
    env = ComposableEnv(
        taskset=taskset,
        harness=Harness(
            run_command="true",
            install_script="install-agent",
            post_install_uploads={"/usr/local/lib/execute_bash.py": "PYTHON SOURCE"},
            post_install_script="install-shims",
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

    # Install script must have already run before the post-install upload.
    install_cmd_idx = next(
        i
        for i, c in enumerate(env.sandbox_client.execute_command.await_args_list)
        if c.args[1] == "install-agent"
    )
    post_install_cmd_idx = next(
        i
        for i, c in enumerate(env.sandbox_client.execute_command.await_args_list)
        if c.args[1] == "install-shims"
    )
    assert install_cmd_idx < post_install_cmd_idx

    # The upload happens between install and post-install script. We can
    # check it was called with the right args; ordering is guaranteed by
    # the sequential awaits in ``post_sandbox_setup``.
    upload_calls = [
        call
        for call in env.upload_content.await_args_list
        if call.args[2] == "/usr/local/lib/execute_bash.py"
    ]
    assert len(upload_calls) == 1
    assert upload_calls[0].args[0] == "sbx"
    assert upload_calls[0].args[1] == "PYTHON SOURCE"


@pytest.mark.asyncio
async def test_post_install_script_failure_raises():
    import verifiers as vf

    taskset = MockSandboxTaskSet(dataset=_make_dataset(), name="test")
    env = ComposableEnv(
        taskset=taskset,
        harness=Harness(
            run_command="true",
            post_install_script="exit 7",
        ),
    )
    env.sandbox_client = SimpleNamespace(
        execute_command=AsyncMock(
            return_value=SimpleNamespace(stdout="boom", stderr="", exit_code=7)
        ),
        teardown=lambda: None,
    )
    env.taskset.setup = AsyncMock()
    env.upload_content = AsyncMock()
    env.upload_file = AsyncMock()

    with pytest.raises(vf.SandboxError, match="Post-install failed"):
        await env.post_sandbox_setup({"sandbox_id": "sbx", "info": {"id": 0}})


def test_build_dir_archive_holds_shared_lock_for_local_path(tmp_path):
    taskset = MockSandboxTaskSet(dataset=_make_dataset(), name="test")
    env = ComposableEnv(
        taskset=taskset,
        harness=Harness(run_command="true"),
    )
    local_source = tmp_path / "srcdir"
    local_source.mkdir()
    (local_source / "file.txt").write_text("hello\n")

    class _FakeTar:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def add(self, path, arcname):
            assert Path(path) == local_source
            with pytest.raises(BlockingIOError):
                with exclusive_path_lock(
                    local_source,
                    suffix=".upload.lock",
                    nonblocking=True,
                ):
                    pass

    original_open = composable_env_module.tarfile.open
    composable_env_module.tarfile.open = lambda *args, **kwargs: _FakeTar()
    try:
        tar_path = env._build_dir_archive(local_source, "/task/srcdir")
    finally:
        composable_env_module.tarfile.open = original_open

    tar_path.unlink(missing_ok=True)


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
    assert state["_harness_metrics"] == {
        "rlm_turns": 3.0,
        "rlm_prompt_tokens": 100.0,
        "rlm_completion_tokens": 25.0,
    }
    await env.rubric.cleanup(state)
    assert state["metrics"] == {
        "rlm_turns": 3.0,
        "rlm_prompt_tokens": 100.0,
        "rlm_completion_tokens": 25.0,
    }


@pytest.mark.asyncio
async def test_rlm_harness_metrics_rubric_does_not_crash_scoring(tmp_path):
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
            metrics_path=harness.metrics_path,
            metrics_key=harness.metrics_key,
            metrics_prefix=harness.metrics_prefix,
        ),
    )
    env.sandbox_client = SimpleNamespace(
        execute_command=AsyncMock(
            return_value=SimpleNamespace(
                stdout=json.dumps({"metrics": metrics}),
                stderr="",
                exit_code=0,
            )
        ),
        teardown=lambda: None,
    )

    state = {
        "sandbox_id": "sbx",
        "info": {"id": 0},
        "prompt": [{"role": "user", "content": "Fix bug #0"}],
        "completion": [{"role": "assistant", "content": "done"}],
        "task": "test",
        "answer": "",
        "timing": {"total_ms": 0},
        "trajectory": [],
        "test_output": "PASS",
    }

    await env.post_rollout(state)
    await env.rubric.score_rollout(state)
    await env.rubric.cleanup(state)

    assert state["reward"] == 1.0
    assert state["metrics"]["solved"] == 1.0
    assert state["metrics"]["rlm_turns"] == 3.0
    assert state["metrics"]["rlm_prompt_tokens"] == 100.0
    assert state["metrics"]["rlm_completion_tokens"] == 25.0


def test_rlm_harness_env_vars_static_int():
    """``summarize_at_tokens`` as an int → constant ``RLM_SUMMARIZE_AT_TOKENS``."""
    harness = rlm_harness(summarize_at_tokens=4000)

    state_a = {"input": {"prompt": "task A"}, "prompt": "task A"}
    state_b = {"input": {"prompt": "task B"}, "prompt": "task B"}
    env_a = harness.environment_vars(state_a)
    env_b = harness.environment_vars(state_b)
    assert env_a["RLM_SUMMARIZE_AT_TOKENS"] == "4000"
    assert env_b["RLM_SUMMARIZE_AT_TOKENS"] == "4000"


def test_rlm_harness_env_vars_range_is_seeded_per_prompt():
    """``summarize_at_tokens`` as ``(lo, hi)`` → per-prompt seeded draw.

    Same prompt → same int (group coherence). Different prompts and
    different ``random_seed`` values → different ints (uses 10**9-wide
    range so collision odds are ~10^-9).
    """
    harness = rlm_harness(summarize_at_tokens=(1, 10**9), random_seed=42)

    state_a = {"input": {"prompt": "task A"}, "prompt": "task A"}
    state_b = {"input": {"prompt": "task B"}, "prompt": "task B"}

    # Group coherence: same prompt → same draw, in range.
    a1 = int(harness.environment_vars(state_a)["RLM_SUMMARIZE_AT_TOKENS"])
    a2 = int(harness.environment_vars(state_a)["RLM_SUMMARIZE_AT_TOKENS"])
    assert a1 == a2
    assert 1 <= a1 <= 10**9

    # Cross-group variation: different prompts → different draws.
    b = int(harness.environment_vars(state_b)["RLM_SUMMARIZE_AT_TOKENS"])
    assert a1 != b

    # Cross-run variation: different random_seed → different draws.
    harness2 = rlm_harness(summarize_at_tokens=(1, 10**9), random_seed=999)
    a_seed999 = int(harness2.environment_vars(state_a)["RLM_SUMMARIZE_AT_TOKENS"])
    assert a1 != a_seed999


def test_rlm_harness_rejects_bad_summarize_at_tokens():
    """Bad shapes raise at harness-build time, not per-rollout."""
    for bad in [0, -1, True, (1, 2, 3), (1000, 500), (-1, 100)]:
        with pytest.raises((ValueError, TypeError)):
            rlm_harness(summarize_at_tokens=bad)
