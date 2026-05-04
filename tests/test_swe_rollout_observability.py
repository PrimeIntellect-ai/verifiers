from types import SimpleNamespace

from verifiers.envs.experimental.composable.harnesses.opencode import (
    build_install_script,
)
from verifiers.envs.experimental.composable.tasksets.swe.swe_rebench_v2 import (
    _patch_failure_message,
)


def test_patch_failure_message_includes_empty_stderr_context():
    message = _patch_failure_message(
        "test_patch",
        "diff --git a/test.py b/test.py\n",
        [
            (
                "git_apply",
                "git apply /tmp/test_patch.patch",
                "/repo",
                SimpleNamespace(exit_code=1, stdout="", stderr=""),
            )
        ],
    )

    assert "patch_sha256=" in message
    assert "patch_size=" in message
    assert "git_apply_command='git apply /tmp/test_patch.patch'" in message
    assert "git_apply_working_dir='/repo'" in message
    assert "git_apply_stdout='<empty>'" in message
    assert "git_apply_stderr='<empty>'" in message


def test_opencode_install_script_wraps_setup_steps():
    script = build_install_script()

    assert 'echo "[setup] start $name"' in script
    assert 'echo "[setup] end $name exit=$exit_code elapsed_s=$elapsed_s"' in script
    assert 'run_setup_step "apt_dependencies"' in script
    assert 'run_setup_step "ripgrep_install"' in script
    assert 'run_setup_step "download_opencode"' in script
    assert 'run_setup_step "verify_opencode_sha256"' in script
    assert 'run_setup_step "install_opencode"' in script
