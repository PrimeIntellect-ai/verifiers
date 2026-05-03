import os
import shlex
import subprocess

from verifiers.envs.experimental.composable.harnesses.opencode import (
    build_opencode_run_command,
)


def _extract_script(command: str) -> str:
    return shlex.split(command)[2]


def test_opencode_run_command_uses_agent_workdir_without_prompt_mutation():
    command = build_opencode_run_command(
        agent_workdir="/fallback workdir",
        prompt_path="/task prompt.txt",
        log_path="/logs/agent/opencode.log",
        system_prompt_path="/opencode/system.txt",
    )
    script = _extract_script(command)

    assert 'OPENCODE_WORKDIR="${AGENT_WORKDIR:-}"' in script
    assert "OPENCODE_WORKDIR='/fallback workdir'" in script
    assert 'cd "$OPENCODE_WORKDIR"' in script
    assert "sed -i" not in script
    assert "{agent_workdir}" not in script


def test_opencode_run_command_honors_runtime_agent_workdir(tmp_path):
    home = tmp_path / "home"
    bin_dir = home / ".opencode" / "bin"
    bin_dir.mkdir(parents=True)
    opencode_bin = bin_dir / "opencode"
    opencode_bin.write_text(
        "#!/usr/bin/env bash\necho PWD=$(pwd)\necho ARGS=$*\necho INPUT=$(cat)\n"
    )
    opencode_bin.chmod(0o755)

    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("solve this")
    log_path = tmp_path / "logs" / "opencode.log"
    fallback_workdir = tmp_path / "fallback"
    runtime_workdir = tmp_path / "runtime"

    command = build_opencode_run_command(
        agent_workdir=str(fallback_workdir),
        prompt_path=str(prompt_path),
        log_path=str(log_path),
        system_prompt_path=None,
    )
    env = {
        **os.environ,
        "HOME": str(home),
        "OPENAI_MODEL": "vllm/test-model",
        "AGENT_WORKDIR": str(runtime_workdir),
    }

    result = subprocess.run(
        command, env=env, shell=True, text=True, capture_output=True
    )

    assert result.returncode == 0, result.stderr
    assert runtime_workdir.is_dir()
    assert not fallback_workdir.exists()
    assert f"PWD={runtime_workdir}" in log_path.read_text()
    assert "INPUT=solve this" in log_path.read_text()
