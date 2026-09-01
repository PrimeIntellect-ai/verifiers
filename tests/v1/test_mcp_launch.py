import subprocess

from verifiers.v1.mcp.launch import _venv_command


def test_sandbox_toolset_venv_setup_is_idempotent(tmp_path) -> None:
    venv = tmp_path / ".vf-venv"
    command = _venv_command(str(venv))

    subprocess.run(["sh", "-c", command], check=True)
    marker = venv / "marker"
    marker.write_text("preserved")
    subprocess.run(["sh", "-c", command], check=True)

    assert marker.read_text() == "preserved"
