import subprocess
import sys

from verifiers.v1.mcp.launch import _locked_command, _venv_command


def test_sandbox_toolset_venv_setup_is_idempotent(tmp_path) -> None:
    venv = tmp_path / ".vf-venv"
    command = _venv_command(str(venv))

    subprocess.run(["sh", "-c", command], check=True)
    marker = venv / "marker"
    marker.write_text("preserved")
    subprocess.run(["sh", "-c", command], check=True)

    assert marker.read_text() == "preserved"


def test_sandbox_toolset_setup_is_serialized(tmp_path) -> None:
    worker = tmp_path / "worker.py"
    output = tmp_path / "output"
    lock = tmp_path / "venv.lock"
    worker.write_text(
        "import pathlib, sys, time\n"
        "path = pathlib.Path(sys.argv[1])\n"
        "with path.open('a') as stream:\n"
        "    stream.write(f'{sys.argv[2]} start\\n')\n"
        "    stream.flush()\n"
        "    time.sleep(0.2)\n"
        "    stream.write(f'{sys.argv[2]} end\\n')\n"
    )

    processes = [
        subprocess.Popen(
            _locked_command(
                str(lock), [sys.executable, str(worker), str(output), name]
            )
        )
        for name in ("first", "second")
    ]
    for process in processes:
        assert process.wait() == 0

    lines = output.read_text().splitlines()
    assert lines in (
        ["first start", "first end", "second start", "second end"],
        ["second start", "second end", "first start", "first end"],
    )
