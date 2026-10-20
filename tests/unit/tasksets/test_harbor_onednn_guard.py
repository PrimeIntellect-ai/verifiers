"""The oneDNN guard turns torch's oneDNN backend off when torch is imported, and
imports nothing before that."""

import shutil
import subprocess
import sys
from pathlib import Path

from verifiers.v1.tasksets.harbor import onednn_guard

FAKE_TORCH = """
class _C:
    mkldnn = True
    @staticmethod
    def _set_mkldnn_enabled(value):
        _C.mkldnn = value
"""


def run(tmp_path: Path, code: str, mode: str) -> str:
    shutil.copyfile(onednn_guard.__file__, tmp_path / f"{onednn_guard.NAME}.py")
    (tmp_path / "torch").mkdir(exist_ok=True)
    (tmp_path / "torch" / "__init__.py").write_text(FAKE_TORCH)
    startup = f"import {onednn_guard.NAME}; " + code
    return subprocess.run(
        [sys.executable, "-c", startup],
        env={"PYTHONPATH": str(tmp_path), "VF_ONEDNN_GUARD": mode},
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def test_importing_torch_turns_its_onednn_backend_off(tmp_path):
    imported = "import torch; print(torch._C.mkldnn)"
    assert run(tmp_path, imported, "1") == "False"
    assert run(tmp_path, imported, "0") == "True"
    # Until something imports torch, the guard has imported nothing.
    assert run(tmp_path, "import sys; print('torch' in sys.modules)", "1") == "False"
