from pathlib import Path

from verifiers.v1.harnesses.utils import compaction, core, mcp
from verifiers.v1.harnesses.utils.launch import bundle_program

PROGRAM_SOURCE = bundle_program(
    (Path(__file__).resolve().parent / "program.py").read_text(),
    mcp,
    compaction,
    core,
)

__all__ = ["PROGRAM_SOURCE"]
