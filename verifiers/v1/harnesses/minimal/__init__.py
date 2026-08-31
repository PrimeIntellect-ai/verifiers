from pathlib import Path

from verifiers.v1.harnesses.standalone import inline_mcp_client

PROGRAM_SOURCE = inline_mcp_client(
    (Path(__file__).resolve().parent / "program.py").read_text()
)

__all__ = ["PROGRAM_SOURCE"]
