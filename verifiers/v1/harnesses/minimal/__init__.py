from pathlib import Path

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()

__all__ = ["PROGRAM_SOURCE"]
