"""Importable CLI entrypoints for host applications."""

CLI_MODULES = {
    "eval": "verifiers.v1.cli.eval.main",
    "init": "verifiers.v1.cli.init",
    "validate": "verifiers.v1.cli.validate",
    "serve": "verifiers.v1.cli.serve",
    "gepa": "verifiers.scripts.gepa",
}

__all__ = ["CLI_MODULES"]
