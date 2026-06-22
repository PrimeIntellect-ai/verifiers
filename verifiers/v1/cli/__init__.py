"""CLI commands. Each exposes a `main()` registered in pyproject's `[project.scripts]` and
runnable via `uv run <name> ...` — a single module (`validate`, `serve`, `init`), or a
subpackage whose entry is `<name>.main` (`eval`)."""
