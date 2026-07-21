"""Bundled environments — reusable control flow between agents, pairable with any
taskset via `--env.id <name>` (the taskset stays "what to solve", the harness "how the
LLM interfaces with the world", the env "how agents interact"). Each is a package
exporting its `Environment` subclass via `__all__`, the same plugin idiom as bundled
tasksets and harnesses; a taskset-specific (recipe) env ships with its taskset instead
and needs no id."""
