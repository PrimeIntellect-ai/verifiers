# dspy-flights

Minimal v1 environment for a third-party DSPy flight-support program.

`load_harness()` uses a sandboxed Python `program.fn` entrypoint. v1 resolves
this package from `pyproject.toml`, installs it in the program sandbox, and then
runs `dspy_flights:run_dspy_flight_program` with normal package dependencies.
