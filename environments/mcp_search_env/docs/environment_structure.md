# Environment Structure

Source: `docs/overview.md`.

`prime env init my-env` creates a self-contained environment module under
`environments/my_env/`. The generated package includes the main implementation
file, `pyproject.toml` dependency metadata, and a README.

Environment modules expose `load_environment(...)`, which returns a Verifiers
environment object and accepts any environment-specific arguments.
