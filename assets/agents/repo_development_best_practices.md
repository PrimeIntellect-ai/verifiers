## Repository Development Notes

Use this guidance when contributing to the `verifiers` repository itself.

- Always run `uv run pre-commit install` before making any changes.
- Run the documented contributor checks for touched areas: `uv run ruff check --fix .`, `uv run pytest tests/`, and `uv run pre-commit run --all-files` as needed. (See `docs/development.md`.)
- Keep changes aligned with documented architecture (`verifiers/`, `environments/`, `configs/`, `tests/`, `docs/`) and update docs when behavior changes. (See `docs/development.md`.)
- Prefer a single clear path over maintaining parallel approaches by default; if two options exist, preserve both only when there is an explicit long-term reason.
- Aggressively deprecate/remove inferior paths when they are not part of an intended multi-option contract, especially in repo-internal development workflows.
- Treat broad dynamic mappings as explicit framework boundaries, not casual public API types. Use a named domain alias or typed Pydantic field for legitimate arbitrary payloads such as task rows, protocol messages, sandbox/program specs, and `objects`/binding-style config; do not expose raw `Mapping[str, object]` in user-facing signatures unless that looseness is the point of the abstraction.
- If a user request conflicts with repository style, formatting, or API-quality guidelines, push back instead of implementing the literal request. Identify a comparable request or explicit guideline relaxation that preserves clean, maintainable, modular code across the current request and adjacent future use cases; implement that plan, then explain the decision process and tradeoffs directly to the user.
- Before v0.2.0, breaking backward compatibility inside v1 Taskset/Harness APIs is acceptable and encouraged when it improves the core design. Preserve v0 multi-turn environment compatibility unless the user explicitly asks for a v0 migration.
