# AGENTS.md

## Writing code

- **Code is the source of truth**: before writing anything, read the v1 code for existing helpers, harnesses, and interfaces instead of reinventing them. Prefer verifiers-native task, trace, server, harness, and runtime interfaces over repeated path/import/discovery plumbing in user packages.
- **Minimal config surface**: expose as few knobs as possible, but as many as needed.
- **Prefer the best design over backwards compatibility**: excessive backwards compatibility is a non-goal. If a breaking change to an API surface gets the codebase to the globally optimal design, make the break. Do not keep shims, aliases or legacy code paths to avoid it. List each break and its migration in the PR's `## Breaking` section.
- **Keep tasksets small**: a basic taskset fits in a few dozen idiomatic lines — typed data/task/config classes, `load()`, and decorated scoring on the task. Don't override `Taskset.__init__` (implement `load()`); don't override `Harness.__init__` (use `setup()`).

## Public API

- **Mark internals with an underscore**: give each new module, name or member that users must not use a leading underscore (PEP 8). Names inside an underscore module are internal already, and so are the members of an internal class.
- **Decide each member of a public class by one question**: will code outside verifiers (a taskset, harness, env, judge or toolset, or a program that runs vf) touch it? If yes, it plays one role: **read** (state outside code reads), **call** (an action outside code performs) or **override** (a hook vf calls and a subclass replaces). If no, or if you are unsure, give it an underscore. A call from another vf module or a test does not make a member public. Pydantic validators and serializers are always private.
- **Document every public name and member**: one short line that says what it is or does for the user. If you can only describe it as framework plumbing, it is internal.
- **Export public names explicitly**: add each public name to `__all__` of `verifiers.v1` or of its plugin package, and add a row to `docs/api.md`. `tests/unit/test_api.py` checks that both lists match, that public signatures use only public types, that every public name and member has a docstring, and that validators are private.
- **Import from the public namespace in docs and examples**: use `import verifiers.v1 as vf`, never a path with an underscore.

## Logging

- **Module loggers**: create them with `get_logger(__name__)` from `verifiers.v1._utils.logging`. It prefixes every line with its scope: `[<episode>]` inside an episode, `[<episode>/<trace> <agent>]` inside an agent run, `[<trace>]` for an agent run outside an episode. Never put a trace or episode id in the message itself.
- **Levels by frequency**: INFO is for process lifecycle (servers, pools) and, per episode, `Starting episode`, `Finished episode` and one `Finished rollout` line per agent run. Steps inside a rollout (runtime start/stop, setup, scoring) and per-turn or per-request events are DEBUG. Every failure and retry is a WARNING. ERROR is for a process component that can't continue, never for a rollout's outcome.
- **Message style**: an active sentence with no trailing period (`Started prime runtime abc in 9.8s`), then `key=value` fields. Durations use `format_time()` and read `in 1.2s` or `in 3m 46s`. Error details come last, after ` - `: `Failed to delete prime runtime abc - APIError: ...`. Clip error messages with `clip()`.
- **Retries**: `Retrying <what> in <delay> (attempt <next>/<total>) - <Type>: <message>`.
- **Tracebacks**: attach one for unexpected errors only. For an expected failure (teardown, cleanup, a refused call), pass `exc_info=logger.isEnabledFor(logging.DEBUG)`.
- **Config warnings**: warn once per process, not once per rollout.

## Running code

- **Always use uv**: run code and commands with `uv run`, never raw `python`. Make sure `uv` is installed ([docs](https://docs.astral.sh/uv/getting-started/installation/)).
- **Check tasksets**: check a taskset without a model with `uv run vf-validate <taskset>`. Run a few episodes with `uv run vf-run <taskset> -n 1 -r 1`. Run full model evals with prime-rl's `uv run eval`.
- **Validate TOML first**: validate config `.toml` files before running them.
- **Don't add dependencies**: never add dependencies or optional extras to the top-level `pyproject.toml`.

## Docs

- **Kept intentionally minimal**: `docs/` and `skills/` are deliberately sparse. Don't touch them unless your change breaks their assumptions.
- **Docs reflect `main`, not history**: describe the current state of the codebase only — no removed/legacy fields, migration paths, or "this used to be X" anecdotes.

## Skills

- **Use bundled skills first**: the `release` skill in `skills/` documents the stable-version release workflow, and the `bump` skill documents how to bump a pinned harness version.

## Testing

- **Two suites**: `tests/unit` runs one component in isolation; `tests/integration` runs real evals (a live model, real runtimes) per behavior across the harness x runtime grid in `tests/integration/grid.py`. Prefer an integration cell over a unit test; a unit test earns its place by stating one component's intended behavior, not by asserting an implementation detail. To check your own work, write a temporary script instead of committing new tests.
- **Run the contributor checks**: run `uv run pre-commit install` once, then for touched areas `uv run ruff check --fix .`, `uv run pytest tests/unit`, and `uv run pre-commit run --all-files`.
