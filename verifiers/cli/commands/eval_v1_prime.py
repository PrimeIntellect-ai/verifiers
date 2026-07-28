"""Prime CLI bridge for Verifiers v1 config-driven evaluations.

Prime CLI injects v0 evaluator options before dispatching the Verifiers
plugin.  For a v1 TOML target, translate only the safe shared overrides and
invoke the canonical v1 ``eval @ config.toml`` entrypoint.  Non-v1 targets keep
the existing v0 command unchanged.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path


def _is_v1_config(value: str) -> bool:
    """Return True when *value* points at a TOML file whose top-level
    ``[env.taskset]`` table marks it as a v1 config."""
    path = Path(value)
    if not path.is_file() or path.suffix.lower() != ".toml":
        return False
    try:
        with path.open("rb") as handle:
            raw = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return False
    env = raw.get("env")
    return isinstance(env, dict) and isinstance(env.get("taskset"), dict)


# -- arg translation -----------------------------------------------------------

# Prime-injected v0-only options that must be dropped (never forwarded to v1).
_DROP_VALUE = {
    "--api-base-url",
    "-b",
    "--api-key-var",
    "-k",
    "--provider",
    "-p",
    "--endpoints-path",
    "-e",
    "--header",
    "--header-from-state",
    "--env-dir-path",
}
_DROP_FLAG = {"--save-results", "-s", "--disable-tui"}

# Options that are safe to forward to v1.
_KEEP_VALUE = {"--model", "-m", "--output-dir", "-o"}
_KEEP_FLAG = {"--dry-run", "--verbose", "--no-push"}
_TRANSLATE_FLAG = {"--skip-upload": "--no-push"}

_SHORT_TO_LONG = {"-m": "--model", "-o": "--output-dir"}


def _translated_v1_args(argv: list[str]) -> list[str]:
    """Translate Prime-injected v0 argv into v1 ``eval @ config.toml`` argv."""
    if not argv or not _is_v1_config(argv[0]):
        raise ValueError("v1 translation requires a v1 TOML target")
    result = ["@", str(Path(argv[0]).resolve())]
    args = argv[1:]
    index = 0
    while index < len(args):
        token = args[index]
        # Drop value-taking v0-only flags
        if token in _DROP_VALUE:
            index += 2  # skip flag + value
            continue
        if any(
            token.startswith(f"{flag}=")
            for flag in _DROP_VALUE
            if flag.startswith("--")
        ):
            index += 1
            continue
        # Keep value-taking safe flags (normalize short → long)
        if token in _KEEP_VALUE:
            if index + 1 >= len(args):
                raise SystemExit(f"{token} requires a value")
            normalized = _SHORT_TO_LONG.get(token, token)
            result.extend([normalized, args[index + 1]])
            index += 2
            continue
        if any(token.startswith(f"{flag}=") for flag in ("--model", "--output-dir")):
            result.append(token)
            index += 1
            continue
        # Keep bare flags and translate equivalent v0 safety flags.
        if token in _KEEP_FLAG:
            result.append(token)
            index += 1
            continue
        if token in _TRANSLATE_FLAG:
            translated = _TRANSLATE_FLAG[token]
            if translated not in result:
                result.append(translated)
            index += 1
            continue
        # Drop v0-only bare flags
        if token in _DROP_FLAG:
            index += 1
            continue
        raise SystemExit(f"unsupported prime eval v1 bridge argument: {token}")
    return result


# -- dispatch -----------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Entry point: route to v1 eval for v1 TOML targets, legacy eval otherwise."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args and _is_v1_config(args[0]):
        from verifiers.v1.cli.eval.main import main as v1_main

        v1_main(_translated_v1_args(args))
        return
    from verifiers.scripts.eval import main as legacy_main

    old = sys.argv
    try:
        sys.argv = [old[0], *args]
        legacy_main()
    finally:
        sys.argv = old


if __name__ == "__main__":
    main()
