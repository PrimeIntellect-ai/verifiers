import sys
from collections.abc import Sequence


SETUP_DEPRECATION_MESSAGE = """vf-setup is deprecated.

Upgrade the Prime CLI, then run:

    prime lab setup
"""


def warn_setup_deprecated() -> None:
    """Print the migration path for the old setup command."""
    print(SETUP_DEPRECATION_MESSAGE, file=sys.stderr, end="")


def run_setup(*_: object, **__: object) -> int:
    """Compatibility shim for callers importing the old setup function."""
    warn_setup_deprecated()
    return 1


def main(argv: Sequence[str] | None = None) -> None:
    """Warn users to use Prime CLI for Lab setup."""
    _ = argv
    raise SystemExit(run_setup())


if __name__ == "__main__":
    main()
