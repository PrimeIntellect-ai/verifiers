from __future__ import annotations

import sys

from .toolset import run_toolset


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m verifiers.v1.toolset_runner CONFIG_JSON")
    run_toolset(sys.argv[1])


if __name__ == "__main__":
    main()
