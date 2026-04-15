"""Evaluation command module for external hosts."""

from verifiers.scripts.eval import (
    build_extra_headers,
    build_parser,
    main,
    merge_sampling_args,
    parse_args,
)

__all__ = [
    "build_extra_headers",
    "build_parser",
    "merge_sampling_args",
    "parse_args",
    "main",
]


if __name__ == "__main__":
    main()
