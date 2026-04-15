"""Evaluation command module for external hosts."""

from verifiers.scripts.eval import build_parser, main, parse_args

__all__ = ["build_parser", "parse_args", "main"]


if __name__ == "__main__":
    main()
