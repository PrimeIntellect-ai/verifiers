"""Check a runtime for conformance: provision it and exercise the backend contract and
every feature it declares. Use it to verify a custom runtime plugin, or a built-in one.

    uv run scripts/check_runtime.py docker
    uv run scripts/check_runtime.py prime --set labels='["vf-ci"]'
    uv run scripts/check_runtime.py custom \\
        --set import_path=my_package.runtime.MyBackend --set region=us-east-1

`--set KEY=VALUE` sets a runtime config field; VALUE is parsed as JSON when it can be.
The command exits non-zero when any check fails."""

import argparse
import asyncio
import json
import sys

from pydantic import TypeAdapter

from verifiers.v1.runtimes import RuntimeConfig
from verifiers.v1.runtimes.conformance import IMAGE, check_runtime


def parse_field(assignment: str) -> tuple[str, object]:
    key, separator, value = assignment.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {assignment!r}")
    try:
        return key, json.loads(value)
    except json.JSONDecodeError:
        return key, value


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "type", help="runtime type, e.g. subprocess, docker, prime, custom"
    )
    parser.add_argument(
        "--set",
        dest="fields",
        action="append",
        type=parse_field,
        default=[],
        metavar="KEY=VALUE",
        help="runtime config field",
    )
    parser.add_argument(
        "--image", default=IMAGE, help="task image for the task_images check"
    )
    args = parser.parse_args()
    config = TypeAdapter(RuntimeConfig).validate_python(
        {"type": args.type, **dict(args.fields)}
    )
    results = asyncio.run(check_runtime(config, image=args.image))
    width = max(len(result.name) for result in results)
    for result in results:
        status = {True: "PASS", False: "FAIL", None: "SKIP"}[result.passed]
        print(f"{status}  {result.name:<{width}}  {result.detail}".rstrip())
    return 1 if any(result.passed is False for result in results) else 0


if __name__ == "__main__":
    sys.exit(main())
