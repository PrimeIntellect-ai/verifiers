# /// script
# dependencies = ["harbor-rewardkit==0.2.4"]
# ///
"""Discover and execute task Python in the verifier; never launch a judge here."""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path


def main() -> None:
    spec = importlib.util.spec_from_file_location("vf_rewardkit", sys.argv[1])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = asyncio.run(module.discover_plan(sys.argv[2], sys.argv[3]))
    Path(sys.argv[4]).write_text(json.dumps(plan, allow_nan=False))


if __name__ == "__main__":
    main()
