# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "internbootcamp @ https://github.com/dmihal/InternBootcamp/archive/2b2d388f4f056cd9bd0cc91b130f0b54b15572b4.tar.gz",
# ]
# ///
"""Run an InternBootcamp scorer inside the rollout container."""

from __future__ import annotations

import inspect
import json
import math
import os
import random
import re
import signal
import sys


def canonical_key(class_name: str) -> str:
    base = re.sub(r"bootcamp$", "", class_name, flags=re.IGNORECASE)
    return re.sub(r"[^0-9a-z]+", "", base.lower())


def discover(module) -> dict[str, type]:
    classes: dict[str, type] = {}
    for name, candidate in vars(module).items():
        if (
            inspect.isclass(candidate)
            and name.lower().endswith("bootcamp")
            and callable(getattr(candidate, "case_generator", None))
            and callable(getattr(candidate, "prompt_func", None))
            and callable(getattr(candidate, "verify_score", None))
        ):
            key = getattr(candidate, "canonical_name", None) or canonical_key(name)
            classes[canonical_key(str(key))] = candidate
    return classes


def _timeout(_signum, _frame) -> None:
    raise TimeoutError("InternBootcamp verifier timed out")


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: verify.py BOOTCAMP IDENTITY_JSON COMPLETION_TEXT")
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, _timeout)
        signal.alarm(int(os.environ.get("INTERNBOOTCAMP_VERIFY_TIMEOUT", "180")))

    import internbootcamp
    import numpy as np

    random.seed(0)
    np.random.seed(0)

    bootcamp_name, identity_path, completion_path = sys.argv[1:]
    classes = discover(internbootcamp)
    if bootcamp_name not in classes:
        raise ValueError(f"unknown InternBootcamp task: {bootcamp_name}")
    bootcamp = classes[bootcamp_name]()
    with open(identity_path, encoding="utf-8") as handle:
        identity = json.load(handle)
    with open(completion_path, encoding="utf-8", errors="replace") as handle:
        completion = handle.read()
    score = float(bootcamp.verify_score(completion, identity))
    if not math.isfinite(score):
        score = 0.0
    print(json.dumps({"score": min(1.0, max(0.0, score))}))


if __name__ == "__main__":
    main()
