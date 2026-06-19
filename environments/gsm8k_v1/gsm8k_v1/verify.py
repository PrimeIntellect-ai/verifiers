# /// script
# dependencies = ["math-verify"]
# ///
"""Score one GSM8K answer, run inside the rollout's runtime via `uv run` (or a warm worker).

uv installs `math-verify` into its own cache here — the dependency never touches the eval
process. `main(argv)` takes the gold answer (`argv[0]`) and the model's prediction (`argv[1]`)
and returns "1.0" if they match the same number, else "0.0".

Exposing `main(argv) -> str` (plus the `__main__` footer) lets the runtime keep this as a warm
worker — `import math_verify` is paid once, not per call (see `Runtime.run_uv_script(warm=True)`)
— while staying `uv run verify.py <gold> <pred>`-able cold. The model is asked for its answer
after '####'. Both the gold and the prediction are wrapped in \\boxed{} before math-verify
(matching the math-env scorer) so parsing is robust; a malformed prediction fails to verify
rather than crashing.
"""

import re
import sys

from math_verify import parse, verify


def main(argv: list[str]) -> str:
    gold, pred = argv[0], argv[1]
    matches = re.findall(r"####\s*(.+)", pred)
    prediction = matches[-1].strip() if matches else pred
    try:
        score = (
            1.0
            if verify(
                parse("\\boxed{" + gold + "}"), parse("\\boxed{" + prediction + "}")
            )
            else 0.0
        )
    except Exception:
        score = 0.0
    return str(score)


if __name__ == "__main__":
    print(main(sys.argv[1:]))
