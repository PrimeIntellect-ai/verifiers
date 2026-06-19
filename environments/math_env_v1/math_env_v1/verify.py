# /// script
# dependencies = ["math-verify"]
# ///
"""Score one math answer by math-verify equivalence of the model's boxed answer vs the gold,
run inside the rollout's runtime via `uv run` (or a warm worker). uv installs `math-verify` into
its own cache here — the dependency never touches the eval process. `main(argv)` takes the gold
answer (argv[0]), the model's prediction (argv[1]), and a timeout in seconds (argv[2]); returns
"1.0" if they're equivalent, else "0.0".

Exposing `main(argv) -> str` (plus the `__main__` footer) lets the runtime keep this as a warm
worker — `import math_verify` paid once, not per call (see `Runtime.run_uv_script(warm=True)`) —
while staying `uv run verify.py <gold> <pred> <timeout>`-able cold. `main` must `return` (never
`sys.exit`, which would kill a reused worker).
"""

import sys

from math_verify import parse, verify


def extract_boxed(text: str) -> str:
    """Content of the last ``\\boxed{...}`` in ``text``, or "" if there is none."""
    start = text.rfind("\\boxed{")
    if start == -1:
        return ""
    i, depth = start + len("\\boxed{"), 1
    while i < len(text) and depth:
        depth += (text[i] == "{") - (text[i] == "}")
        i += 1
    return text[start + len("\\boxed{") : i - 1] if depth == 0 else ""


def main(argv: list[str]) -> str:
    gold, pred, timeout = argv[0], argv[1], int(argv[2])
    if "<think>" in pred and "</think>" not in pred:
        return "0.0"
    pred = pred.split("</think>")[-1]
    answer = extract_boxed(pred)
    if not answer:
        return "0.0"
    try:
        score = (
            1.0
            if verify(
                parse("\\boxed{" + gold + "}", parsing_timeout=timeout),
                parse("\\boxed{" + answer + "}", parsing_timeout=timeout),
                timeout_seconds=timeout,
            )
            else 0.0
        )
    except Exception:
        score = 0.0
    return str(score)


if __name__ == "__main__":
    print(main(sys.argv[1:]))
