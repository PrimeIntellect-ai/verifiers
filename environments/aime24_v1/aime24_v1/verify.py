# /// script
# dependencies = ["math-verify"]
# ///
"""Score one math answer by math-verify equivalence of the model's boxed answer vs the
gold, run inside the rollout's runtime via `uv run`. uv installs `math-verify` into its
own cache here — the dependency never touches the eval process. Takes the gold answer
(argv[1]), the model's prediction (argv[2]), and a timeout in seconds (argv[3]); prints
1.0 if they're equivalent, else 0.0.
"""

import sys

from math_verify import parse, verify

gold, pred, timeout = sys.argv[1], sys.argv[2], int(sys.argv[3])

if "<think>" in pred and "</think>" not in pred:
    print(0.0)
    sys.exit(0)
pred = pred.split("</think>")[-1]


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


answer = extract_boxed(pred)
if not answer:
    print(0.0)
    sys.exit(0)
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
print(score)
