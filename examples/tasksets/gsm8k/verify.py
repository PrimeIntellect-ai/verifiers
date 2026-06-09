# /// script
# dependencies = ["math-verify"]
# ///
"""Score one GSM8K answer, run inside the rollout's runtime via `uv run`.

uv installs `math-verify` into its own cache here — the dependency never touches
the eval process. Takes the gold answer (`argv[1]`) and the model's prediction
(`argv[2]`) and prints 1.0 if they match the same number, else 0.0.

The model is asked for its answer after '####'. Both the gold and the prediction
are wrapped in \\boxed{} before math-verify (matching the math-env scorer) so
parsing is robust; a malformed prediction fails to verify rather than crashing.
"""

import re
import sys

from math_verify import parse, verify

gold, pred = sys.argv[1], sys.argv[2]
# GSM8K answers come after '####'; fall back to the whole response.
match = re.search(r"####\s*(.+)", pred)
prediction = match.group(1).strip() if match else pred
try:
    score = (
        1.0
        if verify(parse("\\boxed{" + gold + "}"), parse("\\boxed{" + prediction + "}"))
        else 0.0
    )
except Exception:
    score = 0.0
print(score)
