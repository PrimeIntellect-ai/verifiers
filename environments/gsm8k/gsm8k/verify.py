# /// script
# dependencies = ["math-verify"]
# ///
"""Score one GSM8K answer, run inside the rollout's runtime via `uv run`.

uv installs `math-verify` into its own cache here — the dependency never touches
the eval process. Takes the gold answer (`argv[1]`) and the model's prediction
(`argv[2]`) and prints 1.0 if they match the same number, else 0.0.

The model is asked for its answer after '####'; a reply without it scores 0.0.
Both the gold and the prediction are wrapped in \\boxed{} before math-verify
(matching the math-env scorer) so parsing is robust; a malformed prediction fails
to verify rather than crashing.
"""

import re
import sys

from math_verify import parse, verify

gold, pred = sys.argv[1], sys.argv[2]
matches = re.findall(r"####\s*(.+)", pred)
if not matches:
    # The prompt asks for the answer after '####'. Without it math-verify would pick a
    # number out of free text, so "definitely not 18" or "25 or 18" could pass.
    print(0.0)
    sys.exit()
# Markdown emphasis and a full stop around the number are formatting, not the answer.
prediction = matches[-1].strip().strip("*_").rstrip(".").strip("*_ ")
try:
    score = (
        1.0
        if verify(parse("\\boxed{" + gold + "}"), parse("\\boxed{" + prediction + "}"))
        else 0.0
    )
except Exception:  # noqa: BLE001 - malformed answers can fail anywhere in math-verify
    score = 0.0
print(score)
