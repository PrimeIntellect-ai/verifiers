# /// script
# dependencies = ["math-verify"]
# ///
"""Score one GSM8K answer, run inside the rollout's runtime via `uv run`.

uv installs `math-verify` into its own cache here — the dependency never touches
the eval process. Takes the gold answer (`argv[1]`) and the model's prediction
(`argv[2]`) and prints 1.0 if they match the same number, else 0.0.

The model is asked for a single number after '####'; a reply without the marker,
or with something other than a number after it, scores 0.0.
Both the gold and the prediction are wrapped in \\boxed{} before math-verify
(matching the math-env scorer) so parsing is robust; a malformed prediction fails
to verify rather than crashing.
"""

import re
import sys

from math_verify import parse, verify

# The prompt asks for a single number. math-verify grades one value out of whatever the
# slot holds ("25 = 18" and "18 \text{ is wrong}" both read as 18), so only grade the slot
# when it is a number. Wrappers and signs that do not change the value are allowed; a
# percent sign is left to math-verify as before.
NUMBER = re.compile(
    r"[-\u2212]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:\s*\\?%)?"
    r"|[-\u2212]?\d+/\d+|[-\u2212]?\\[dt]?frac\{\d+\}\{\d+\}"
)
WRAPPERS = (
    r"\\boxed\{(.*)\}",
    r"\\\((.*)\\\)",
    r"\\\[(.*)\\\]",
    r"\$\$(.*)\$\$",
    r"\$(.+)\$",
    r"\\(?:textbf|mathbf)\{(.*)\}",
    r"\{(.*)\}",
    r"\*\*(.+)\*\*",
    r"\*(.+)\*",
    r"__(.+)__",
    r"_(.+)_",
)


def is_single_number(slot):
    s, previous = slot, None
    while s != previous:
        previous = s
        s = s.strip().rstrip(".").strip()
        for wrapper in WRAPPERS:
            m = re.fullmatch(wrapper, s)
            if m:
                s = m.group(1).strip()
        s = re.sub(r"^(?:\\?\$|\+)", "", s)
    return NUMBER.fullmatch(s.replace("{,}", ",")) is not None


gold, pred = sys.argv[1], sys.argv[2]
matches = re.findall(r"####\s*(.+)", pred)
if not matches:
    # The prompt asks for the answer after '####'. Without it math-verify would pick a
    # number out of free text, so "definitely not 18" or "25 or 18" could pass.
    print(0.0)
    sys.exit()
# Markdown emphasis and a full stop around the number are formatting, not the answer.
prediction = matches[-1].strip().strip("*_").rstrip(".").strip("*_ ")
if not is_single_number(prediction):
    print(0.0)
    sys.exit()
try:
    score = (
        1.0
        if verify(parse("\\boxed{" + gold + "}"), parse("\\boxed{" + prediction + "}"))
        else 0.0
    )
except Exception:  # noqa: BLE001 - malformed answers can fail anywhere in math-verify
    score = 0.0
print(score)
