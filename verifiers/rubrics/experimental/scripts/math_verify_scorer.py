"""Sandbox scoring script for math_verify.

Reads ground truth from ground_truth.txt and the agent's answer from answer.txt.
Prints a single float (1.0 or 0.0) to stdout.
"""

from pathlib import Path

from math_verify import parse, verify

ground_truth = Path("ground_truth.txt").read_text()
response = Path("answer.txt").read_text()

if not response:
    print(0.0)
else:
    try:
        score = float(
            verify(
                parse(ground_truth, parsing_timeout=5),
                parse(response, parsing_timeout=5),
                timeout_seconds=5,
            )
        )
        print(score)
    except BaseException:
        print(0.0)
