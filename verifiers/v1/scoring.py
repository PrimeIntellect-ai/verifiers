from __future__ import annotations

import re
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING

from math_verify import parse, verify
from math_verify.errors import TimeoutException as MathVerifyTimeout

from verifiers.v1.errors import SandboxError

if TYPE_CHECKING:
    from verifiers.v1.runtimes import Runtime
    from verifiers.v1.trace import Trace

BOXED_START = "\\boxed{"
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PYTEST_OUTCOMES = ("PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL", "XPASS")


def extract_boxed_answer(text: str, strict: bool = False) -> str:
    start = text.rfind(BOXED_START)
    if start == -1:
        return "" if strict else text

    # Regex is the wrong tool for nested braces, so walk the final box by hand.
    answer_start = start + len(BOXED_START)
    depth = 1
    for index, char in enumerate(text[answer_start:], start=answer_start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        if depth == 0:
            return text[answer_start:index]

    return "" if strict else text


async def read_answer_file_or_last_reply(
    runtime: Runtime, path: str, trace: Trace
) -> str:
    try:
        answer = (await runtime.read(path)).decode(errors="replace").strip()
    except (FileNotFoundError, OSError, SandboxError):
        answer = ""
    return answer or trace.last_reply


def parse_judge_choice(
    content: str | None, choices: Sequence[str] = ("A", "B", "C")
) -> str | None:
    if not content or ("<think>" in content and "</think>" not in content):
        return None

    text = content.rsplit("</think>", 1)[-1].strip()
    text = extract_boxed_answer(text, strict=True).strip() or text

    text_upper = text.upper()
    choices_by_upper = {choice.upper(): choice for choice in choices}
    allowed = "|".join(re.escape(choice) for choice in choices_by_upper)
    choice_re = rf"(?<!\w)({allowed})(?!\w)"
    verdict_re = (
        r"(?:^|\n)\s*(?:FINAL\s+JUDGMENT|FINAL\s+ANSWER|FINAL\s+VERDICT|"
        r"JUDGMENT|VERDICT|ANSWER)\s*(?:IS\s*)?[:\-]?\s*"
    )

    verdict = re.search(verdict_re, text_upper)
    if verdict:
        match = re.search(choice_re, text_upper[verdict.end() :])
        return choices_by_upper.get(match.group(1)) if match else None

    matches = re.findall(choice_re, text_upper)
    return choices_by_upper.get(matches[-1]) if matches else None


def verify_boxed_math_answer(
    response: str | None, answer: str, *, timeout_seconds: int = 5
) -> float:
    if not response or ("<think>" in response and "</think>" not in response):
        return 0.0

    prediction_text = response.rsplit("</think>", 1)[-1]
    prediction = extract_boxed_answer(prediction_text, strict=True).strip()
    gold = (extract_boxed_answer(answer, strict=True) or answer).strip()
    if not prediction or not gold:
        return 0.0

    # math-verify expects both sides as boxed math expressions.
    try:
        parsed_gold = parse(f"\\boxed{{{gold}}}", parsing_timeout=timeout_seconds)
        parsed_prediction = parse(
            f"\\boxed{{{prediction}}}",
            parsing_timeout=timeout_seconds,
        )
        return float(
            verify(
                parsed_gold,
                parsed_prediction,
                timeout_seconds=timeout_seconds,
            )
        )
    except (Exception, MathVerifyTimeout):
        return 0.0


def compare_stdout_results(
    exec_stdout: str, expected: str, *, tolerance: float = 1e-3
) -> bool:
    if exec_stdout.strip() == expected.strip():
        return True

    actual_lines = [line.strip() for line in exec_stdout.splitlines() if line.strip()]
    expected_lines = [line.strip() for line in expected.splitlines() if line.strip()]
    if actual_lines == expected_lines:
        return True

    actual_tokens = [token for line in actual_lines for token in line.split()]
    expected_tokens = [token for line in expected_lines for token in line.split()]
    if actual_tokens == expected_tokens:
        return True
    if len(actual_tokens) != len(expected_tokens) or not actual_tokens:
        return False

    # Code datasets often allow tiny floating-point drift in stdout.
    try:
        actual_numbers = [Decimal(token) for token in actual_tokens]
        expected_numbers = [Decimal(token) for token in expected_tokens]
    except (InvalidOperation, ValueError):
        return False
    if not all(number.is_finite() for number in actual_numbers + expected_numbers):
        return False

    limit = Decimal(str(tolerance))
    return all(
        abs(actual - expected) <= limit
        for actual, expected in zip(actual_numbers, expected_numbers)
    )


def parse_pytest_outcomes(output: str | None) -> dict[str, str]:
    outcomes: dict[str, str] = {}
    for raw_line in ANSI_RE.sub("", output or "").splitlines():
        parts = raw_line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue

        outcome, test_id = parts
        if outcome not in PYTEST_OUTCOMES:
            continue

        # SKIPPED short-summary rows use "[N] file.py:line: reason", not a node id.
        if test_id.startswith("["):
            continue

        # These summary rows append " - <reason>"; passing node ids do not.
        if outcome in ("FAILED", "ERROR", "XFAIL", "XPASS") and " - " in test_id:
            parts = test_id.split(" - ")
            for index in range(1, len(parts)):
                node_id = " - ".join(parts[:index])
                if node_id.count("[") == node_id.count("]"):
                    test_id = node_id
                    break
        outcomes[test_id.rstrip()] = outcome
    return outcomes
