"""Scoring helpers: stdout comparison, pytest outcome parsing, and judge-choice parsing."""

from decimal import Context, localcontext

import pytest

import verifiers.v1 as vf


def test_compare_stdout_results_accepts_token_equal_text() -> None:
    assert vf.compare_stdout_results("hello   world\n", "hello world\n")


def test_compare_stdout_results_keeps_numeric_tolerance() -> None:
    assert vf.compare_stdout_results("1.0001 2.0\n", "1.0002 2.0009\n")


@pytest.mark.parametrize(
    ("actual", "expected"),
    [
        ("1e1000000", "0"),
        ("-1e1000000", "0"),
        ("0", "1e1000000"),
        ("0", "-1e1000000"),
        ("1e1000000", "-1e1000000"),
    ],
)
def test_compare_stdout_results_rejects_overflowing_difference(
    actual: str, expected: str
) -> None:
    with localcontext(Context()) as context:
        original_flags = context.flags.copy()
        original_traps = context.traps.copy()
        assert not vf.compare_stdout_results(actual, expected)
        assert context.flags == original_flags
        assert context.traps == original_traps
        assert context.prec == 28
        assert context.Emax == 999999


@pytest.mark.parametrize(
    ("actual", "expected", "matches"),
    [
        ("1e1000000", "10e999999", True),
        ("1e1000000", "1e1000000", True),
        ("1.0001", "1.0002", True),
        ("1.1", "1.2", False),
        ("NaN", "0", False),
        ("Infinity", "0", False),
        ("-Infinity", "0", False),
        ("NaN", "NaN", True),
        ("Infinity", "Infinity", True),
    ],
)
def test_compare_stdout_results_preserves_numeric_and_exact_match_behavior(
    actual: str, expected: str, matches: bool
) -> None:
    with localcontext(Context()):
        assert vf.compare_stdout_results(actual, expected) is matches


def test_parse_pytest_outcomes_strips_xfail_xpass_reasons() -> None:
    output = (
        "XFAIL tests/test_mod.py::test_xfail - known bug - still tracked\n"
        "XPASS tests/test_mod.py::test_xpass - always xfail - unexpectedly passed\n"
        "FAILED tests/test_mod.py::test_param[a - b] - assert left - right\n"
        "PASSED tests/test_mod.py::test_ok"
    )

    assert vf.parse_pytest_outcomes(output) == {
        "tests/test_mod.py::test_xfail": "XFAIL",
        "tests/test_mod.py::test_xpass": "XPASS",
        "tests/test_mod.py::test_param[a - b]": "FAILED",
        "tests/test_mod.py::test_ok": "PASSED",
    }


def test_parse_judge_choice_prefers_final_marker_then_boxed() -> None:
    assert (
        vf.parse_judge_choice(
            "Draft: \\boxed{A}\nFinal Judgment: B", choices=("A", "B")
        )
        == "B"
    )
    assert (
        vf.parse_judge_choice(
            "Draft: \\boxed{A}\nFinal Judgment:\nReasoning mentions B",
            choices=("A", "B"),
        )
        == "A"
    )
    assert (
        vf.parse_judge_choice(
            "Draft: \\boxed{B}\nFinal Judgment: This is a \\boxed{B}",
            choices=("A", "B"),
        )
        == "B"
    )
    assert (
        vf.parse_judge_choice(
            "Final Judgment: A better choice is \\boxed{B}", choices=("A", "B")
        )
        == "A"
    )
    assert (
        vf.parse_judge_choice(
            "Final Judgment: \\boxed{A}\nFinal Judgment: B", choices=("A", "B")
        )
        == "B"
    )
    assert (
        vf.parse_judge_choice("Draft: \\boxed{A}\nAnswer: B", choices=("A", "B")) == "A"
    )
