"""Scoring helpers: stdout comparison, pytest outcome parsing, and judge-choice parsing."""

import verifiers.v1 as vf


def test_compare_stdout_results_accepts_token_equal_text() -> None:
    assert vf.compare_stdout_results("hello   world\n", "hello world\n")


def test_compare_stdout_results_keeps_numeric_tolerance() -> None:
    assert vf.compare_stdout_results("1.0001 2.0\n", "1.0002 2.0009\n")


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
