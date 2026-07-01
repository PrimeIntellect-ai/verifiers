import verifiers.v1 as vf


def test_compare_stdout_results_accepts_token_equal_text() -> None:
    assert vf.compare_stdout_results("hello   world\n", "hello world\n")


def test_compare_stdout_results_keeps_numeric_tolerance() -> None:
    assert vf.compare_stdout_results("1.0001 2.0\n", "1.0002 2.0009\n")


def test_parse_pytest_outcomes_strips_xfail_xpass_reasons() -> None:
    output = "\n".join(
        [
            "XFAIL tests/test_mod.py::test_xfail - known bug - still tracked",
            "XPASS tests/test_mod.py::test_xpass - always xfail - unexpectedly passed",
            "FAILED tests/test_mod.py::test_param[a - b] - assert left - right",
            "PASSED tests/test_mod.py::test_ok",
        ]
    )

    assert vf.parse_pytest_outcomes(output) == {
        "tests/test_mod.py::test_xfail": "XFAIL",
        "tests/test_mod.py::test_xpass": "XPASS",
        "tests/test_mod.py::test_param[a - b]": "FAILED",
        "tests/test_mod.py::test_ok": "PASSED",
    }


def test_parse_judge_choice_uses_first_choice_after_verdict_marker() -> None:
    response = "Final Judgment: B because it is a better answer"

    assert vf.parse_judge_choice(response, choices=("A", "B")) == "B"
