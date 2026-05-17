"""
Tests for the pandas-debugger environment.

Run with:
    pip install pytest pandas numpy verifiers datasets
    pytest tests/test_pandas_debugger.py -v
"""

from __future__ import annotations

import asyncio
import json
import sys
import os

import pytest

# Ensure the package is importable when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pandas_debugger import (
    _TASKS,
    _build_dataset,
    _extract_fixed_code,
    _run_code_safe,
    _bug_type_mentioned,
    correctness_reward,
    format_reward,
    reasoning_quality_reward,
    load_environment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_completion(text: str) -> list[dict]:
    """Wrap text in a single assistant message (verifiers format)."""
    return [{"role": "assistant", "content": text}]


def _make_answer(task: dict) -> str:
    return json.dumps({
        "fixed_code": task["fixed_code"],
        "check_expr": task["check_expr"],
        "bug_type":   task["bug_type"],
    })


async def _async(coro):
    return await coro


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Task bank integrity
# ---------------------------------------------------------------------------

class TestTaskBank:
    def test_task_count(self):
        """At least 10 tasks in the bank."""
        assert len(_TASKS) >= 10

    def test_task_keys(self):
        """Every task has all required fields."""
        required = {"bug_type", "buggy_code", "fixed_code", "check_expr"}
        for i, t in enumerate(_TASKS):
            missing = required - set(t.keys())
            assert not missing, f"Task {i} missing keys: {missing}"

    def test_all_fixed_codes_pass(self):
        """Every ground-truth fixed_code must pass its own check_expr."""
        failures = []
        for i, task in enumerate(_TASKS):
            passed, stderr = _run_code_safe(task["fixed_code"], task["check_expr"])
            if not passed:
                failures.append(f"Task {i} ({task['bug_type']}): {stderr[:200]}")
        assert not failures, "Ground-truth fixed_code(s) failed:\n" + "\n".join(failures)

    def test_all_buggy_codes_fail(self):
        """Every buggy_code should FAIL its check_expr (it contains the bug)."""
        failures = []
        for i, task in enumerate(_TASKS):
            # Some buggy codes may error out (e.g., merge on wrong key raises);
            # both error and FAIL are acceptable — we just want them not to PASS
            passed, _ = _run_code_safe(task["buggy_code"], task["check_expr"])
            if passed:
                failures.append(f"Task {i} ({task['bug_type']}): buggy code unexpectedly passed")
        assert not failures, "Buggy code(s) unexpectedly passed:\n" + "\n".join(failures)

    def test_bug_types_covered(self):
        """All major bug categories should be represented."""
        represented = {t["bug_type"] for t in _TASKS}
        key_types = {"off_by_one", "dtype_cast", "merge_key", "agg_axis",
                     "fillna_method", "sort_ascending", "inplace_return", "copy_alias"}
        missing = key_types - represented
        assert not missing, f"Missing bug types: {missing}"


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

class TestDatasetBuilder:
    def test_dataset_length(self):
        ds = _build_dataset(_TASKS)
        assert len(ds) == len(_TASKS)

    def test_dataset_columns(self):
        ds = _build_dataset(_TASKS)
        assert set(ds.column_names) >= {"question", "answer", "info"}

    def test_answer_is_valid_json(self):
        ds = _build_dataset(_TASKS)
        for row in ds:
            meta = json.loads(row["answer"])
            assert "fixed_code" in meta
            assert "check_expr" in meta
            assert "bug_type" in meta

    def test_question_is_buggy_code(self):
        """question field should match the buggy_code from the task bank."""
        ds = _build_dataset(_TASKS)
        for row, task in zip(ds, _TASKS):
            assert row["question"].strip() == task["buggy_code"].strip()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestExtractFixedCode:
    def test_basic_extraction(self):
        text = "<fixed_code>\nresult = 42\n</fixed_code>"
        assert _extract_fixed_code(text) == "result = 42"

    def test_markdown_fence_stripped(self):
        text = "<fixed_code>\n```python\nresult = 42\n```\n</fixed_code>"
        assert _extract_fixed_code(text) == "result = 42"

    def test_missing_tags(self):
        assert _extract_fixed_code("no tags here") == ""

    def test_multiline(self):
        text = "<fixed_code>\nimport pandas as pd\ndf = pd.DataFrame()\n</fixed_code>"
        extracted = _extract_fixed_code(text)
        assert "import pandas" in extracted
        assert "pd.DataFrame" in extracted


class TestRunCodeSafe:
    def test_passing_check(self):
        code = "x = 2 + 2"
        passed, _ = _run_code_safe(code, "x == 4")
        assert passed

    def test_failing_check(self):
        code = "x = 1"
        passed, _ = _run_code_safe(code, "x == 99")
        assert not passed

    def test_syntax_error(self):
        code = "def broken(:"
        passed, _ = _run_code_safe(code, "True")
        assert not passed

    def test_timeout_returns_false(self):
        code = "import time; time.sleep(999)"
        passed, stderr = _run_code_safe(code, "True", timeout=2)
        assert not passed

    def test_pandas_check(self):
        import textwrap
        code = textwrap.dedent("""\
            import pandas as pd
            df = pd.DataFrame({'a': [1, 2, 3]})
            result = df['a'].sum()
        """)
        passed, _ = _run_code_safe(code, "result == 6")
        assert passed


class TestBugTypeMentioned:
    def test_off_by_one_detected(self):
        assert _bug_type_mentioned("the iloc index is off by one", "off_by_one")

    def test_dtype_cast_detected(self):
        assert _bug_type_mentioned("should use astype(float) instead of astype(int)", "dtype_cast")

    def test_false_positive_low(self):
        # completely irrelevant text should not match merge_key
        assert not _bug_type_mentioned("the weather is nice today", "merge_key")


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

class TestCorrectnessReward:
    def test_correct_fix_scores_1(self):
        task = _TASKS[0]  # off_by_one
        answer = _make_answer(task)
        good_response = f"<reasoning>Fix</reasoning><fixed_code>\n{task['fixed_code']}\n</fixed_code>"
        score = run(correctness_reward(_make_completion(good_response), answer))
        assert score == 1.0, f"Expected 1.0, got {score}"

    def test_no_tags_scores_0(self):
        task = _TASKS[0]
        answer = _make_answer(task)
        score = run(correctness_reward(_make_completion("I think the code is fine."), answer))
        assert score == 0.0

    def test_syntax_error_scores_0_25(self):
        task = _TASKS[0]
        answer = _make_answer(task)
        bad_code = "<fixed_code>def broken(:</fixed_code>"
        score = run(correctness_reward(_make_completion(bad_code), answer))
        assert score == 0.0  # syntax error → 0 (ast.parse fails)

    def test_valid_but_wrong_code_scores_partial(self):
        task = _TASKS[0]  # off_by_one: check len(result)==5
        answer = _make_answer(task)
        # valid code but wrong answer
        wrong_fix = textwrap.dedent("""\
            import pandas as pd
            data = pd.DataFrame({"x": list(range(10))})
            result = data.iloc[:3]  # wrong count
        """)
        bad_response = f"<fixed_code>\n{wrong_fix}\n</fixed_code>"
        score = run(correctness_reward(_make_completion(bad_response), answer))
        # should be 0.25 (valid syntax, ran, but failed check)
        assert 0.0 <= score <= 0.5

    def test_invalid_answer_json(self):
        score = run(correctness_reward(_make_completion("<fixed_code>x=1</fixed_code>"), "not-json"))
        assert score == 0.0

    def test_multiple_tasks_pass(self):
        """Spot-check first 5 tasks with their ground-truth fix."""
        for i, task in enumerate(_TASKS[:5]):
            answer = _make_answer(task)
            response = f"<reasoning>fix</reasoning><fixed_code>\n{task['fixed_code']}\n</fixed_code>"
            score = run(correctness_reward(_make_completion(response), answer))
            assert score == 1.0, f"Task {i} ({task['bug_type']}) expected 1.0, got {score}"


class TestFormatReward:
    def test_both_tags_scores_1(self):
        text = "<reasoning>explanation</reasoning><fixed_code>x=1</fixed_code>"
        score = run(format_reward(_make_completion(text)))
        assert score == 1.0

    def test_only_reasoning_scores_half(self):
        text = "<reasoning>explanation</reasoning>"
        score = run(format_reward(_make_completion(text)))
        assert score == 0.5

    def test_only_fixed_code_scores_half(self):
        text = "<fixed_code>x=1</fixed_code>"
        score = run(format_reward(_make_completion(text)))
        assert score == 0.5

    def test_no_tags_scores_0(self):
        score = run(format_reward(_make_completion("plain text")))
        assert score == 0.0


class TestReasoningQualityReward:
    def test_correct_keyword_mention(self):
        task = _TASKS[0]  # off_by_one
        answer = _make_answer(task)
        text = "<reasoning>The iloc index is off by one</reasoning><fixed_code>x</fixed_code>"
        score = run(reasoning_quality_reward(_make_completion(text), answer))
        assert score == 1.0

    def test_missing_keyword(self):
        task = _TASKS[0]  # off_by_one
        answer = _make_answer(task)
        text = "<reasoning>I changed something random</reasoning><fixed_code>x</fixed_code>"
        score = run(reasoning_quality_reward(_make_completion(text), answer))
        assert score == 0.0

    def test_invalid_answer(self):
        score = run(reasoning_quality_reward(_make_completion("<reasoning>x</reasoning>"), "bad-json"))
        assert score == 0.0


# ---------------------------------------------------------------------------
# Environment loader
# ---------------------------------------------------------------------------

class TestLoadEnvironment:
    def test_returns_environment(self):
        """load_environment should return a verifiers.Environment instance."""
        import verifiers as vf
        env = load_environment()
        assert isinstance(env, vf.Environment)

    def test_custom_seed(self):
        env = load_environment(seed=99)
        import verifiers as vf
        assert isinstance(env, vf.Environment)

    def test_limited_examples(self):
        env = load_environment(num_train_examples=5, num_eval_examples=3)
        import verifiers as vf
        assert isinstance(env, vf.Environment)

    def test_custom_system_prompt(self):
        env = load_environment(system_prompt="Custom prompt.")
        assert env.system_prompt == "Custom prompt."


# ---------------------------------------------------------------------------
# Integration: dataset roundtrip
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_all_tasks_end_to_end(self):
        """For every task, simulating a perfect model response should yield reward 1.0."""
        errors = []
        for i, task in enumerate(_TASKS):
            answer = _make_answer(task)
            response = (
                f"<reasoning>I found the bug in this {task['bug_type']} scenario.</reasoning>"
                f"<fixed_code>\n{task['fixed_code']}\n</fixed_code>"
            )
            completion = _make_completion(response)

            c_score = run(correctness_reward(completion, answer))
            f_score = run(format_reward(completion))
            r_score = run(reasoning_quality_reward(completion, answer))

            total = 1.0 * c_score + 0.2 * f_score + 0.1 * r_score
            if c_score != 1.0:
                errors.append(
                    f"Task {i} ({task['bug_type']}): correctness={c_score}"
                )
        assert not errors, "End-to-end failures:\n" + "\n".join(errors)


# need textwrap for the test above
import textwrap  # noqa: E402 (already imported in the module)
