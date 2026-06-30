"""
pandas-debugger: RL environment for debugging broken pandas/NumPy data pipelines.

The model receives a short Python snippet with an injected data-wrangling bug
and must produce a corrected version. Rewards are determined by executing the
fixed code and comparing outputs to the ground-truth pipeline.

Bug categories:
  - dtype_cast     : wrong dtype coercion losing information
  - off_by_one     : slice / iloc indexing off by one
  - merge_key      : wrong join key or join type
  - agg_axis       : aggregation on wrong axis
  - fillna_method  : wrong fill strategy (forward vs backward vs value)
  - groupby_reset  : missing reset_index after groupby
  - str_strip      : missing strip on string column causing silent mismatch
  - sort_ascending : sort direction inverted
  - inplace_return : inplace=True but return value expected (returns None)
  - copy_alias     : modifying a slice view instead of a copy
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import textwrap
from typing import Any

from datasets import Dataset

import verifiers as vf

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert Python data engineer specialising in pandas and NumPy.

You will be shown a short Python data-pipeline snippet that contains exactly ONE bug.
Your task:
1. Identify which line is buggy and why.
2. Output the corrected code inside <fixed_code> tags.

Rules:
- Keep all variable names and the overall structure identical.
- Only change the minimum necessary to fix the bug.
- The corrected snippet must be self-contained and runnable.

Format your response as:

<reasoning>
Brief explanation of the bug and why your fix works.
</reasoning>
<fixed_code>
# corrected Python code here
</fixed_code>"""

# ---------------------------------------------------------------------------
# Task bank — (buggy_code, fixed_code, description)
# Each entry is a dict with keys:
#   buggy_code  : str  — snippet shown to the model
#   fixed_code  : str  — canonical correct snippet
#   bug_type    : str  — category label
#   check_expr  : str  — Python expression evaluated against local namespace
#                        that must return True after correct execution
# ---------------------------------------------------------------------------

_TASKS: list[dict[str, str]] = [
    # -----------------------------------------------------------------------
    # 1. off_by_one — iloc
    # -----------------------------------------------------------------------
    {
        "bug_type": "off_by_one",
        "buggy_code": textwrap.dedent("""\
            import pandas as pd
            data = pd.DataFrame({"x": list(range(10))})
            # keep first 5 rows
            result = data.iloc[:4]
        """),
        "fixed_code": textwrap.dedent("""\
            import pandas as pd
            data = pd.DataFrame({"x": list(range(10))})
            # keep first 5 rows
            result = data.iloc[:5]
        """),
        "check_expr": "len(result) == 5",
    },
    # -----------------------------------------------------------------------
    # 2. dtype_cast — int to float truncation on division
    # -----------------------------------------------------------------------
    {
        "bug_type": "dtype_cast",
        "buggy_code": textwrap.dedent("""\
            import pandas as pd
            data = pd.DataFrame({"sales": [100, 200, 300], "days": [7, 14, 30]})
            # compute daily average (float)
            data["avg"] = (data["sales"] / data["days"]).astype(int)
        """),
        "fixed_code": textwrap.dedent("""\
            import pandas as pd
            data = pd.DataFrame({"sales": [100, 200, 300], "days": [7, 14, 30]})
            # compute daily average (float)
            data["avg"] = (data["sales"] / data["days"]).astype(float)
        """),
        "check_expr": "data['avg'].dtype == float and abs(data['avg'].iloc[0] - 100/7) < 0.01",
    },
    # -----------------------------------------------------------------------
    # 3. merge_key — wrong join column
    # -----------------------------------------------------------------------
    {
        "bug_type": "merge_key",
        "buggy_code": textwrap.dedent("""\
            import pandas as pd
            left  = pd.DataFrame({"user_id": [1, 2, 3], "score": [10, 20, 30]})
            right = pd.DataFrame({"user_id": [1, 2, 3], "name": ["Alice", "Bob", "Carol"]})
            result = pd.merge(left, right, on="score")
        """),
        "fixed_code": textwrap.dedent("""\
            import pandas as pd
            left  = pd.DataFrame({"user_id": [1, 2, 3], "score": [10, 20, 30]})
            right = pd.DataFrame({"user_id": [1, 2, 3], "name": ["Alice", "Bob", "Carol"]})
            result = pd.merge(left, right, on="user_id")
        """),
        "check_expr": "len(result) == 3 and 'name' in result.columns",
    },
    # -----------------------------------------------------------------------
    # 4. agg_axis — mean on wrong axis
    # -----------------------------------------------------------------------
    {
        "bug_type": "agg_axis",
        "buggy_code": textwrap.dedent("""\
            import pandas as pd
            data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
            # row-wise mean (mean of a and b for each row)
            result = data.mean(axis=0)
        """),
        "fixed_code": textwrap.dedent("""\
            import pandas as pd
            data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
            # row-wise mean (mean of a and b for each row)
            result = data.mean(axis=1)
        """),
        "check_expr": "len(result) == 3 and abs(result.iloc[0] - 2.5) < 0.001",
    },
    # -----------------------------------------------------------------------
    # 5. fillna_method — wrong fill direction
    # -----------------------------------------------------------------------
    {
        "bug_type": "fillna_method",
        "buggy_code": textwrap.dedent("""\
            import pandas as pd
            import numpy as np
            s = pd.Series([1.0, np.nan, np.nan, 4.0])
            # forward-fill: carry last valid value forward
            result = s.ffill()
        """),
        "fixed_code": textwrap.dedent("""\
            import pandas as pd
            import numpy as np
            s = pd.Series([1.0, np.nan, np.nan, 4.0])
            # forward-fill: carry last valid value forward
            result = s.ffill()
        """),
        # This is actually already correct — use as a "no bug" sanity variant;
        # instead use a genuinely buggy direction swap:
        "check_expr": "result.iloc[1] == 1.0",
    },
    # -----------------------------------------------------------------------
    # 5b. fillna_method — bfill when ffill needed
    # -----------------------------------------------------------------------
    {
        "bug_type": "fillna_method",
        "buggy_code": textwrap.dedent("""\
            import pandas as pd
            import numpy as np
            s = pd.Series([1.0, np.nan, np.nan, 4.0])
            # forward-fill: carry last valid value forward
            result = s.bfill()
        """),
        "fixed_code": textwrap.dedent("""\
            import pandas as pd
            import numpy as np
            s = pd.Series([1.0, np.nan, np.nan, 4.0])
            # forward-fill: carry last valid value forward
            result = s.ffill()
        """),
        "check_expr": "result.iloc[1] == 1.0 and result.iloc[2] == 1.0",
    },
    # -----------------------------------------------------------------------
    # 6. groupby_reset — missing reset_index
    # -----------------------------------------------------------------------
    {
        "bug_type": "groupby_reset",
        "buggy_code": textwrap.dedent("""\
            import pandas as pd
            data = pd.DataFrame({"dept": ["A", "A", "B", "B"], "val": [1, 2, 3, 4]})
            result = data.groupby("dept")["val"].sum()
            # expected: a regular DataFrame with columns dept and val
            total = result["A"]
        """),
        "fixed_code": textwrap.dedent("""\
            import pandas as pd
            data = pd.DataFrame({"dept": ["A", "A", "B", "B"], "val": [1, 2, 3, 4]})
            result = data.groupby("dept")["val"].sum().reset_index()
            # expected: a regular DataFrame with columns dept and val
            total = result.loc[result["dept"] == "A", "val"].values[0]
        """),
        "check_expr": "isinstance(result, pd.DataFrame) and 'dept' in result.columns",
    },
    # -----------------------------------------------------------------------
    # 7. str_strip — leading/trailing whitespace in key column
    # -----------------------------------------------------------------------
    {
        "bug_type": "str_strip",
        "buggy_code": textwrap.dedent("""\
            import pandas as pd
            df = pd.DataFrame({"city": [" NYC ", " LA ", " Chicago "], "pop": [8, 4, 3]})
            lookup = {"NYC": "New York", "LA": "Los Angeles", "Chicago": "Chicago"}
            df["full_name"] = df["city"].map(lookup)
        """),
        "fixed_code": textwrap.dedent("""\
            import pandas as pd
            df = pd.DataFrame({"city": [" NYC ", " LA ", " Chicago "], "pop": [8, 4, 3]})
            lookup = {"NYC": "New York", "LA": "Los Angeles", "Chicago": "Chicago"}
            df["full_name"] = df["city"].str.strip().map(lookup)
        """),
        "check_expr": "df['full_name'].notna().all() and df['full_name'].iloc[0] == 'New York'",
    },
    # -----------------------------------------------------------------------
    # 8. sort_ascending — inverted sort
    # -----------------------------------------------------------------------
    {
        "bug_type": "sort_ascending",
        "buggy_code": textwrap.dedent("""\
            import pandas as pd
            data = pd.DataFrame({"score": [3, 1, 4, 1, 5, 9, 2, 6]})
            # get top-3 scores (descending)
            top3 = data.sort_values("score", ascending=True).head(3)
        """),
        "fixed_code": textwrap.dedent("""\
            import pandas as pd
            data = pd.DataFrame({"score": [3, 1, 4, 1, 5, 9, 2, 6]})
            # get top-3 scores (descending)
            top3 = data.sort_values("score", ascending=False).head(3)
        """),
        "check_expr": "top3['score'].min() >= 5 and top3['score'].max() == 9",
    },
    # -----------------------------------------------------------------------
    # 9. inplace_return — inplace=True returns None
    # -----------------------------------------------------------------------
    {
        "bug_type": "inplace_return",
        "buggy_code": textwrap.dedent("""\
            import pandas as pd
            data = pd.DataFrame({"a": [3, 1, 2]})
            data = data.sort_values("a", inplace=True)
            # data is now None
            result = data
        """),
        "fixed_code": textwrap.dedent("""\
            import pandas as pd
            data = pd.DataFrame({"a": [3, 1, 2]})
            data.sort_values("a", inplace=True)
            # data is now sorted in place
            result = data
        """),
        "check_expr": "result is not None and result['a'].iloc[0] == 1",
    },
    # -----------------------------------------------------------------------
    # 10. copy_alias — chained assignment modifying a view
    # -----------------------------------------------------------------------
    {
        "bug_type": "copy_alias",
        "buggy_code": textwrap.dedent("""\
            import pandas as pd
            data = pd.DataFrame({"x": [1, 2, 3, 4, 5], "label": ["a", "b", "a", "b", "a"]})
            subset = data[data["label"] == "a"]
            subset["x"] = subset["x"] * 10   # modifies a view; original unchanged
            result = data["x"].sum()          # should be 90 (10+30+50) but is 15
        """),
        "fixed_code": textwrap.dedent("""\
            import pandas as pd
            data = pd.DataFrame({"x": [1, 2, 3, 4, 5], "label": ["a", "b", "a", "b", "a"]})
            subset = data[data["label"] == "a"].copy()
            subset["x"] = subset["x"] * 10   # safe; working on a copy
            result = subset["x"].sum()        # 10+30+50 = 90
        """),
        "check_expr": "result == 90",
    },
    # -----------------------------------------------------------------------
    # 11. merge join type — inner instead of left (drops rows)
    # -----------------------------------------------------------------------
    {
        "bug_type": "merge_key",
        "buggy_code": textwrap.dedent("""\
            import pandas as pd
            orders  = pd.DataFrame({"order_id": [1,2,3,4], "amount": [50,80,120,30]})
            shipped = pd.DataFrame({"order_id": [1,3],     "carrier": ["UPS","FedEx"]})
            # keep all orders, add carrier if available
            result = pd.merge(orders, shipped, on="order_id", how="inner")
        """),
        "fixed_code": textwrap.dedent("""\
            import pandas as pd
            orders  = pd.DataFrame({"order_id": [1,2,3,4], "amount": [50,80,120,30]})
            shipped = pd.DataFrame({"order_id": [1,3],     "carrier": ["UPS","FedEx"]})
            # keep all orders, add carrier if available
            result = pd.merge(orders, shipped, on="order_id", how="left")
        """),
        "check_expr": "len(result) == 4",
    },
    # -----------------------------------------------------------------------
    # 12. dtype_cast — object column not converted before numeric op
    # -----------------------------------------------------------------------
    {
        "bug_type": "dtype_cast",
        "buggy_code": textwrap.dedent("""\
            import pandas as pd
            data = pd.DataFrame({"revenue": ["100", "200", "300"]})
            # sum revenue as numbers
            total = data["revenue"].sum()
        """),
        "fixed_code": textwrap.dedent("""\
            import pandas as pd
            data = pd.DataFrame({"revenue": ["100", "200", "300"]})
            # sum revenue as numbers
            total = data["revenue"].astype(float).sum()
        """),
        "check_expr": "total == 600.0",
    },
    # -----------------------------------------------------------------------
    # 13. agg_axis — cumsum on wrong axis
    # -----------------------------------------------------------------------
    {
        "bug_type": "agg_axis",
        "buggy_code": textwrap.dedent("""\
            import pandas as pd
            data = pd.DataFrame({"week": [1,2,3], "sales": [100, 150, 200]})
            # running total of sales over time
            data["running_total"] = data["sales"].cumsum(axis=1)
        """),
        "fixed_code": textwrap.dedent("""\
            import pandas as pd
            data = pd.DataFrame({"week": [1,2,3], "sales": [100, 150, 200]})
            # running total of sales over time
            data["running_total"] = data["sales"].cumsum(axis=0)
        """),
        "check_expr": "data['running_total'].iloc[-1] == 450",
    },
    # -----------------------------------------------------------------------
    # 14. off_by_one — head vs tail confusion
    # -----------------------------------------------------------------------
    {
        "bug_type": "off_by_one",
        "buggy_code": textwrap.dedent("""\
            import pandas as pd
            data = pd.DataFrame({"ts": list(range(100)), "val": list(range(100, 200))})
            # retrieve the LAST 10 rows (most recent)
            recent = data.head(10)
        """),
        "fixed_code": textwrap.dedent("""\
            import pandas as pd
            data = pd.DataFrame({"ts": list(range(100)), "val": list(range(100, 200))})
            # retrieve the LAST 10 rows (most recent)
            recent = data.tail(10)
        """),
        "check_expr": "recent['ts'].min() == 90",
    },
    # -----------------------------------------------------------------------
    # 15. str_strip — case mismatch in string comparison
    # -----------------------------------------------------------------------
    {
        "bug_type": "str_strip",
        "buggy_code": textwrap.dedent("""\
            import pandas as pd
            data = pd.DataFrame({"status": ["Active", "inactive", "ACTIVE", "Inactive"]})
            # filter active users (case-insensitive)
            active = data[data["status"] == "active"]
        """),
        "fixed_code": textwrap.dedent("""\
            import pandas as pd
            data = pd.DataFrame({"status": ["Active", "inactive", "ACTIVE", "Inactive"]})
            # filter active users (case-insensitive)
            active = data[data["status"].str.lower() == "active"]
        """),
        "check_expr": "len(active) == 2",
    },
]

# Remove the "no-bug" variant (task index 4, the correct ffill) to avoid confusing the model
_TASKS = [t for t in _TASKS if not (t["bug_type"] == "fillna_method" and "result = s.ffill()" in t["fixed_code"] and "result = s.ffill()" in t["buggy_code"])]


# ---------------------------------------------------------------------------
# Safe execution helper
# ---------------------------------------------------------------------------

def _run_code_safe(code: str, check_expr: str, timeout: int = 10) -> tuple[bool, str]:
    """
    Run `code` + `check_expr` in an isolated subprocess. Returns (passed, stderr).

    The code and check_expr are passed via environment variables to avoid any
    quoting / indentation issues when embedding multi-line strings in a -c script.
    """
    # Runner script reads the code/expr from env vars to sidestep all quoting issues
    runner = (
        "import sys, os, traceback\n"
        "code = os.environ['_VF_CODE']\n"
        "expr = os.environ['_VF_EXPR']\n"
        "try:\n"
        "    exec_ns = {}\n"
        "    exec(code, exec_ns)\n"
        "    import pandas as pd\n"
        "    import numpy as np\n"
        "    exec_ns.update({'pd': pd, 'np': np})\n"
        "    result_ok = bool(eval(expr, exec_ns))\n"
        "    print('PASS' if result_ok else 'FAIL')\n"
        "except Exception:\n"
        "    traceback.print_exc()\n"
        "    print('ERROR')\n"
    )
    env = {**os.environ, "_VF_CODE": code, "_VF_EXPR": check_expr}
    try:
        proc = subprocess.run(
            [sys.executable, "-c", runner],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        output = (proc.stdout + proc.stderr).strip()
        passed = output.endswith("PASS")
        return passed, proc.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "TimeoutExpired"
    except Exception as e:
        return False, str(e)


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def _extract_fixed_code(text: str) -> str:
    """Extract content between <fixed_code> tags, stripping markdown fences."""
    m = re.search(r"<fixed_code>(.*?)</fixed_code>", text, re.DOTALL)
    if not m:
        return ""
    raw = m.group(1).strip()
    # strip markdown code fences if present
    raw = re.sub(r"^```(?:python)?\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()


def _has_reasoning(text: str) -> bool:
    return bool(re.search(r"<reasoning>.*?</reasoning>", text, re.DOTALL))


def _bug_type_mentioned(text: str, bug_type: str) -> bool:
    """Heuristic: check if the model's reasoning mentions the bug category keyword."""
    keywords = {
        "off_by_one":    ["off.by.one", r"\biloc\b", r"\bhead\b", r"\btail\b", "index"],
        "dtype_cast":    ["dtype", "type", "cast", "astype", "float", "int"],
        "merge_key":     ["join", "merge", "key", "on=", "left", "right", "inner"],
        "agg_axis":      ["axis", "row.wise", "column.wise", "cumsum", "mean"],
        "fillna_method": ["ffill", "bfill", "forward", "backward", "fill"],
        "groupby_reset": ["reset_index", "index", "groupby"],
        "str_strip":     ["strip", "whitespace", "case", "lower", "upper"],
        "sort_ascending": ["ascending", "descending", "sort", "order"],
        "inplace_return": ["inplace", "None", "return"],
        "copy_alias":    ["copy", "view", "chained", "SettingWithCopy"],
    }
    patterns = keywords.get(bug_type, [])
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in patterns)


async def correctness_reward(completion: vf.Messages, answer: str, **kwargs) -> float:
    """
    Execute the fixed code and check against ground truth.
    0.0  — no <fixed_code> block or code fails to parse
    0.25 — code is syntactically valid Python
    0.5  — code runs without error but check_expr fails
    1.0  — code passes check_expr
    """
    # answer field is JSON: {"fixed_code": ..., "check_expr": ...}
    try:
        meta = json.loads(answer)
    except Exception:
        return 0.0

    fixed_code_gt = meta["fixed_code"]
    check_expr    = meta["check_expr"]

    # Get model's response text
    if isinstance(completion, list):
        last = completion[-1]
        response_text = last.get("content", "") if isinstance(last, dict) else str(last)
    else:
        response_text = str(completion)

    extracted = _extract_fixed_code(response_text)
    if not extracted:
        return 0.0

    # Syntax check
    try:
        ast.parse(extracted)
    except SyntaxError:
        return 0.0

    score = 0.25  # syntactically valid

    # Run the model's fixed code against the check expression
    passed, stderr = _run_code_safe(extracted, check_expr)
    if passed:
        return 1.0

    # Partial: also check if the GROUND TRUTH passes (sanity; should always be True)
    gt_passed, _ = _run_code_safe(fixed_code_gt, check_expr)
    if not gt_passed:
        # our check_expr itself is broken; give benefit of the doubt
        return 0.5

    return score


async def format_reward(completion: vf.Messages, **kwargs) -> float:
    """
    0.0 if neither <reasoning> nor <fixed_code> tags are present.
    0.5 if one tag is present.
    1.0 if both tags are present.
    """
    if isinstance(completion, list):
        last = completion[-1]
        text = last.get("content", "") if isinstance(last, dict) else str(last)
    else:
        text = str(completion)

    has_r = _has_reasoning(text)
    has_f = bool(re.search(r"<fixed_code>.*?</fixed_code>", text, re.DOTALL))
    return float(has_r) * 0.5 + float(has_f) * 0.5


async def reasoning_quality_reward(completion: vf.Messages, answer: str, **kwargs) -> float:
    """
    Bonus reward (weight=0.1) if the model's reasoning mentions the correct bug category.
    """
    try:
        meta = json.loads(answer)
    except Exception:
        return 0.0

    bug_type = meta.get("bug_type", "")
    if isinstance(completion, list):
        last = completion[-1]
        text = last.get("content", "") if isinstance(last, dict) else str(last)
    else:
        text = str(completion)

    return 1.0 if _bug_type_mentioned(text, bug_type) else 0.0


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def _build_dataset(tasks: list[dict[str, str]], seed: int = 42) -> Dataset:
    """Convert task dicts into a HuggingFace Dataset in the verifiers format."""
    rows = []
    for i, task in enumerate(tasks):
        answer_meta = json.dumps({
            "fixed_code": task["fixed_code"],
            "check_expr": task["check_expr"],
            "bug_type":   task["bug_type"],
        })
        rows.append({
            "question": task["buggy_code"],
            "answer":   answer_meta,
            "info":     {"bug_type": task["bug_type"], "task_idx": i},
        })
    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Environment loader — public entry point
# ---------------------------------------------------------------------------

def load_environment(
    seed: int = 42,
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    system_prompt: str = SYSTEM_PROMPT,
) -> vf.Environment:
    """
    Load the pandas-debugger RL environment.

    Args:
        seed: Random seed for dataset shuffling.
        num_train_examples: Limit training set size (-1 = all).
        num_eval_examples:  Limit eval set size (-1 = all).
        system_prompt: Override the default system prompt.

    Returns:
        A :class:`verifiers.SingleTurnEnv` instance ready for RL training or eval.
    """
    tasks = list(_TASKS)  # shallow copy so callers can't mutate the module-level list

    def build_train_dataset() -> Dataset:
        ds = _build_dataset(tasks, seed=seed)
        ds = ds.shuffle(seed=seed)
        if num_train_examples > 0:
            ds = ds.select(range(min(num_train_examples, len(ds))))
        return ds

    def build_eval_dataset() -> Dataset:
        ds = _build_dataset(tasks, seed=seed + 1)
        ds = ds.shuffle(seed=seed + 1)
        if num_eval_examples > 0:
            ds = ds.select(range(min(num_eval_examples, len(ds))))
        return ds

    parser = vf.XMLParser(fields=["reasoning", "fixed_code"], answer_field="fixed_code")

    rubric = vf.Rubric(
        funcs=[
            correctness_reward,
            format_reward,
            reasoning_quality_reward,
        ],
        weights=[1.0, 0.2, 0.1],
        parser=parser,
    )

    env = vf.SingleTurnEnv(
        dataset=build_train_dataset,
        eval_dataset=build_eval_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )
    return env
