# pandas-debugger

<a href="https://github.com/PrimeIntellect-ai/verifiers/tree/main/environments/pandas_debugger">
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="Source Code">
</a>

### Overview

- **Environment ID**: `pandas-debugger`
- **Short description**: The model receives a short Python data-wrangling snippet containing exactly one injected bug and must output the corrected code. Rewards are determined by executing the fixed snippet and comparing outputs to ground-truth assertions.
- **Tags**: code, debugging, data-science, pandas, single-turn, xml, python, reasoning

### Why this environment?

Data wrangling is the dominant activity in applied ML pipelines — studies consistently show 60–80 % of a data scientist's time is spent cleaning and transforming data. Debugging subtle pandas bugs (off-by-one slices, wrong join type, dtype coercion, `inplace=True` traps, chained-assignment view aliasing) requires the same careful chain-of-thought reasoning as math problem solving, but applied to code. This environment trains LLMs to:

1. **Identify** the buggy line and its root cause.
2. **Produce** a minimally-invasive fix verified by execution.
3. **Articulate** the fix with clear reasoning.

### Bug Categories

| Category | Description | Example mistake |
| -------- | ----------- | --------------- |
| `off_by_one` | Slice/index boundary error | `iloc[:4]` when `:5` intended |
| `dtype_cast` | Wrong type coercion or missing cast | `.astype(int)` on a float column |
| `merge_key` | Wrong join column or join type | `on="score"` instead of `on="user_id"` |
| `agg_axis` | Aggregation on wrong axis | `mean(axis=0)` for row-wise mean |
| `fillna_method` | Wrong fill direction | `bfill()` when `ffill()` needed |
| `groupby_reset` | Missing `reset_index()` | Result is a Series, not DataFrame |
| `str_strip` | Whitespace or case mismatch | Missing `.str.strip()` / `.str.lower()` |
| `sort_ascending` | Sort direction inverted | `ascending=True` for top-N query |
| `inplace_return` | `inplace=True` reassigned (returns `None`) | `data = data.sort_values(..., inplace=True)` |
| `copy_alias` | Mutating a view instead of a copy | Missing `.copy()` after boolean filter |

### Datasets

- **Primary dataset**: Self-contained task bank (14 curated tasks) embedded in the environment module — no external download required.
- **Format**: Each sample has a `question` (buggy code) and `answer` (JSON with `fixed_code`, `check_expr`, `bug_type`).

### Task

- **Type**: Single-turn, code reasoning
- **Model output format**:
  ```xml
  <reasoning>
  Explanation of the bug and the fix.
  </reasoning>
  <fixed_code>
  # corrected Python code
  </fixed_code>
  ```
- **Verification**: The extracted `<fixed_code>` block is executed in an isolated subprocess against a boolean check expression derived from the ground-truth pipeline.

### Reward Structure

| Reward Function | Weight | Description |
| --------------- | ------ | ----------- |
| `correctness_reward` | 1.0 | 0.0 = no code / syntax error; 0.25 = valid syntax; 0.5 = runs but wrong; 1.0 = passes check |
| `format_reward` | 0.2 | 0.5 per XML tag present (`<reasoning>` + `<fixed_code>`) |
| `reasoning_quality_reward` | 0.1 | 1.0 if reasoning text mentions the correct bug category keywords |

### Quickstart

Run an evaluation with default settings:

```bash
prime eval run pandas-debugger
```

Configure model and sampling:

```bash
prime eval run pandas-debugger \
  -m openai/gpt-4.1-mini \
  -n 14 -r 3 -t 1024 -T 0.7 \
  -a '{"seed": 42, "num_eval_examples": 14}'
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `seed` | int | `42` | RNG seed for dataset shuffling |
| `num_train_examples` | int | `-1` | Training set size limit (-1 = all 14) |
| `num_eval_examples` | int | `-1` | Eval set size limit (-1 = all 14) |
| `system_prompt` | str | (built-in) | Override the default system prompt |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `correctness_reward` | Primary execution-based correctness score |
| `format_reward` | Structural format adherence |
| `reasoning_quality_reward` | Bug-category identification quality |
| `num_turns` | Always 1 (single-turn environment) |

### Design Notes

- **No external sandbox needed** — execution is sandboxed via `subprocess.run` with a timeout, using only the stdlib + pandas/numpy which are declared dependencies.
- **Graded reward signal** — the 0/0.25/0.5/1.0 ladder provides a richer learning signal than binary correct/wrong, helping RL training converge faster.
- **Extensible task bank** — the `_TASKS` list is a plain Python list of dicts; adding new bug types requires no framework changes.
- **Reasoning bonus** — the 0.1-weight `reasoning_quality_reward` encourages models to develop explicit chain-of-thought debugging before producing code.
