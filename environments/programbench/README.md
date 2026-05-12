# programbench

### Overview
- **Environment ID**: `programbench`
- **Short description**: Agent reconstructs compilable source code from an execute-only binary and its documentation, scored by hidden pytest tests.
- **Tags**: reverse-engineering, multi-turn, sandbox, binary-analysis, eval

### Datasets
- **Primary dataset(s)**: `PrimeIntellect/programbench-processed` — 195 tasks (C, C++, Go, Rust) with README, binary on HF, and hidden pytest test archives
- **Source links**: [ProgramBench paper](https://arxiv.org/abs/2503.13066)
- **Split sizes**: 195 tasks (train split): 32 C, 11 C++, 46 Go, 106 Rust

### Task
- **Type**: multi-turn, tool use (mini-SWE-agent)
- **Output format**: source files written to `/workspace/src/` + `compile.sh` that produces `/workspace/executable`
- **Rubric overview**: `solved = n_tests_passed / n_tests_total` from hidden pytest suite; reward = solved (weight 1.0)

### Quickstart

```bash
# Requires: HF_TOKEN (private dataset + test archives), OPENAI_API_KEY
prime eval run programbench -m openai/gpt-4.1-mini -n 5 -r 1
```

Filter by language or difficulty:

```bash
prime eval run programbench -m openai/gpt-4.1 -n 20 -r 1 \
  -a '{"filter_language": "go", "max_tasks": 10}'
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `filter_language` | str | `null` | Restrict to `"c"`, `"cpp"`, `"go"`, or `"rust"` |
| `filter_difficulty` | str | `null` | Restrict to a difficulty tier |
| `max_tasks` | int | `null` | Cap number of tasks loaded |
| `hide_tests_from_agent` | bool | `true` | Keep test archive hidden until scoring |
| `dataset_name` | str | `PrimeIntellect/programbench-processed` | HF dataset ID |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` / `solved` | Fraction of hidden pytest tests passed (0–1) |
| `compile_success` | Whether `compile.sh` produced an executable |
| `compile_exit_code` | Exit code of the compile step |
| `n_tests_passed` | Raw count of passing tests |
| `n_tests_total` | Total tests in the hidden suite |
| `pytest_log` | Last 4KB of pytest output |
