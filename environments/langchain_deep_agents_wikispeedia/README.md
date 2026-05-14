# langchain-deep-agents-wikispeedia

LangChain deep-agents trained on Wikispeedia navigation through a v1 `Taskset`/`Harness`.

### Overview
- **Environment ID**: `langchain-deep-agents-wikispeedia`
- **Short description**: Multi-turn navigation through the Wikispeedia article graph with LangChain `create_deep_agent` (todos, virtual files, sub-agents) plus two task tools (`click_link`, `go_back`).
- **Tags**: v1, taskset, harness, multi-turn, tool-use, langchain, deep-agents, wikispeedia, navigation

### Datasets
- **Source**: SNAP Wikispeedia ([snap.stanford.edu/data/wikispeedia](https://snap.stanford.edu/data/wikispeedia.html)) — 4,604 Wikipedia articles, ~120K hyperlinks, precomputed shortest-path distance matrix, plus aggregate human-play stats.
- **Splits**: 50K train pairs / 1K eval pairs, sampled evenly across shortest-path buckets within `min_path_length..max_path_length`. Train and eval target articles are **disjoint** (no target ever crosses splits). Deterministic via `split_seed`.

### Task
- **Type**: `vf.Env` with a Wikispeedia `vf.Taskset` and LangChain Deep Agents `vf.Harness`
- **Goal**: navigate from a source Wikipedia article to a target article using only on-page hyperlinks.
- **Boundary**: the taskset owns the Wikispeedia graph, `click_link`/`go_back` tools, rewards, and metrics; the harness only adapts the resolved taskset tools into LangChain Deep Agents.
- **Output format**: agent calls `click_link(article)` until the target is reached. The `TARGET REACHED` tool message tells the agent to stop and reply briefly.
- **Scoring**: binary `reached_target` reward plus zero-weight path/tool metrics. `path_efficiency` becomes a weighted reward when `efficiency_weight > 0`.

### Quickstart

Install the env locally:
```bash
prime env install ./environments/langchain_deep_agents_wikispeedia
```

Run an evaluation with default settings:
```bash
prime eval run langchain-deep-agents-wikispeedia
```

Configure model and difficulty band:
```bash
prime eval run langchain-deep-agents-wikispeedia \
  -m openai/gpt-4.1-mini \
  -n 20 -r 3 -t 4096 -T 0.7 \
  -a '{"config": {"taskset": {"min_path_length": 4, "max_path_length": 6, "max_turns": 40}}}'
```

Disable `go_back` (force planning over backtracking):
```bash
prime eval run langchain-deep-agents-wikispeedia \
  -m openai/gpt-4.1-mini -n 20 -r 3 \
  -a '{"config": {"taskset": {"allow_go_back": false}}}'
```

Notes:
- The first run downloads ~5MB of SNAP data into `~/.cache/wikispeedia` (override with `cache_dir`).
- Set `OPENAI_API_KEY` (or whatever the policy endpoint expects) for the agent.

### Taskset Config

| Field | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `cache_dir` | str \| None | `None` | SNAP cache directory (defaults to `~/.cache/wikispeedia`). |
| `min_path_length` | int | `3` | Drop pairs with shortest path shorter than this. |
| `max_path_length` | int | `6` | Drop pairs with shortest path longer than this (only ~470 pairs exist at dist=8, 5 at dist=9). |
| `train_size` | int | `50000` | Number of train pairs to sample. |
| `eval_size` | int | `1000` | Number of eval pairs to sample. |
| `eval_target_fraction` | float | `0.1` | Fraction of articles reserved as eval-only targets. |
| `split_seed` | int | `0` | Seed for deterministic train/eval split. |
| `links_only` | bool | `False` | Render articles as just the link menu (ablation: tests whether the agent navigates from semantic content or link names alone). |
| `allow_go_back` | bool | `True` | Expose the `go_back` tool. |
| `max_turns` | int | `50` | Per-rollout turn cap. |
| `efficiency_weight` | float | `0.0` | If `> 0`, mix `path_efficiency` into the reward at this weight (a near-optimal route earns up to `1 + efficiency_weight`; a wanderer that reaches the target still earns `1`). Default `0.0` keeps reward as pure binary reachability. |
| `stratify_path_length` | bool | `True` | Take equal counts at each shortest-path bucket inside `[min_path_length, max_path_length]`, capped at the smallest non-empty bucket. The SNAP graph's natural distribution heavily skews toward the lower end of any band (4-6 → 83% sp=4); without stratification the policy over-trains on the trivial floor. Set `False` to recover the natural distribution. |

### Harness Config

| Field | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `max_turns` | int | `50` | LangChain recursion limit fallback when runtime config does not provide one. |
| `timeout_seconds` | float | `1200.0` | Per-rollout wall-clock cap. |

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `reward` | weighted sum (defaults to `reached_target`) |
| `reached_target` | 1.0 if the agent navigated to the target (always a weighted reward; weight 1.0) |
| `path_efficiency` | `shortest_path / actual_path_length` if reached, else 0. Zero-weight by default; becomes a weighted reward at `efficiency_weight` when that arg is `> 0` |
| `path_length` | number of edges traversed (zero-weight) |
| `shortest_path` | precomputed shortest path length for the pair (zero-weight) |
| `agent_timeout` | 1.0 if rollout hit `timeout_seconds` |
| `calls_click_link`, `calls_go_back` | navigation tool counts (zero-weight) |
| `calls_write_todos`, `calls_write_file`, `calls_read_file`, `calls_ls`, `calls_edit_file`, `calls_grep`, `calls_task` | deep-agent tool counts (zero-weight) |
| `total_tool_calls`, `assistant_turns` | trajectory shape (zero-weight) |
| `invalid_link_rate` | fraction of `click_link` calls that named a non-existent link (hallucination canary, zero-weight) |

### Notes
- Reward is `reached_target` only — exact, deterministic, no judge required. The deep-agent structural metrics are zero-weight so they show up in eval tables without shaping the policy.
- `min_path_length=4, max_path_length=6` is the calibrated RL difficulty band for Nemotron-30B-A3B-BF16 — predicted ~0.3-0.4 reach rate, the useful-gradient zone. The 3-5 band landed at 0.61 mean reach (dominated by the trivial sp=3 floor where the deep-agent scaffolding is decorative); the 5-7 band landed at 0.13 with 27% timeouts.
- This is the primary LangChain Deep Agents example because tool use is load-bearing: the model cannot reach the target without invoking `click_link`.
