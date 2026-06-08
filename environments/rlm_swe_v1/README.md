# rlm-swe-v1

### Overview
- **Environment ID**: `rlm-swe-v1`
- **Short description**: v1 RLM coding environment on R2E-Gym SWE tasks
- **Tags**: rlm, swe, cli-agent, v1

### Datasets
- **Primary dataset(s)**: `R2E-Gym/R2E-Gym-Subset`
- **Source links**: [R2E-Gym/R2E-Gym-Subset](https://huggingface.co/datasets/R2E-Gym/R2E-Gym-Subset)
- **Split sizes**: Uses the dataset `train` split by default

### Task
- **Type**: multiturn, cli_agent
- **Rubric overview**: Runs each instance's hidden test command and parses pytest output for pass/fail reward

### Quickstart
Run an evaluation with default settings:

```bash
prime eval run rlm-swe-v1
```

Configure model and sampling:

```bash
prime eval run rlm-swe-v1 \
  -m openai/gpt-4.1-mini \
  -n 5 -r 1 -t 4096 -T 0.2 \
  -a '{"config": {"taskset": {"timeout_minutes": 90}, "harness": {"tools": ["bash", "edit"], "cwd": "/testbed"}}}'
```

Notes:
- v1 task settings belong under `config.taskset`; reusable RLM agent settings belong under `config.harness`.
- The taskset is discovered from `taskset.py`; the harness is discovered from `harness.py`.

### Taskset Config
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset_name` | str | `"R2E-Gym/R2E-Gym-Subset"` | Dataset to load |
| `repo_path` | str | `"/testbed"` | Repository path inside the sandbox |
| `filter_repos` | list[str] \| null | `null` | Repositories to exclude |
| `ds_num_proc` | int \| null | `null` | Dataset processing parallelism |
| `ds_keep_in_memory` | bool | `true` | Keep processed dataset rows in memory |
| `timeout_minutes` | int \| null | `null` | Override task runtime timeout |
| `env` | object \| null | `null` | Extra task program environment values |

### Harness Config
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `cwd` | str \| null | `"/testbed"` | Agent working directory |
| `tools` | list[str] | `["bash", "edit"]` | RLM tool names exposed to the agent |
| `exec_timeout` | int | `300` | RLM tool execution timeout |
| `max_depth` | int | `0` | RLM recursive agent depth |
| `summarize_at_tokens` | int \| null | `null` | Optional RLM summarization threshold |
| `append_to_system_prompt` | str | `""` | Additional RLM system prompt text |

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `reward` | Parsed hidden-test reward |
| `rlm_sub_llm_call_count` | RLM sub-model call count |
| `rlm_sub_llm_total_turns` | Total RLM sub-model turns |
| `rlm_sub_llm_total_tool_calls` | Total RLM sub-agent tool calls |

### How It Works
1. `R2ESWETaskset` loads R2E-Gym rows and converts them into typed v1 tasks with sandbox/runtime metadata.
2. `RLM` owns the reusable CLI command, endpoint interception, and agent configuration.
3. The v1 runtime resolves task and harness runtime settings at rollout time.
4. Reward is computed from hidden test output after the agent finishes.
