# OpenSeeker Search Taskset

Composable search taskset for [`PolarSeeker/OpenSeeker-v1-Data`](https://huggingface.co/datasets/PolarSeeker/OpenSeeker-v1-Data), the OpenSeeker v1 release associated with arXiv `2603.15594`.

OpenSeeker v1 data contains synthesized deep-search QA pairs plus trajectories generated with `search` and `visit` tools. The public OpenSeeker evaluator scores only the final answer: it sends the question, gold answer, and model response to an LLM judge and expects `A` for correct or `B` for incorrect. This backend preserves that binary semantic answer-judge contract.

## Usage

```python
from verifiers.envs.experimental.composable.tasksets.search import make_search_taskset

taskset = make_search_taskset(backend="openseeker")
```

## Arguments

| Argument | Default | Description |
|---|---:|---|
| `dataset_name` | `PolarSeeker/OpenSeeker-v1-Data` | Hugging Face dataset name. |
| `split` | `train` | Dataset split. |
| `trajectory_correctness` | `Correct` | Keep rows with this trajectory label. Use `None` or `all` for all rows. |
| `min_tool_calls` | `None` | Optional lower bound for `number of tool calls`. |
| `max_tool_calls` | `None` | Optional upper bound for `number of tool calls`. |
| `include_trajectory` | `False` | Include the large source trajectory in task metadata. |
| `answer_file` | `/task/answer.txt` | Final answer path in the sandbox. |
| `judge_model` | `openai/gpt-5.4-mini` | OpenAI-compatible model used for binary answer judging. |
| `judge_base_url` | `https://api.pinference.ai/api/v1` | Judge API base URL. |
| `judge_api_key_var` | `PRIME_API_KEY` | Env var containing the judge API key. |
| `judge_sampling_args` | `None` | Extra sampling args for judge calls. |

## Output Contract

Agents should write one final answer to `/task/answer.txt`. The answer should directly satisfy the question and may include supporting URLs/citations. The judge ignores citation verification and evaluates whether the final response semantically contains the gold answer without contradictions.
