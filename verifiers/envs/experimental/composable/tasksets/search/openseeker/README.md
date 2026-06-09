# OpenSeeker Search Taskset

Composable search taskset for [`PolarSeeker/OpenSeeker-v1-Data`](https://huggingface.co/datasets/PolarSeeker/OpenSeeker-v1-Data), the OpenSeeker v1 release associated with arXiv `2603.15594`.

OpenSeeker v1 data contains synthesized deep-search QA pairs plus trajectories generated with `search` and `visit` tools. The public OpenSeeker evaluator scores only the final answer: it sends the question, gold answer, and model response to an LLM judge and expects `A` for correct or `B` for incorrect. This backend preserves that binary semantic answer-judge contract.

By default, the taskset uses the full dataset. Use the shared `filter_fn`
argument for row subsets such as source trajectory quality or tool-call count.
The `trajectory_correctness` metadata describes the stored OpenSeeker source
trajectory, not the validity of the question or gold answer.

## Usage

```python
from verifiers.envs.experimental.composable.tasksets.search import make_search_taskset

taskset = make_search_taskset(backend="openseeker")

correct_source_trajectories = make_search_taskset(
    backend="openseeker",
    filter_fn="lambda x: x['info']['trajectory_correctness'] == 'Correct'",
)

shorter_source_trajectories = make_search_taskset(
    backend="openseeker",
    filter_fn="lambda x: (x['info']['number_of_tool_calls'] or 0) <= 20",
)
```

## Arguments

| Argument | Default | Description |
|---|---:|---|
| `dataset_name` | `PolarSeeker/OpenSeeker-v1-Data` | Hugging Face dataset name. |
| `split` | `train` | Dataset split. |
| `filter_fn` | `None` | Optional composable taskset filter over normalized rows. |
| `include_trajectory` | `False` | Include the large source trajectory in task metadata. |
| `answer_file` | `/task/answer.txt` | Final answer path in the sandbox. |
| `judge_model` | `openai/gpt-5.4-mini` | OpenAI-compatible model used for binary answer judging. |
| `judge_base_url` | `https://api.pinference.ai/api/v1` | Judge API base URL. |
| `judge_api_key_var` | `PRIME_API_KEY` | Env var containing the judge API key. |
| `judge_sampling_args` | `None` | Extra sampling args for judge calls. |

## Output Contract

Agents should write one final answer to `/task/answer.txt`. The answer should directly satisfy the question and may include supporting URLs/citations. The judge ignores citation verification and evaluates whether the final response semantically contains the gold answer without contradictions.
