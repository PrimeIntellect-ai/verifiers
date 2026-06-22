# S1 DeepResearch Search Taskset

Verifiable closed-ended multi-hop research tasks from ScienceOne-AI's S1 DeepResearch dataset.

## Source

- Dataset: [`ScienceOne-AI/S1-DeepResearch-15k`](https://huggingface.co/datasets/ScienceOne-AI/S1-DeepResearch-15k)
- Upstream project: [`ScienceOne-AI/S1-DeepResearch`](https://github.com/ScienceOne-AI/S1-DeepResearch)

The dataset is distributed as one JSONL file with `meta` and `messages` fields. This taskset intentionally ignores `messages`, keeps only `meta` rows whose `type` is `Closed-ended Multi-hop Resolution`, and defaults to English (`"en"`) instances.

## Task Contract

Each example is a verifiable deep-research question. The agent should produce one final answer in `/task/answer.txt`, with supporting URLs/citations when useful.

## Scoring

`S1DeepResearchTaskSet` reuses the existing REDSearcher answer-match rubric: it compares the final response against `meta["answer"]` with an exact-match shortcut before falling back to an OpenAI-compatible binary semantic judge.

## Common Arguments

| Argument | Default | Description |
|---|---:|---|
| `dataset_name` | `ScienceOne-AI/S1-DeepResearch-15k` | Hugging Face dataset name. |
| `split` | `train` | Dataset split. Only `train` is currently exposed by the dataset. |
| `filter_fn` | `None` | Optional composable taskset filter over normalized rows. |
| `language` | `en` | Language filter. Pass `None` to include all languages. |
| `answer_file` | `/task/answer.txt` | Final answer path in the sandbox. |
| `judge_model` | `openai/gpt-5.4-mini` | OpenAI-compatible model for answer-match judging. |
| `judge_base_url` | `https://api.pinference.ai/api/v1` | Judge API base URL. |
| `judge_api_key_var` | `PRIME_API_KEY` | Env var containing the judge API key. |
| `judge_max_retries` | `5` | Number of parse retries for the A/B judge response. |
