# S1-DeepResearch Search Taskset

Deep-research queries from ScienceOne-AI's `S1-DeepResearch-15k` ported into the composable search taskset framework.

## Source

- Dataset: [`ScienceOne-AI/S1-DeepResearch-15k`](https://huggingface.co/datasets/ScienceOne-AI/S1-DeepResearch-15k)
- Model repo: [`ScienceOne-AI/S1-DeepResearch`](https://huggingface.co/ScienceOne-AI)

The dataset contains ~15,000 rows. Each row has a `meta` object (`id`, `question`, `answer`, `language`, `type`) plus the full reference deep-research `messages` trajectory. Rows come in two flavours via `meta.type`:

- `Closed-ended Multi-hop Resolution` — verifiable tasks that carry a ground-truth `answer`.
- `Open-ended Exploration` — open-ended tasks with no ground-truth `answer`.

This taskset keeps the verifiable closed-ended subset by default (the only rows that can be answer-matched), mirroring how the REDSearcher backend treats its `problem`/`answer` pairs. Questions are in English and Chinese.

## Loading

The upstream repo declares `meta` with the `Json` feature type, which only exists in `datasets>=4.7`. Verifiers pins `datasets<4.7`, so `load_dataset("ScienceOne-AI/S1-DeepResearch-15k")` raises `Feature type 'Json' not found`. To stay robust across the pinned range, this taskset downloads the raw `data.jsonl` artifact (cached via `huggingface_hub`) and parses it line by line, extracting only the `meta` fields it needs.

## Task Contract

Each example is a long-horizon web-search question. The agent should research across sources and produce one final answer in `/task/answer.txt`, with supporting URLs/citations when available.

The paired `rlm_search` environment prompts RLM to write this file and provides web search/open-page skills. The rubric can fall back to the final assistant text if the answer file is empty, but agents should still write the file directly.

## Scoring

`S1DeepResearchRubric` compares the final response against the released `answer` label. It first applies a strict normalized exact-answer shortcut for unambiguous matches. Otherwise it uses an OpenAI-compatible LLM-as-judge prompt (the shared DeepTraceHub BROWSECOMP answer-match evaluator) and returns binary accuracy.

A reward of `1.0` means the final response matched the ground-truth answer; `0.0` means it did not, or no final answer was produced. Judge provider failures are preserved as `vf.Error` values on `state["error"]`.

## Common Arguments

| Argument | Default | Description |
|---|---:|---|
| `dataset_name` | `ScienceOne-AI/S1-DeepResearch-15k` | Hugging Face dataset name. |
| `split` | `train` | Dataset split (only `train` is published). |
| `data_file` | `data.jsonl` | Raw JSONL artifact parsed from the dataset repo. |
| `verifiable_only` | `True` | Keep only `Closed-ended Multi-hop Resolution` rows (those with a ground-truth answer). |
| `language` | `None` | Optional language filter (`en` or `zh`). |
| `max_examples` | `None` | Optional cap on the number of rows kept. |
| `filter_fn` | `None` | Optional composable taskset filter over normalized rows, for example `lambda x: x['info']['language'] == 'en'`. |
| `answer_file` | `/task/answer.txt` | Final answer path in the sandbox. |
| `judge_model` | `openai/gpt-5.4-mini` | OpenAI-compatible model for answer-match judging. |
| `judge_base_url` | `https://api.pinference.ai/api/v1` | Judge API base URL. |
| `judge_api_key_var` | `PRIME_API_KEY` | Env var containing the judge API key. |
| `judge_max_retries` | `5` | Number of parse retries for the A/B judge response. |
| `use_exact_match_shortcut` | `True` | Return `1.0` without an LLM call when the normalized final response exactly equals the normalized ground-truth answer. |
