# REDSearcher Search Taskset

Text RL queries from REDSearcher ported into the composable search taskset framework.

## Source

- Dataset: [`Zchu/REDSearcher_RL_1K`](https://huggingface.co/datasets/Zchu/REDSearcher_RL_1K)
- Collection: [`Zchu/redsearcher`](https://huggingface.co/collections/Zchu/redsearcher)
- Upstream project: [`RedSearchAgent/REDSearcher`](https://github.com/RedSearchAgent/REDSearcher)
- Paper: [`arXiv:2602.14234`](https://arxiv.org/abs/2602.14234)

The released text RL dataset contains 1,000 rows with `problem`, `answer`, and `difficulty` columns. The upstream REDSearcher repo describes converting each row into a Slime-style `prompt` plus `label`; this taskset keeps the same problem/answer boundary while adapting it to Verifiers' taskset format.

## Task Contract

Each example is a long-horizon web-search question. The agent should research across sources and produce one final answer in `/task/answer.txt`, with supporting URLs/citations when available.

The paired `rlm_search` environment prompts RLM to write this file and provides web search/open-page skills. The rubric can fall back to the final assistant text if the answer file is empty, but agents should still write the file directly.

## Scoring

`RedSearcherRubric` compares the final response against the released `answer` label. It first applies a strict normalized exact-answer shortcut for unambiguous matches. Otherwise it uses an OpenAI-compatible LLM-as-judge prompt that matches REDSearcher's released DeepTraceHub BROWSECOMP evaluator prompt and returns binary accuracy.

A reward of `1.0` means the final response matched the ground-truth answer; `0.0` means it did not, or no final answer was produced. Judge provider failures are preserved as `vf.Error` values on `state["error"]`.

## Common Arguments

| Argument | Default | Description |
|---|---:|---|
| `dataset_name` | `Zchu/REDSearcher_RL_1K` | Hugging Face dataset name. |
| `split` | `train` | Dataset split. |
| `difficulty` | `None` | Optional difficulty filter: `easy`, `medium`, `hard`, or `all`. |
| `answer_file` | `/task/answer.txt` | Final answer path in the sandbox. |
| `judge_model` | `openai/gpt-5.4-mini` | OpenAI-compatible model for answer-match judging. |
| `judge_base_url` | `https://api.pinference.ai/api/v1` | Judge API base URL. |
| `judge_api_key_var` | `PRIME_API_KEY` | Env var containing the judge API key. |
| `judge_max_retries` | `5` | Number of parse retries for the A/B judge response. |
| `use_exact_match_shortcut` | `True` | Return `1.0` without an LLM call when the normalized final response exactly equals the normalized ground-truth answer. |
