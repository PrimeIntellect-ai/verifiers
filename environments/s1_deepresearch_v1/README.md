# s1-deepresearch-v1

A `verifiers.v1` taskset for ScienceOne-AI's [`S1-DeepResearch-15k`](https://huggingface.co/datasets/ScienceOne-AI/S1-DeepResearch-15k) deep-research dataset: answer multi-hop research questions by searching the web, scored against the released ground truth.

## What it does

- **Tasks** — loads the **verifiable** (`Closed-ended Multi-hop Resolution`) subset of the dataset, each a multi-hop question with a ground-truth `answer` (English + Chinese).
- **Tools** — a shared, read-only `vf.Toolset` exposing `web_search` (Serper) and `web_visit` (fetch + parse a URL, HTML or PDF).
- **Scoring** — one `@reward`: a normalized exact-match shortcut, then an LLM-as-judge (the BROWSECOMP answer-match prompt) returning binary correctness (`1.0`/`0.0`). A `@metric` records whether a final answer was produced.

## Loading note

The upstream repo declares its `meta` column with the `Json` feature type, which only exists in `datasets>=4.7` (verifiers pins `datasets<4.7`), so `load_dataset` raises `Feature type 'Json' not found`. This taskset downloads the raw `data.jsonl` (cached via `huggingface_hub`) and parses it line by line.

## Run

```bash
# needs SERPER_API_KEY (search) and PRIME_API_KEY (judge, default Prime inference)
SERPER_API_KEY=... PRIME_API_KEY=... uv run eval s1-deepresearch-v1 -n 5 -r 2 --harness.id rlm
```

## Config (`--taskset.*`)

| flag | default | meaning |
|---|---|---|
| `dataset_name` | `ScienceOne-AI/S1-DeepResearch-15k` | HF dataset repo. |
| `data_file` | `data.jsonl` | Raw JSONL artifact parsed from the repo. |
| `verifiable_only` | `True` | Keep only closed-ended rows (those with a ground-truth answer). |
| `language` | `None` | Optional language filter (`en` / `zh`). |
| `max_examples` | `None` | Optional cap on the number of tasks loaded. |
| `use_exact_match_shortcut` | `True` | Skip the judge when the normalized answer matches exactly. |
| `judge.model` | `openai/gpt-5.4-mini` | OpenAI-compatible judge model. |
| `judge.base_url` / `judge.api_key_var` | Prime inference / `PRIME_API_KEY` | Judge endpoint (inherited from `BaseClientConfig`). |
| `tools.shared` | `True` | One web-search server for the whole eval (stateless tools). |
| `tools.num_results` | `5` | Organic Serper results per `web_search`. |

> The first load downloads the full ~1.4 GB `data.jsonl` (then cached); `max_examples` caps the tasks built, not the download.
