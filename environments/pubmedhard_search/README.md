# pubmedhard-search

### Overview
- **Environment ID**: `pubmedhard-search`
- **Short description**: Multi-turn tool-use QA over PubMed OA Markdown contexts with a lightweight BM25 retriever and token-F1 scoring (optional judge).
- **Tags**: biomedical, pmc, retrieval, tools, multi-turn, rag, judge

### Datasets
- **Primary dataset(s)**: `casperhansen/pmc-oa-markdown-qa` (HF)
- **Source links**: Hugging Face Datasets; PMC processing utility [`pmc-python`](https://github.com/casper-hansen/pmc-python)

### Task
- **Type**: multi-turn tool use
- **Rubric overview**: Default ToolRubric + TokenF1 rubric; JudgeRubric included (0-weight by default)

### How it works
- **Corpus load**: Loads QA rows; concatenates `context_list` or `context` as Markdown text.
- **Indexing**: In-memory BM25-like index built on first load; fast and dependency-light.
- **Tools**:
  - `search(query) -> [{idx}]`: returns top-k row indices ranked by BM25.
  - `read(idx) -> str`: returns the Markdown context for the given row index.
- **Scoring**: Token-level F1 vs gold answer; optional judge rubric included.

### Quickstart
```bash
uv run vf-eval pubmedhard-search -m gpt-4.1-mini -n 5 -r 1 \
  -a '{"dataset_name":"casperhansen/pmc-oa-markdown-qa","dataset_split":"train"}'
```

### Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `judge_model` | str | `"gpt-4.1-mini"` | Judge model name |
| `judge_base_url` | str | `"https://api.openai.com/v1"` | Judge provider base URL |
| `judge_api_key_var` | str | `"OPENAI_API_KEY"` | Env var for judge API key |
| `dataset_name` | str | `"casperhansen/pmc-oa-markdown-qa"` | QA dataset id |
| `dataset_split` | str | `"train"` | HF split to load |
| `max_turns` | int | `10` | Max tool-use turns |

### References
- Wiki search template: [`wiki_search.py`](https://github.com/PrimeIntellect-ai/verifiers/blob/main/environments/wiki_search/wiki_search.py)
- PMC parsing utility: [`pmc-python`](https://github.com/casper-hansen/pmc-python)
- Dataset: [`pmc-oa-markdown-qa`](https://huggingface.co/datasets/casperhansen/pmc-oa-markdown-qa)


