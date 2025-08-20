# vf-wiki-search

### Overview
- **Environment ID**: `vf-wiki-search`
- **Short description**: Multi-turn tool-use QA over a small Wikipedia corpus using ChromaDB and OpenAI embeddings, with judge-based scoring.
- **Tags**: retrieval, tools, multi-turn, embeddings, judge

### Datasets
- **Primary dataset(s)**: `willcb/wiki-trivia-questions` (HF) and a local wiki markdown corpus indexed in ChromaDB
- **Source links**: Hugging Face Datasets, ChromaDB
- **Split sizes**: Uses the `train` split for prompts; corpus is on-disk

### Task
- **Type**: multi-turn tool use
- **Parser**: XML-like tool formatting inside `<tool>` blocks; final answer in `<answer>`
- **Rubric overview**: Combines the default tool rubric with a `JudgeRubric` for answer quality

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval vf-wiki-search
```

Configure model and sampling:

```bash
uv run vf-eval vf-wiki-search \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"judge_model": "gpt-4.1-mini", "judge_base_url": "https://api.openai.com/v1", "judge_api_key_var": "OPENAI_API_KEY", "embed_model": "text-embedding-3-small", "embed_base_url": "https://api.openai.com/v1", "embed_api_key_var": "OPENAI_API_KEY"}'
```

Notes:
- Requires a prebuilt ChromaDB at `notebooks/.chroma_db` and markdown pages under `notebooks/data/wiki`.
- Reports are written under `./environments/vf_wiki_search/reports/` and auto-embedded below.

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `judge_model` | str | `"gpt-4.1-mini"` | Judge model name |
| `judge_base_url` | str | `"https://api.openai.com/v1"` | Judge provider base URL |
| `judge_api_key_var` | str | `"OPENAI_API_KEY"` | Env var for judge API key |
| `embed_model` | str | `"text-embedding-3-small"` | Embedding model name |
| `embed_base_url` | str | `"https://api.openai.com/v1"` | Embedding provider base URL |
| `embed_api_key_var` | str | `"OPENAI_API_KEY"` | Env var for embed API key |
| `wiki_dir` | str | `notebooks/data/wiki` | Path to markdown wiki pages |
| `chroma_db_dir` | str | `notebooks/.chroma_db` | Path to ChromaDB index |

### Metrics
| Metric | Meaning |
| ------ | ------- |
| ToolRubric metrics | Tool execution success and format adherence |
| JudgeRubric metrics | Judge-scored answer quality |

