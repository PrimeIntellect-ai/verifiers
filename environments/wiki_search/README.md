# wiki-search

<a href="https://github.com/PrimeIntellect-ai/verifiers/tree/main/environments/wiki_search">
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="Source Code">
</a>

### Overview
- **Environment ID**: `wiki-search`
- **Short description**: Multi-turn tool-use QA over a small Wikipedia corpus using ChromaDB and OpenAI embeddings, with judge-based scoring.
- **Tags**: retrieval, tools, multi-turn, embeddings, judge

### Datasets
- **Primary dataset(s)**: `willcb/wiki-trivia-questions` (HF) and a Wikipedia corpus indexed in ChromaDB (from `willcb/rare-wiki-pages`, indexed at `.chroma_db` on first run)
- **Source links**: Hugging Face Datasets, ChromaDB
- **Split sizes**: Uses the `train` split for prompts

### Task
- **Type**: multi-turn tool use
- **Rubric overview**: Combines the default tool rubric with a `JudgeRubric` for answer quality

### How it works
- **Corpus load**: Reads `willcb/rare-wiki-pages` (HF) into memory: `id → title`, `id → content`.
- **Indexing**: Creates/opens a persistent Chroma collection `wiki_titles` under `.chroma_db`, using OpenAI embeddings to index page titles. Missing titles are upserted in small batches on first run.
- **Tools**:
  - `search_pages(query)`: Embedding search over titles; returns top 10 `{page_id, title}`.
  - `view_sections(page_id)`: Parses the page content for Markdown-style headings (`# ...`) and returns section ids/names. Falls back to a single `full` section if no headings.
  - `read_section(section_id)`: Returns the content slice for the requested section (or full page).
- **Scoring**: Adds a `JudgeRubric` on top of the default tool rubric for answer quality.

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval wiki-search
```

Configure model and sampling:

```bash
uv run vf-eval wiki-search \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"judge_model": "gpt-4.1-mini", "judge_base_url": "https://api.openai.com/v1", "judge_api_key_var": "OPENAI_API_KEY", "embed_model": "text-embedding-3-small", "embed_base_url": "https://api.openai.com/v1", "embed_api_key_var": "OPENAI_API_KEY"}'
```

Notes:
- Set `OPENAI_API_KEY` in your environment for both judge and embedding calls.
- The first run builds the Chroma index and may take a few minutes.


### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `judge_model` | str | `"gpt-4.1-mini"` | Judge model name |
| `judge_base_url` | str | `"https://api.openai.com/v1"` | Judge provider base URL |
| `judge_api_key_var` | str | `"OPENAI_API_KEY"` | Env var for judge API key |
| `embed_model` | str | `"text-embedding-3-small"` | Embedding model name |
| `embed_base_url` | str | `"https://api.openai.com/v1"` | Embedding provider base URL |
| `embed_api_key_var` | str | `"OPENAI_API_KEY"` | Env var for embed API key |
| `corpus_dataset` | str | `"willcb/rare-wiki-pages"` | HF dataset id containing pages |
| `corpus_split` | str | `"train"` | HF split to load |
| `chroma_db_dir` | str | `.chroma_db` | Path to ChromaDB index |

### Metrics
| Metric | Meaning |
| ------ | ------- |
| ToolRubric metrics | Tool execution success and format adherence |
| JudgeRubric metrics | Judge-scored answer quality |

