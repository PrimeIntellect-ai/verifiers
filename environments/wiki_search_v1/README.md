# wiki-search-v1

<a href="https://github.com/PrimeIntellect-ai/verifiers/tree/main/environments/wiki_search_v1">
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="Source Code">
</a>

### Overview
- **Environment ID**: `wiki-search-v1`
- **Short description**: Multi-turn tool-use QA over a small Wikipedia corpus using a v1 task-owned MCP toolset.
- **Tags**: retrieval, tools, multi-turn, v1

### Datasets
- **Primary dataset(s)**: `willcb/wiki-trivia-questions` (HF) and a Wikipedia corpus from `willcb/rare-wiki-pages`
- **Source links**: Hugging Face Datasets
- **Split sizes**: Uses the `train` split for prompts

### Task
- **Type**: `vf.Env` with a wiki QA `vf.Taskset`, base `vf.Harness`, and env-scope wiki `Toolset`.
- **Rubric overview**: Answer-substring reward against the reference answer.

### How it works
- **Corpus load**: Reads `willcb/rare-wiki-pages` (HF) into memory: `id → title`, `id → content`.
- **Tools**:
  - `search_pages(query)`: Lexical ranking over page titles and content; returns top 10 `{page_id, title}`.
  - `view_sections(page_id)`: Parses the page content for Markdown-style headings (`# ...`) and returns section ids/names. Falls back to a single `full` section if no headings.
  - `read_section(section_id)`: Returns the content slice for the requested section (or full page).
- **Scoring**: The taskset rewards final assistant responses that contain the reference answer.

### Quickstart
Run an evaluation with default settings:

```bash
prime eval run wiki-search-v1
```

Configure model and sampling:

```bash
prime eval run wiki-search-v1 \
  -m openai/gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"config": {"taskset": {"max_turns": 10, "toolsets": {"wiki": {"corpus_dataset": "willcb/rare-wiki-pages", "corpus_split": "train"}}}}}'
```

### Required Environment Variables

No task-specific environment variables are required.

### Taskset Config
| Field | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `max_turns` | int | `10` | Maximum model turns per rollout |

### Wiki Toolset Config
| Field | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `corpus_dataset` | str | `"willcb/rare-wiki-pages"` | HF dataset id containing pages |
| `corpus_split` | str | `"train"` | HF split to load |

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `reward` | 1.0 if the final assistant response contains the reference answer, else 0.0 |
| `num_turns` | Number of recorded model turns |
