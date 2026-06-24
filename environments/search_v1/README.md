# search-v1

### Overview
- **Environment ID**: `search-v1`
- **Short description**: Composable search/research tasksets — QUEST, OpenSeeker, and REDSearcher — as a native Verifiers v1 `vf.Taskset`. Harness-agnostic: the agent searches the web and writes its final answer to `/task/answer.txt`, which the taskset reads out of the live runtime and grades.
- **Tags**: search, research, qa, rlm, sandbox, quest, openseeker, redsearcher, v1

A v1 port of the v0 composable `search` taskset family. The taskset ships **no tools** — pair it with a harness that gives the model web-search tooling and a way to write `/task/answer.txt` (e.g. the `rlm` harness with the `websearch` / `open_webpage` skills; see the `rlm_search_v1` environment in research-environments).

### Datasets
- **Primary dataset(s)** (selected by `backend`):
  - `quest` → `osunlp/QUEST-RL-Data` (objective tasks graded by the dataset's generated `eval_scripts/{task_id}.py`; open-ended tasks graded by a pairwise rubric against `reward_model`).
  - `openseeker` → `PolarSeeker/OpenSeeker-v1-Data` (binary semantic LLM judge vs. gold answer).
  - `redsearcher` → `Zchu/REDSearcher_RL_1K` (normalized exact-match shortcut, then a BROWSECOMP-style LLM judge).
- **Split**: each backend defaults to `train`; override with `--taskset.split`.

### Task
- **Type**: Single-objective web research; the agent writes a final answer to `/task/answer.txt`.
- **Output format expectations**: Final answer text (with supporting URLs/citations) in `/task/answer.txt`; no scratch reasoning or tool traces in the file. If the file is absent, scoring falls back to the agent's last assistant message.
- **Rubric overview**: One reward in `[0, 1]` per backend — QUEST verification-tree / pairwise score, OpenSeeker `[CORRECT]`/`[INCORRECT]` judge, REDSearcher exact-match-or-judge. Judge runs against `judge_base_url` (default Prime inference) authenticated by `judge_api_key_var` (default `PRIME_API_KEY`).

### Quickstart
The taskset is harness-agnostic; run it under any harness that gives the model web search and a shell/file-write (the `rlm` harness is the intended pairing). A web-search backend in the runtime needs `SERPER_API_KEY`; scoring needs `PRIME_API_KEY` for the judge.

```bash
uv run eval search-v1 -a '{"backend": "openseeker"}'   -n 1 -r 1
uv run eval search-v1 -a '{"backend": "redsearcher", "difficulty": "easy"}' -n 1 -r 1
uv run eval search-v1 -a '{"backend": "quest", "category": "objective"}'    -n 1 -r 1
```

### Environment Arguments
Configured via `SearchConfig` (`--taskset.<field>` or `-a '{...}'`):

| Arg | Type | Default | Notes |
| --- | --- | --- | --- |
| `backend` | `quest \| openseeker \| redsearcher` | `quest` | Selects the taskset. |
| `dataset_name` / `split` | `str` | backend default | HF dataset override. |
| `answer_file` | `str` | `/task/answer.txt` | Where the agent writes its answer. |
| `judge_model` | `str` | `openai/gpt-5.4-mini` | LLM judge. |
| `judge_base_url` | `str` | Prime inference | `None` → OpenAI. |
| `judge_api_key_var` | `str` | `PRIME_API_KEY` | Env var holding the judge key. |
| `category` | `objective \| open-ended \| all` | `objective` | QUEST only. |
| `quest_eval_scripts_dir` | `str` | auto (HF download) | QUEST objective eval scripts. |
| `quest_eval_concurrency` | `int` | `8` | QUEST eval concurrency. |
| `difficulty` | `str` | — | REDSearcher: keep only matching-difficulty rows. |
| `redsearcher_judge_max_retries` | `int` | `5` | REDSearcher judge retries. |
| `redsearcher_exact_match_shortcut` | `bool` | `true` | REDSearcher exact-match shortcut. |
| `include_trajectory` | `bool` | `false` | OpenSeeker: include trajectory field. |

QUEST objective scoring additionally needs the PDF/eval deps declared in `pyproject.toml` (`pymupdf`, `pillow`, `aiohttp`); the generated eval scripts import the vendored `obj_task_eval` evaluator.
