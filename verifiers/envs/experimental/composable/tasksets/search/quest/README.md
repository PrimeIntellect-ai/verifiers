# QUEST Search Taskset

Objective QUEST tasks ported into the composable search taskset framework.

## Source

- Dataset: [`osunlp/QUEST-RL-Data`](https://huggingface.co/datasets/osunlp/QUEST-RL-Data)
- Upstream project: [`OSU-NLP-Group/QUEST`](https://github.com/OSU-NLP-Group/QUEST)

The taskset loads the Hugging Face dataset, filters to `rl_task_category == "objective"` by default, and uses the dataset-provided generated evaluation scripts under `eval_scripts/*.py`.

## Task Contract

Each example is a live research question. The agent should produce one final answer in `/task/answer.txt`.

The paired `rlm_search` environment prompts RLM to write this file and provides web search/open-page skills. The rubric can fall back to the final assistant text if the answer file is empty, but agents should still write the file directly.

## Scoring

`QuestRubric` loads the generated eval script for the example's `task_id` and calls its async `evaluate_answer(...)` entrypoint using the vendored minimal `obj_task_eval` runtime. The rollout reward is `summary["final_score"]`, clipped to `[0.0, 1.0]`.

Generated scripts may request URL-backed verification. PDF URLs are detected and parsed with the upstream QUEST PDF parser path before falling back to generic webpage retrieval.

This port intentionally preserves upstream QUEST behavior for URL-backed verification semantics. The upstream verifier generally treats invalid, irrelevant, or inaccessible cited webpages as unsupported claims, which can assign `0.0` to the affected verification node even when the immediate cause is source access such as a bot challenge, rate limit, timeout, or parser failure. Future work should consider a finer-grained source-access taxonomy so verifier infrastructure limitations can be distinguished from model-provided bad URLs or unsupported claims.

A reward of `0.0` with no `state["error"]` means the QUEST evaluator ran and judged the answer incorrect under the upstream-compatible scoring path. Infrastructure and evaluator failures outside normal QUEST source verification are represented with `vf.Error` subclasses instead of ad hoc success metrics.

## Error Handling

QUEST uses Verifiers' framework-managed error field for non-answer failures:

- Missing live sandbox or answer-file read failure: `vf.SandboxError`.
- Missing judge API key or OpenAI-compatible judge request failure: `vf.ModelError`.
- Missing task metadata, eval script download failure, eval script load failure, or generated evaluator crash: `vf.InfraError`.

Wrong or empty answers remain ordinary scored outcomes and return `0.0` without setting `state["error"]`.

## Common Arguments

| Argument | Default | Description |
|---|---:|---|
| `dataset_name` | `osunlp/QUEST-RL-Data` | Hugging Face dataset name. |
| `split` | `train` | Dataset split. |
| `category` | `objective` | Initial implementation supports objective tasks only. |
| `answer_file` | `/task/answer.txt` | Final answer path in the sandbox. |
| `judge_model` | `openai/o4-mini` | OpenAI-compatible model for QUEST verifier calls. |
| `judge_base_url` | `https://api.pinference.ai/api/v1` | Judge API base URL. |
| `judge_api_key_var` | `PRIME_API_KEY` | Env var containing the judge API key. |
| `quest_eval_scripts_dir` | HF cache | Optional local directory containing `eval_scripts/*.py`. |
| `quest_cache_dir` | `~/.cache/verifiers/quest` | Host cache for QUEST verifier state. |
