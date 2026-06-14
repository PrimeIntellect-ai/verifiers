# Search Tasksets

Composable search/research tasksets for agents that solve live information-seeking tasks in a sandbox.

The search family is intentionally backend-oriented, mirroring the SWE taskset pattern while keeping the task contract research-centric: each task expects a single final answer rather than a code patch. Agents may use web/search tools, browser helpers, or other sandbox resources provided by the paired environment.

## Backends

| Backend | Source | Default dataset | Status |
|---|---|---|---|
| `openseeker` | [PolarSeeker/OpenSeeker](https://github.com/PolarSeeker/OpenSeeker) | [`PolarSeeker/OpenSeeker-v1-Data`](https://huggingface.co/datasets/PolarSeeker/OpenSeeker-v1-Data) | Binary semantic answer judge |
| `quest` | [OSU-NLP-Group/QUEST](https://github.com/OSU-NLP-Group/QUEST) | [`osunlp/QUEST-RL-Data`](https://huggingface.co/datasets/osunlp/QUEST-RL-Data) | Objective tasks supported |
| `redsearcher` | [RedSearchAgent/REDSearcher](https://github.com/RedSearchAgent/REDSearcher) | [`Zchu/REDSearcher_RL_1K`](https://huggingface.co/datasets/Zchu/REDSearcher_RL_1K) | Text RL query set supported |

## Usage

```python
from verifiers.envs.experimental.composable.tasksets.search import make_search_taskset

taskset = make_search_taskset(backend="openseeker")
taskset = make_search_taskset(backend="quest", category="objective")
redsearcher = make_search_taskset(
    backend="redsearcher",
    filter_fn="lambda x: x['info']['difficulty'] == 'easy'",
)
```

`make_search_taskset()` dispatches by backend name. Unknown backends raise `ValueError` with the available backend list.

## Output Contract

Search tasksets should define their own output contract. The `quest`, `openseeker`, and `redsearcher` backends expect the agent to write one final researched response to `/task/answer.txt`, including supporting URLs/citations when available. Scratch reasoning, tool traces, and logs should not be written as the final answer.

## Error Handling

Search tasksets should use the framework error taxonomy for infrastructure failures:

- `vf.SandboxError` for sandbox setup, command, or lifecycle failures.
- `vf.ModelError` for judge/model provider failures.
- `vf.InfraError` for dataset, evaluator, or external runtime failures.

Incorrect answers should not set `state["error"]`; they should score normally, often as `0.0`.
