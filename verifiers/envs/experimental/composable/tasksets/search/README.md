# Search Tasksets

Composable search/research tasksets for agents that solve live information-seeking tasks in a sandbox.

The search family is intentionally backend-oriented, mirroring the SWE taskset pattern while keeping the task contract research-centric: each task expects a single final answer rather than a code patch. Agents may use web/search tools, browser helpers, or other sandbox resources provided by the paired environment.

## Backends

| Backend | Source | Default dataset | Status |
|---|---|---|---|
| `quest` | [OSU-NLP-Group/QUEST](https://github.com/OSU-NLP-Group/QUEST) | [`osunlp/QUEST-RL-Data`](https://huggingface.co/datasets/osunlp/QUEST-RL-Data) | Objective tasks supported |

## Usage

```python
from verifiers.envs.experimental.composable.tasksets.search import make_search_taskset

taskset = make_search_taskset(backend="quest", category="objective")
```

`make_search_taskset()` dispatches by backend name. Unknown backends raise `ValueError` with the available backend list.

## Output Contract

Search tasksets should define their own output contract. The initial `quest` backend expects the agent to write one final researched response to `/task/answer.txt`, including supporting URLs/citations when available. Scratch reasoning, tool traces, and logs should not be written as the final answer.

## Error Handling

Search tasksets should use the framework error taxonomy for infrastructure failures:

- `vf.SandboxError` for sandbox setup, command, or lifecycle failures.
- `vf.ModelError` for judge/model provider failures.
- `vf.InfraError` for dataset, evaluator, or external runtime failures.

Incorrect answers should not set `state["error"]`; they should score normally, often as `0.0`.
