# Search Judge Tasksets

Composable search/research tasksets whose scoring is driven by task-specific judge or verifier logic rather than a simple reference-answer comparison.

## Backends

| Backend | Source | Default dataset | Status |
|---|---|---|---|
| `quest` | [OSU-NLP-Group/QUEST](https://github.com/OSU-NLP-Group/QUEST) | [`osunlp/QUEST-RL-Data`](https://huggingface.co/datasets/osunlp/QUEST-RL-Data) | Objective and open-ended judge tasks supported |

## Usage

```python
from verifiers.envs.experimental.composable.tasksets.search_judge import make_search_judge_taskset

taskset = make_search_judge_taskset(backend="quest", category="objective")
taskset = make_search_judge_taskset(backend="quest", category="open-ended")
```

`make_search_judge_taskset()` dispatches by backend name. Unknown backends raise `ValueError` with the available backend list.
