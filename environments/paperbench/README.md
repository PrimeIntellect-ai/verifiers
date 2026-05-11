# paperbench

V1 taskset/harness environment for PaperBench paper-reproduction submissions.

```python
from paperbench_env import load_environment

env = load_environment()
```

Each task corresponds to a PaperBench paper ID. The prompt mirrors PaperBench's
submission contract: the agent receives paper materials under `/home/paper` and
must create a git repository at `/home/submission`.

The default `code_only=True` mode matches PaperBench Code-Dev evaluation:

- `/home/submission` exists
- `/home/submission/.git` exists
- `/home/submission/README.md` exists

When `code_only=False`, the environment additionally requires
`/home/submission/reproduce.sh`, matching the full PaperBench reproduction
contract. The reward is local and structural so the environment can be imported
and unit-tested without OpenAI judge keys, GPUs, or PaperBench data. Downstream
runners can use the preserved `paper_id`, split, submission path, and code-only
flag to call the official PaperBench reproduction and judge stack.
