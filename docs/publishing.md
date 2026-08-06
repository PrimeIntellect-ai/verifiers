# Publishing an environment to the Environments Hub

This guide takes you from an empty directory to an environment installable by anyone from the
[Environments Hub](https://app.primeintellect.ai/dashboard/environments). It covers the **v0
authoring API** (`load_environment`), which is what the Hub installs and runs today, and ends
with a short note on the **v1 taskset** migration.

<Note>
Which API should I write? If your goal is to publish to the Hub now, use the **v0**
`load_environment` API below — that is what `prime env init`, `vf-eval`, and the Hub currently
expect. The **v1** taskset API (`verifiers.v1`) is the future of the framework and is covered
under [Building Tasksets](v1/tasksets.md); it is not yet the Hub's install format. See
[v0 vs v1](#v0-vs-v1) at the end.
</Note>

## 1. Prerequisites

```bash
# uv (package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Prime CLI, authenticated
uv tool install prime
prime config set-api-key <api-key>   # from your Prime Intellect account
```

## 2. Scaffold

```bash
prime env init my-environment
cd environments/my_environment
```

This creates:

```text
environments/my_environment/
├── my_environment.py     # implements load_environment()
├── pyproject.toml        # name, version, dependencies
├── README.md             # what the env tests, dataset provenance, results
└── outputs/              # sample eval outputs (committed so reviewers see behavior)
```

## 3. Implement `load_environment()`

An environment is a dataset of tasks plus a rubric that scores model responses. The minimal
single-turn shape:

```python
import verifiers as vf
from datasets import Dataset


def load_environment(**kwargs) -> vf.Environment:
    dataset = Dataset.from_list(
        [
            {"question": "What is 2 + 2?", "answer": "4"},
            {"question": "What is 3 * 5?", "answer": "15"},
        ]
    )

    async def correct_answer(completion, answer) -> float:
        response = completion[-1]["content"]
        return 1.0 if answer in response else 0.0

    rubric = vf.Rubric(funcs=[correct_answer])
    return vf.SingleTurnEnv(dataset=dataset, rubric=rubric)
```

Each dataset row becomes one **rollout**: the `question` (or a pre-built `prompt` message list)
is sent to the model, the model's reply becomes the `completion`, and each function in the
rubric scores it. Use `answer` for a ground-truth string and/or `info` (a dict, or a JSON
string when rows have differing schemas) for structured metadata.

### Accept configuration through `**kwargs`

Anything a user should be able to vary — split, difficulty, judge model, number of rows —
should be a keyword argument with a sensible default, so the environment is reusable:

```python
def load_environment(split: str = "test", max_examples: int = -1, **kwargs) -> vf.Environment:
    ...
```

### Multi-step and tool-using environments

- **Tools:** use `vf.ToolEnv` and expose tools as MCP servers when the task needs the model to
  call functions (search, code execution, an API sandbox).
- **Custom control flow:** subclass `vf.MultiTurnEnv` to own the rollout loop, stop conditions,
  and per-turn state.

See [Environments (v0)](v0/environments.md) for rubric composition (multiple reward functions,
weights, group-based and monitor rubrics), datasets, and the full multi-turn protocol.

## 4. Test locally

```bash
uv pip install -e .
uv run vf-eval my-environment                       # runs a few rollouts against a model
uv run vf-eval my-environment -n 5 -r 3             # 5 tasks, 3 rollouts each
```

`vf-eval` needs an OpenAI-compatible endpoint. Set `OPENAI_API_KEY` (and `OPENAI_BASE_URL` for
a non-OpenAI provider) before running. Inspect the sampled rollouts and reward distribution in
`outputs/` and confirm the reward separates good answers from bad — a reward that returns the
same value for every completion trains nothing.

<Warning>
Before publishing, sanity-check that your rubric can't be trivially gamed (e.g. `answer in
response` rewards a model that echoes every option). Reviewers look for this first.
</Warning>

## 5. Publish

```bash
prime env push
```

Bumping the `version` in `pyproject.toml` triggers CI to build and publish the environment to
the Hub under the `primeintellect` organization automatically — no manual release step.

## 6. What reviewers look for

- One environment per PR, under `environments/<name>/`.
- `README.md` states **what capability is tested**, the **dataset source + license**, and shows
  a **baseline result** (a known model's score) so the reward is calibrated.
- Committed `outputs/` from a real `vf-eval` run.
- A reward that is **discriminative** (varies with answer quality) and **not gameable**.
- Pinned dependencies in `pyproject.toml`; the env installs from a clean checkout.

## v0 vs v1

verifiers is mid-migration. The two APIs differ in shape, not just names:

| | **v0 (Hub today)** | **v1 (`verifiers.v1`, future)** |
|---|---|---|
| import | `import verifiers as vf` | `import verifiers.v1 as vf` |
| unit | `load_environment()` → `vf.Environment` | `vf.Taskset[Task, Config]` with `load()` |
| data | HF `Dataset` rows | typed `vf.TaskData` per task |
| scoring | `vf.Rubric(funcs=[...])` | `@vf.reward` methods on a `vf.Task` |
| scaffold | `prime env init <name>` | `uv run init <name>-v1` |
| run | `vf-eval <name>` | `uv run eval <taskset-id>` |

The v1 model makes scoring first-class (rewards are methods that see the full `Trace` and can
execute verification inside the rollout runtime) and separates configurable values onto
`TasksetConfig` / `TaskConfig`. If you're authoring for the Hub right now, stay on v0; if you're
contributing to the verifiers repo's `main`, use v1 and follow [Building Tasksets](v1/tasksets.md).
