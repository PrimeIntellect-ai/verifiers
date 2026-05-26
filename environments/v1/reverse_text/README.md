# reverse-text (v1)

### Overview

- **Environment ID**: `reverse-text`
- **Short description**: Reverse the input text character-by-character; score is the LCS similarity between the parsed answer and the ground-truth reversal.
- **Tags**: v1, taskset, single-turn, text, xml

### Layout

This is the **pure v1 shape**: the whole env is a `Taskset` plus a `load_taskset(config)` factory. There is no `EnvConfig` subclass, no `load_environment`, and no `load_harness`. The default harness auto-resolves to the base `verifiers.v1.Harness`.

```python
class ReverseTextTasksetConfig(vf.TasksetConfig):
    dataset_name: str = "PrimeIntellect/Reverse-Text-RL"
    dataset_split: str = "train"


class ReverseTextTaskset(vf.Taskset[ReverseTextTasksetConfig]):
    def load_tasks(self) -> vf.Tasks: ...
    def load_system_prompt(self) -> vf.SystemPrompt: ...

    @vf.reward(weight=1.0)
    async def lcs_reward(self, task, state) -> float: ...


def load_taskset(config: ReverseTextTasksetConfig) -> ReverseTextTaskset:
    return ReverseTextTaskset(config=config)
```

### Quickstart

Only loadable through the v1 CLI (`vf-eval`). The legacy `vf-eval-legacy` expects a `load_environment` function and will fail on this env.

```bash
# default harness
vf-eval reverse-text -m openai/gpt-4.1-mini --num-examples 5

# swap to RLM via the second positional
vf-eval reverse-text rlm --harness.rlm-max-turns 25 --harness.system-prompt-merge harness

# override the taskset config
vf-eval reverse-text --taskset.dataset-split test
```

### Conflict with the legacy `reverse-text` package

This package and the legacy `environments/reverse_text/` package both publish under the env id `reverse-text` and both install a top-level Python module called `reverse_text`. They cannot be installed simultaneously: pick one.

Use this package (`environments/v1/reverse_text/`) when running through `vf-eval`. Use the legacy `environments/reverse_text/` package when running through `vf-eval-legacy` (multi-env evaluations, ablation sweeps, short flags like `-n / -r / -m`).
