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
    instruction: str = "Reverse the text character-by-character. ..."


class ReverseTextTaskset(vf.Taskset[ReverseTextTasksetConfig]):
    # The task instruction is prepended to each user prompt (no system prompt).
    def load_tasks(self) -> vf.Tasks: ...

    @vf.reward(weight=1.0)
    async def lcs_reward(self, task, state) -> float: ...


def load_taskset(config: ReverseTextTasksetConfig) -> ReverseTextTaskset:
    return ReverseTextTaskset(config=config)
```

### Quickstart

Loadable through `vf-eval-v1` (the v1 CLI) or `vf.load_environment`. The legacy `vf-eval` does not understand v1 modules that omit `load_environment`, and `vf-eval-v1` is v1-only: it rejects modules that don't expose `load_taskset`.

```bash
# default harness
vf-eval-v1 reverse-text --client.model openai/gpt-4.1-mini --num-examples 5

# swap to the rlm harness module (install `environments/v1/harnesses/rlm/` first)
vf-eval-v1 reverse-text rlm --harness.rlm-max-turns 25 --harness.system-prompt-merge harness

# override the taskset config
vf-eval-v1 reverse-text --taskset.dataset-split test
```

### Conflict with the legacy `reverse-text` package

This package and the legacy `environments/reverse_text/` package both publish under the env id `reverse-text` and both install a top-level Python module called `reverse_text`. They cannot be installed simultaneously: pick one.

Install this package (`environments/v1/reverse_text/`) when running through `vf-eval-v1`. Install the legacy `environments/reverse_text/` package when running through `vf-eval` (multi-env evaluations, ablation sweeps, short flags like `-n / -r / -m`).
