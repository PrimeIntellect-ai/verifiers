# rlm-harness (v1)

### Overview

- **Harness name**: `rlm`
- **Module installed**: `rlm`
- **Short description**: Installable v1 harness module wrapping `verifiers.v1.packages.harnesses:RLM`.

### Contract

The module exposes a single factory:

```python
def load_harness(config: RLMConfig) -> RLM: ...
```

That's the contract the v1 CLI looks for. Any other harness package that wants to be referenceable by name from `vf-eval-v1 <taskset> <harness>` follows the same pattern: a top-level module with a `load_harness(config: HarnessConfig) -> Harness` factory.

### Usage

Install side-by-side with any v1 taskset:

```bash
uv pip install -e environments/v1/harnesses/rlm
uv pip install -e environments/v1/reverse_text

vf-eval-v1 reverse-text rlm \
    --harness.rlm-max-turns 5 \
    --harness.system-prompt-merge harness \
    --timeout 240
```

The CLI imports `rlm`, reads `RLMConfig` from `load_harness`'s signature, validates `--harness.<field>` against it, then calls `load_harness(config=...)`.
