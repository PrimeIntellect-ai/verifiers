# CLI Quick Reference

For the full CLI reference, see: https://docs.primeintellect.ai/cli-reference/introduction

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install prime
prime login
prime lab setup                    # initialize workspace
prime lab setup --prime-rl         # clone + install prime-rl trainer
prime lab setup --vf-rl            # install vf.RLTrainer configs
```

## Environment Management

```bash
prime env init my-env
prime env install my-env                     # local install
prime env install owner/env                  # from Hub
prime env install owner/env@1.0.0            # specific version
prime env push --path ./environments/my_env
prime env push --auto-bump                   # auto-increment version
prime env list --owner primeintellect
```

## Evaluation

```bash
prime eval run my-env -m openai/gpt-4.1-mini -n 10 -r 1
prime eval my-env -m openai/gpt-4.1-mini     # shorthand
prime eval run primeintellect/math-python     # from Hub (auto-installs)
prime eval run configs/eval/benchmark.toml    # multi-env eval from TOML
prime eval run my-env -a '{"difficulty": "hard"}'  # env args
prime eval run my-env -x '{"max_turns": 20}'       # extra env kwargs
prime eval tui                                # view results TUI
prime eval push                               # upload results
```

### Custom Endpoint

```bash
prime eval run my-env -m deepseek-reasoner \
  --api-base-url https://api.deepseek.com/v1 \
  --api-key-var DEEPSEEK_API_KEY
```

## Hosted Training

```bash
prime rl models                    # list available models
prime rl run configs/lab/my-env.toml
```

## Secrets

```bash
prime secrets           # manage global secrets
prime env secrets       # manage per-environment secrets
```

## Endpoint Shortcuts

Configure in `configs/endpoints.py`:

```python
ENDPOINTS = {
    "gpt-4.1-mini": {
        "model": "gpt-4.1-mini",
        "url": "https://api.openai.com/v1",
        "key": "OPENAI_API_KEY"
    },
}
```

## Common Errors

| Error | Solution |
|-------|----------|
| `ModuleNotFoundError: verifiers` | `uv add verifiers` or `prime env install my-env` |
| `load_environment` not found | Module name collision; rename env or check install |
| Environment not found | `prime env install owner/env@latest` |
| `MissingKeyError` | Set env vars per instructions; use `prime secrets` |
| Pydantic validation in config | Check TOML field types match schema |
