# SynLogic

SynLogic is a collection of synthetic reasoning and puzzle tasks from MiniMax. This environment wraps the Hugging Face dataset [`MiniMaxAI/SynLogic`](https://huggingface.co/datasets/MiniMaxAI/SynLogic) and the upstream verifier logic from [`MiniMax-AI/SynLogic`](https://github.com/MiniMax-AI/SynLogic) as a Verifiers `SingleTurnEnv`.

Algora bounty: https://algora.io/PrimeIntellect-ai/bounties/g42D7Yrh5u25gjtb

## Task

Each example asks the model to solve one logic puzzle. The model should respond with:

```xml
<think>reasoning process here</think><answer>final answer here</answer>
```

The hidden `extra_info.game_data_str` payload from the dataset is used only by the reward functions, not shown separately to the model beyond the original prompt.

## Rewards

- `synlogic_reward` (weight `1.0`): requires the exact SynLogic response format and then runs the upstream verifier for the task type.
- `format_reward` (weight `0.0`): tracks whether the response has exactly one `<think>...</think>` and one `<answer>...</answer>` block.
- `accuracy_reward` (weight `0.0`): tracks verifier correctness independent of the format gate.

Validation dataset sources such as `val/campsite` are normalized to the upstream verifier key `campsite`.

## Usage

```python
import verifiers as vf

env = vf.load_environment("synlogic", config="easy", max_examples=512, max_eval_examples=128)
```

Config options:

- `config="easy"` or `config="hard"`
- `max_examples=None` to load all training examples
- `max_eval_examples=None` to load all validation examples
- `shuffle=True`, `seed=0`

## Local checks

```bash
uvx ruff check environments/synlogic
uv run --no-dev vf-install synlogic
uv run --no-dev python - <<'PY'
import verifiers as vf
env = vf.load_environment('synlogic', max_examples=1, max_eval_examples=1)
print(type(env).__name__)
print(len(env.get_dataset()), len(env.get_eval_dataset()))
PY
CHANGED_ENVS=synlogic uv run --no-dev --with pytest pytest tests/test_envs.py -q -k 'pyproject or readme'
```

The upstream SynLogic verifier code is vendored under `synlogic_src/` because the reference repository does not expose an installable Python package.
