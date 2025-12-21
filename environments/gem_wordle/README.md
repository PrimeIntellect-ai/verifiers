# gem_wordle

<a href="https://github.com/PrimeIntellect-ai/verifiers/tree/main/environments/gem_wordle">
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="Source Code">
</a>

### Overview
- **Environment ID**: `gem_wordle`
- **Short description**: Multi-turn Wordle game environment powered by the GEM framework. Models must guess a 5-letter word using `\boxed{}` format actions.
- **Tags**: games, multi-turn, wordle, gem, regex, feedback

### Datasets
- **Primary dataset(s)**: GEM `game:Wordle-v0` (environment auto-generates episodes)
- **Source links**: [AxonRL GEM](https://github.com/axon-rl/gem)
- **Split sizes**: Number of episodes controlled via args (auto-generated dummy dataset)

### Task
- **Type**: multi-turn (gym environment interaction)
- **Parser**: Identity (GEM environment parses `\boxed{GUESS}` internally via regex; the env passes raw model text through so format/validity penalties remain part of the training signal)
- **Rubric overview**: Sum of per-step rewards returned by GEM (includes shaping + terminal success reward, plus small negative penalties for format/invalid actions; commonly `-0.1`)

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval gem_wordle
```

Configure model and sampling (recommend higher `-t` so the model reliably emits the closing `}`):

```bash
export OPENAI_API_KEY=EMPTY
uv run vf-eval gem_wordle \
  -b http://127.0.0.1:8000/v1 -k OPENAI_API_KEY \
  -m Qwen/Qwen3-30B-A3B-Instruct-2507 \
  -n 20 -r 3 -t 1024 \
  -a '{"num_train_episodes": 1000, "num_eval_episodes": 20}' \
  -s
```

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_train_episodes` | int | `1000` | Number of training episodes (auto-generated) |
| `num_eval_episodes` | int | `20` | Number of evaluation episodes (auto-generated) |

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `sum_step_rewards` | Sum of GEM per-step rewards (training reward) |
| `win_rate` | 1.0 if episode ends with “Congratulations!”, else 0.0 |
