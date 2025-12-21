# wordle

<a href="https://github.com/PrimeIntellect-ai/verifiers/tree/main/environments/wordle">
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="Source Code">
</a>

### Overview
- **Environment ID**: `wordle`
- **Short description**: Wordle game environment
- **Tags**: games, train, eval, multi-turn, wordle

### Datasets
- **Primary dataset(s)**: TextArena `Wordle-v0` (environment provides episodes)
- **Source links**: TextArena
- **Split sizes**: Number of episodes controlled via args

### Task
- **Type**: multi-turn (game interaction)
- **Parser**: `XMLParser` with `think`/`guess`
- **Rubric overview**: Exact guess match, partial credit from feedback, length bonus, and format check

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval wordle
```

Configure model and sampling:

```bash
uv run vf-eval wordle \
  -m gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"num_train_examples": 2000, "num_eval_examples": 20}'
```

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_train_examples` | int | `2000` | Number of training episodes |
| `num_eval_examples` | int | `20` | Number of evaluation episodes |

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `correct_answer` | 1.0 if final guess equals target, else 0.0 |
| `partial_answer` | Partial credit from greens/yellows in feedback |
| `length_bonus` | Higher score for solving in fewer turns |
| `format_reward` | Adherence to expected XML format |

### Changelog

#### v0.1.6 (Dec 10, 2025)

- Setup environment in `setup_state` instead of `env_response`
- Fix checking game completion condition
- Fix feedback parsing to correctly handle feedback on game completion and invalid guesses
- Add logger
- Rename reward functions
