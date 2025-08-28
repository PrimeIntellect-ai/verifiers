# wordle

### Overview
- **Environment ID**: `wordle`
- **Short description**: Multi-turn Wordle game environment with optional `<think>` reasoning; rewards correctness, partial credit, turns, and format.
- **Tags**: games, multi-turn, wordle, xml, feedback

### Datasets
- **Primary dataset(s)**: TextArena `Wordle-v0` (environment provides episodes)
- **Source links**: TextArena
- **Split sizes**: Number of episodes controlled via args

### Task
- **Type**: multi-turn (game interaction)
- **Parser**: `XMLParser` with `think`/`guess` or just `guess` depending on `use_think`
- **Rubric overview**: Exact guess match, partial credit from feedback, turns-based reward, and format check

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
  -a '{"num_train_examples": 2000, "num_eval_examples": 20, "use_think": true}'
```

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_train_examples` | int | `2000` | Number of training episodes |
| `num_eval_examples` | int | `20` | Number of evaluation episodes |
| `use_think` | bool | `true` | Use `<think>` with `guess`; if false, guess-only format |

### Metrics
| Metric | Meaning |
| ------ | ------- |
| `check_answer_reward_func` | 1.0 if final guess equals target, else 0.0 |
| `partial_credit_reward_func` | Partial credit from greens/yellows in feedback |
| `count_turns_reward_func` | Higher score for solving in fewer turns |
| `format_reward` | Adherence to expected XML format |
