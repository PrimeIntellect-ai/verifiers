# internbootcamp-v1

### Overview

- **Environment ID**: `internbootcamp-v1`
- **Description**: seeded Verifiers v1 adapter for the procedural reasoning tasks in InternBootcamp.
- **Tags**: reasoning, procedural, single-turn, train, eval

The environment discovers Bootcamp classes from a pinned Apache-2.0 snapshot of
[`dmihal/InternBootcamp`](https://github.com/dmihal/InternBootcamp/tree/2b2d388f4f056cd9bd0cc91b130f0b54b15572b4),
uses the selected class's native `case_generator` and `prompt_func`, and delegates
reward calculation to its native `verify_score` implementation.

### Security model

InternBootcamp contains task-specific verifiers that may evaluate model-produced
expressions. For that reason, `InternBootcampTask` declares `NEEDS_CONTAINER = True`
and executes the pinned scorer inside the rollout runtime. The scorer never receives
model output in the environment host process. Use a Docker or Prime runtime; the
subprocess runtime is intentionally rejected.

### Quickstart

Validate the default Game24 configuration:

```bash
uv run validate internbootcamp-v1 --runtime.type docker -n 1
```

Run another Bootcamp by class name or canonical key:

```bash
uv run eval internbootcamp-v1 \
  --harness.runtime.type docker -n 20 \
  --taskset.bootcamp Sudoku --taskset.num-examples 50 --taskset.seed 123
```

### Environment arguments

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `bootcamp` | string | `"game24"` | Bootcamp class name or normalized canonical key. |
| `num_examples` | integer | `50` | Number of procedural cases generated before run-time selection. |
| `seed` | integer | `0` | Seed for Python, NumPy, Faker, and constructors that accept `seed`. |
| `system_prompt` | string | `"Think step by step to solve the puzzle."` | System instruction for every task. |
| `task.verifier_timeout` | integer | `180` | Maximum scoring-call time in seconds. |

Only Bootcamps with a `case_generator`, a default constructor, and JSON-serializable
generated identities are supported. Unsupported classes fail during taskset loading
with a descriptive error instead of silently changing their data.

### Task and scoring

Each case is single-turn. Output formatting depends on the selected Bootcamp and is
described in its generated prompt. The only reward, `upstream_score`, is the selected
Bootcamp's native scalar score clamped to `[0, 1]`. Empty completions are scored during
validation to preflight the pinned scorer without a model call.

### Reproducibility and provenance

Generation and scoring use the exact commit
[`2b2d388`](https://github.com/dmihal/InternBootcamp/commit/2b2d388f4f056cd9bd0cc91b130f0b54b15572b4).
The environment seeds common randomness sources and passes `seed` to Bootcamp
constructors that expose it. The original project and task data remain subject to
their upstream license and dataset terms.
