Mastermind (Multi-Turn)

Mastermind is a classic deductive reasoning game, first analyzed algorithmically by Donald Knuth, who showed that the standard 4×6 version can always be solved in at most five guesses using a minimax search strategy. For slightly larger boards, exact worst-case bounds are known only in a few cases, and the general problem is NP-hard to solve optimally. As the number of positions or colors increases, the combinatorial search space grows exponentially, making exhaustive reasoning infeasible. Larger puzzles therefore require adaptive, information-efficient strategies that can infer structure and make effective guesses under uncertainty.

The model plays the codebreaker and receives peg feedback after each guess until it either solves the code or runs out of attempts.

- Code length: configurable (default 4)
- Symbols: digits 0..S-1 (default 0..5)
- Duplicates: allowed by default
- Max turns: Can be set explicitly, but is calculated by default from an information-theoretic bound with additive slack based on code length.
  - Calculation supports the following parameters `slack_factor` (default 0.3) and `min_slack` (default 2).

The environment enforces strict output formatting with XML tags and provides reward shaping signals beyond success.

Note: by default, this environment rewards the model based on reduction to the candidate search space, but this calculation scales combinatorially and can become prohibitively slow for more complex puzzles. You can disable it with `use_candidate_reduction_reward=false`.

Install & Run

```bash
# from repo root
vf-install mastermind
vf-eval mastermind -m gpt-4.1-mini -n 5
```

Pass parameters via kwargs:

```bash
vf-eval mastermind -m gpt-4.1-mini -n 10 \
  --kwargs '{"code_length":4, "num_symbols":6, "allow_duplicates":true, "use_think":true, "use_candidate_reduction_reward":true, "slack_factor":0.3, "min_slack":2}'
```

Reward Shaping

- solved_reward (1.0 if solved; else 0)
- speed_reward (1/turns if solved)
- partial_feedback_reward (normalized from final turn’s B/W)
- candidate_reduction_reward (normalized log shrink of consistent code space; small weight). Toggle via `use_candidate_reduction_reward` (default true). When disabled, the environment skips candidate counting entirely.
- format_reward (parser-driven format compliance)

You can override weights via rubric_weights in load_environment kwargs by metric name.

Notes

- For the standard 4 pegs, 6 symbols (duplicates allowed), Knuth’s algorithm guarantees a solution in at most 5 guesses. For other settings, optimal worst-case bounds are not always known; the shaping terms are designed to encourage informative, narrowing guesses even when optimality is unknown.

Files

- mastermind.py — environment implementation
- pyproject.toml — install metadata
