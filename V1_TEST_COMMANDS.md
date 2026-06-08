# V1 Test Commands

Run these from the repository root.

## Default V1 Evals

Each command uses the environment default settings. For the current example
packages, that means `n=5` and `r=3`. `--disable-tui` and
`--abbreviated-summary` only change display.

Run these in separate terminals when you want broad coverage:

```bash
prime eval run reverse-text-v1 --disable-tui --abbreviated-summary
```

```bash
prime eval run alphabet-sort-v1 --disable-tui --abbreviated-summary
```

```bash
prime eval run mcp-search-env-v1 --disable-tui --abbreviated-summary
```

```bash
prime eval run math-python-v1 --disable-tui --abbreviated-summary
```

```bash
prime eval run hello-group-reward-v1 --disable-tui --abbreviated-summary
```

```bash
prime eval run sft-replay-v1 --disable-tui --abbreviated-summary
```

Stateful user/tool environments are slower and noisier. Run these separately
from the quick eval batch:

```bash
prime eval run openenv-echo-v1 --disable-tui --abbreviated-summary
```

```bash
prime eval run openenv-textarena-v1 --disable-tui --abbreviated-summary
```

```bash
prime eval run tau2-bench-v1 --disable-tui --abbreviated-summary
```
