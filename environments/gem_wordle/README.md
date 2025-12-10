# gem_wordle

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
- **Parser**: Identity (GEM environment internally parses `\boxed{GUESS}` using Regex)
- **Rubric overview**: Accumulated dense rewards from environment steps (correct letters/positions) + terminal success bonus.

### Quickstart
Run an evaluation with default settings:

```bash
uv run vf-eval gem_wordle