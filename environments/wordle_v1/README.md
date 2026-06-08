# wordle-v1

### Overview
- **Environment ID**: `wordle-v1`
- **Short description**: Wordle game environment built on the v1 TextArena taskset.
- **Tags**: textarena, multi-turn, reasoning, game, v1

### Task
- **Type**: multi-turn game interaction
- **Rubric overview**: Scores exact answer, partial Wordle feedback, shorter successful games, and `<guess>...</guess>` formatting.

### Quickstart
```bash
prime eval run wordle-v1
```

### Configuration
The environment uses the packaged `TextArenaTaskset` for generic TextArena mechanics.
`wordle_v1/taskset.py` owns the Wordle prompt, `WordleUser` response shaping, rewards,
and defaults for `Wordle-v0`.
