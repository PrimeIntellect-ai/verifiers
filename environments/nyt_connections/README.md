# nyt-connections

<a href="https://github.com/PrimeIntellect-ai/verifiers/tree/main/environments/nyt_connections">
  <img src="https://primeintellect.ai/prime.svg" alt="Prime Intellect" width="120" />
</a>

### Overview
- **Environment ID**: `nyt-connections`
- **Short description**: Multi-turn NYT Connections word puzzle environment.
- **Tags**: connections, nyt, word-game, multi-turn, reasoning, v1, taskset
- **Primary dataset**: [Eyefyre/NYT-Connections-Answers](https://github.com/Eyefyre/NYT-Connections-Answers)

### Task
The model receives 16 words and must find four hidden groups of four connected
words. It has four lives. Each assistant turn should contain a short rationale
and a guess in this format:

```xml
<think>These four words share a theme.</think>
<guess>WORD1, WORD2, WORD3, WORD4</guess>
```

Correct guesses reveal the group and remove those words from the board.
Incorrect or invalid guesses cost one life.

### Quickstart
```bash
prime env install environments/nyt_connections
prime eval run nyt-connections -n 1 -r 1
```

### Rubric
The environment combines:
- success reward for solving all four groups,
- progress reward for each solved group,
- efficiency reward for solving with lives remaining,
- format reward for valid `<think>` and `<guess>` tags.
