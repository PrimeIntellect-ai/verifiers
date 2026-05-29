# lisanbench

### Overview
- **Environment ID**: `lisanbench`
- **Short description**: LisanBench-style open-ended word-chain task for planning, vocabulary depth, and constraint adherence.
- **Tags**: word-chain, lisanbench, levenshtein, planning, vocabulary, single-turn, train, eval

### Task
The model receives a starting word and must output the longest possible comma-separated chain of English words. Each adjacent pair must have Levenshtein edit distance exactly 1, all words must be valid English words, and no word may repeat.

### Quickstart
```bash
prime env install environments/lisanbench
prime eval run lisanbench -n 1 -r 1
```

### Rubric
The reward combines:

- correct starting word,
- valid edit-distance-1 transitions,
- length of the valid prefix,
- no duplicates,
- list-only formatting.

The dictionary is cached from `dwyl/english-words` on first use, matching the public LisanBench implementation. A small fallback dictionary is included for offline smoke tests.

### References
- Algora bounty: https://algora.io/PrimeIntellect-ai/bounties/dDffD24XfkQUaR7a
- Source benchmark: https://github.com/voice-from-the-outer-world/lisan-bench
