# bfcl-v3-v1

Berkeley Function Calling Leaderboard v3 on the v1 Taskset/Harness runtime.

```bash
prime eval run bfcl-v3-v1 -a '{"config": {"taskset": {"test_category": "simple_python"}}}'
```

Single-turn categories provision schema-backed tools into a taskset-owned
rollout toolset. Multi-turn categories use a taskset-owned custom harness
program for BFCL's official tool execution loop.

Configure one category per v1 eval. Use multiple eval entries to run multiple
BFCL categories.
