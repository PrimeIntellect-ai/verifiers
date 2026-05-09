# bfcl-v3

Berkeley Function Calling Leaderboard v3 on the v1 Taskset/Harness runtime.

```bash
prime env install bfcl-v3 --from-repo
prime eval run bfcl-v3 -a '{"test_categories": ["simple_python"]}'
```

Single-turn categories use task-local schema-backed toolsets. Multi-turn
categories use a taskset-owned custom harness program for BFCL's official tool
execution loop.
