# tool-interception

Demonstrates immediate native tool interception in one deterministic rollout:

- a proposed command is blocked before execution;
- a real successful tool result is replaced before the agent sees it;
- a failed tool result is observed synchronously.

Run it with a harness that advertises native tool interception. The default Bash
harness works without installing another agent CLI:

```bash
PYTHONPATH=environments/tool_interception uv run eval tool-interception --no-push
```

The reward verifies that the blocked command never ran, both replacements reached
the trace, and the failed result was observed.
