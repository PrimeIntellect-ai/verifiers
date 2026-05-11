# Kimi-K2 Tool Sim

Deterministic single-turn tool-use simulation tasks inspired by Kimi-K2 style
agentic function-calling evaluations. The model does not execute tools. It must
infer the exact tool call sequence and arguments from a compact task description
and return the result as JSON.

## Usage

```python
import verifiers as vf

env = vf.load_environment("kimi-k2-tool-sim")
```

Expected completion format:

```json
{
  "tool_calls": [
    {"name": "orders.find", "arguments": {"email": "sam@example.com"}}
  ],
  "answer": "Created a refund for order ORD-8842."
}
```

## Scoring

The reward is provider-free and deterministic:

- 40% exact tool-name sequence match
- 40% argument-value matching across expected tool calls
- 20% final answer match

Malformed JSON, missing tool calls, or an incorrect number of calls receives
zero reward.
