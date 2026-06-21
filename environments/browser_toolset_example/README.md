# browser-toolset-example

A minimal v1 environment demonstrating **`verifiers.v1.toolsets.browser`** — the
Claude computer-use action space driven over a raw Chrome DevTools Protocol
(CDP) browser, with a **pluggable backend**. A vision model controls a real
browser to complete short web tasks; an LLM judge (borrowing the rollout model)
scores the final answer.

### Overview
- **Environment ID**: `browser-toolset-example`
- **Tools**: the `browser` toolset — `computer` (Anthropic `computer_20250124`
  action enum) plus decomposed `navigate`, `left_click`, `type_text`, `key`,
  `scroll`, `screenshot`, … (selectable via `mode`).
- **Backends**: `browserbase` (managed, isolated session per rollout, default)
  or `cdp` (connect to any browser exposing a CDP endpoint).
- **Reward**: an LLM judge scores task success in `[0, 1]`.

> The model must be **vision-capable** — every browser action returns a
> screenshot, delivered as image content in the tool result.

### Quickstart

```bash
prime env install browser-toolset-example

# Browserbase (default): requires BROWSERBASE_API_KEY + BROWSERBASE_PROJECT_ID
BROWSERBASE_API_KEY=... BROWSERBASE_PROJECT_ID=... \
  prime eval run browser-toolset-example -m claude-sonnet-4-6 -n 3 -r 1

# Bring your own browser via any CDP endpoint, e.g. local Chrome:
#   chrome --headless=new --remote-debugging-port=9222
prime eval run browser-toolset-example -m claude-sonnet-4-6 \
  -a '{"backend": "cdp", "cdp_url": "http://localhost:9222"}'
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `backend` | `"browserbase"` \| `"cdp"` | `"browserbase"` | Browser infrastructure backend. |
| `cdp_url` | str \| null | `null` | CDP endpoint for `backend="cdp"` (`ws(s)://` socket or `http(s)://host:port`). |
| `proxies` | bool | `false` | Enable Browserbase proxies. |
| `mode` | `"computer"` \| `"decomposed"` \| `"both"` | `"both"` | Which tool surface to expose. |
| `viewport_width` / `viewport_height` | int | `1280` / `800` | Emulated viewport. |
| `max_turns` | int | `15` | Max agent turns per rollout. |
| `num_examples` | int | `-1` | Number of bundled tasks to use (`-1` = all). |

### Tasks

Three short, self-contained browsing tasks (read a heading on `example.com`,
compute a derivative on Wolfram Alpha, find the latest arXiv quantum-computing
preprint) — enough to exercise navigate / type / click / scroll / read.

### Using the toolset in your own environment

```python
import verifiers as vf
from verifiers.v1.toolsets.browser import browser_toolset, BrowserbaseBackend

harness = vf.Harness(config=vf.HarnessConfig(max_turns=20))
harness.add_toolset({"browser": browser_toolset(backend=BrowserbaseBackend())})
```
