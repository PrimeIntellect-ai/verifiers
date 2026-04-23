# harnesses

Optional agent harness package for `verifiers`.

Install:

```bash
uv add harnesses
```

This package provides:

- `harnesses.Harness` for agent-side install/run configuration
- `harnesses.make_native_harness` for native CLI harnesses that rebuild their
  run command when tasksets provide MCP tools or skills
- `harnesses.build_harness_from_config` for TOML-friendly harness lookup
- `harnesses.opencode` (OpenCode `Harness` factory)
- `harnesses.codex` (OpenAI Codex CLI `Harness` factory)
- `harnesses.claude_code` (Claude Code `Harness` factory)
- `harnesses.openclaw` (OpenClaw `Harness` factory)
- `harnesses.mini_swe_agent` (mini-SWE-agent `Harness` factory)
- `harnesses.pi_mono` (pi coding agent `Harness` factory)
- `harnesses.terminus_2` (Terminus 2 `Harness` factory)

See [TOOL_REGISTRATION.md](TOOL_REGISTRATION.md) for the task-provided tool
registration contract used by composable tasksets and harnesses.

The Codex harness installs the upstream `@openai/codex` CLI, runs
`codex exec` non-interactively against the intercepted OpenAI-compatible
endpoint, and supports Codex MCP server config plus Agent Skills discovery.

The OpenCode harness installs a pinned OpenCode CLI release, writes
`opencode.json` for the intercepted OpenAI-compatible endpoint, and supports
OpenCode MCP server config.

The Claude Code harness installs the upstream `@anthropic-ai/claude-code` CLI,
runs `claude --print` with streaming JSON logs, and supports MCP server config,
Agent Skills discovery, and pre-seeded memory files.

The OpenClaw harness installs the upstream `openclaw` CLI, writes a
rollout-local `openclaw.json` for the intercepted OpenAI-compatible endpoint,
runs `openclaw agent --local` with JSON output, and supports OpenClaw MCP
server config plus workspace Agent Skills discovery.

The mini-SWE-agent harness installs the upstream `mini-swe-agent` Python
package in a sandbox venv and runs the `mini` CLI against the intercepted
OpenAI-compatible endpoint with a small direct OpenAI SDK model adapter.

The pi-mono harness installs the upstream `@mariozechner/pi-coding-agent` CLI,
writes a rollout-local `models.json` for the intercepted OpenAI-compatible
endpoint, and runs `pi` non-interactively with JSON event logs and optional
Agent Skills discovery.

The Terminus 2 harness installs a dependency-light tmux runner with JSON/XML
command parsing, optional MCP server and Agent Skills discovery context, and
three-step summarization handoffs for long histories.

Each harness exports a `<HARNESS>_CONFIG` constant with the generated install
script, pinned CLI/source version, and SHA-256 hash used by the install-time
integrity check.

`verifiers` core remains usable without this package.
