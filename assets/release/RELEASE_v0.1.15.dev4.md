# Verifiers v0.1.15.dev4 Release Notes

*Date:* 05/14/2026

## Highlights since v0.1.15.dev3

- **v1 config and binding surfaces are stricter and more explicit.** v1 tasksets, harnesses, users, toolsets, env args, prompts, tools, and runtime handles now use named Pydantic-backed types and protocols instead of loose object-shaped config. Direct object binding is rejected so private runtime objects stay behind object loaders and explicit binding boundaries.
- **v1 rollout, lifecycle, and taskset utilities are consolidated.** Shared config, binding, lifecycle, runtime registry, serialization, task freezing, taskset, usage, sandbox, and program plumbing now live in dedicated utility modules. Agent text extraction and rollout extraction stay explicit, and empty completions are handled cleanly in v1 reward paths.
- **Environment examples and docs now follow the current v1 contract.** BFCL, RLM, subagent, Harbor/OpenCode, OpenEnv, wiki-search, tau2, and related examples have been updated for the current config and message-access patterns. OpenEnv remains optional at import time, OpenCode Harbor defaults are documented, and reusable v1 authoring guidance now lives in the BYO Harness docs.
- **Eval runs can show estimated Prime cost.** `prime eval run` fetches Prime Inference pricing, records `metadata.cost` when usage and pricing are available, and renders cost in live eval displays, final summaries, saved-results TUI headers, and non-TUI usage output.
- **Repo and agent guidance reflects the current contributor workflow.** AGENTS, docs, and environment skills now describe the canonical `prime eval run` path, public config expectations, taskset/harness ownership, downstream-consumer checks, and focused validation defaults.

## Changes included in v0.1.15.dev4 (since v0.1.15.dev3)

### v1 environment model

- Tighten v1 config, bindings, and handler typing (#1362)

### Evaluation UX

- [codex] Display eval cost from Prime pricing (#1368)

### Docs, guidance, and maintenance

- Add repo development best practices to AGENTS, docs, and skills (#1361)

**Full Changelog**: https://github.com/PrimeIntellect-ai/verifiers/compare/v0.1.15.dev3...v0.1.15.dev4
