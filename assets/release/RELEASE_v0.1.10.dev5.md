# Verifiers v0.1.10.dev5 Release Notes

*Date:* 02/10/2026

**Full Changelog**: https://github.com/PrimeIntellect-ai/verifiers/compare/v0.1.10.dev4...v0.1.10.dev5

## Highlights since v0.1.10.dev4

- Improved environment worker reliability and metadata tracking by migrating sandbox execution internals and recording verifiers/environment versions in rollout metadata.
- Polished evaluation UX with a fix for non-TUI live overflow rendering and broader logging improvements.
- Added and refined developer-facing guidance for skills and agent workflows, including clearer skill handling and browse-environment quality priorities.
- Extended opencode harbor support with TITO support, tunnel sync stop behavior, and a terminal-bench task addition.

## Incremental changes since v0.1.10.dev4

- env worker integration (#832)
- track vf + env version in metadata (#881)
- misc logging improvs (#882)
- rlm: migrate sandbox executor to SandboxMixin (#875)
- opencode env: TITO support, tunnel sync stop, add terminal-bench (#874)
- Fix vf-eval non-TUI live overflow rendering (#883)
- Fix for dir resolutions (#879)
- Update browse-environments freshness and quality priorities (#884)
- Clarify agent skill handling (#886)
