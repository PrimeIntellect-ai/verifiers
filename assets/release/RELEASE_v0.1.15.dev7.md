# Verifiers v0.1.15.dev7 Release Notes

*Date:* 05/15/2026

## Highlights since v0.1.15.dev6

- **PyPI release pipeline moved to Trusted Publishing.** The `Tag and Release` workflow now publishes via PyPA OIDC, gated on the `pypi-prod` GitHub environment, with build, publish, and GitHub-release split into separate jobs so build-time code never sees the publishing identity. The long-lived `PYPI_TOKEN` is no longer used to release `verifiers`.
- **Per-eval naming for duplicate environment runs.** `prime eval run` now accepts per-eval name labels so duplicate environments produce distinct, disambiguated runs.

## Changes included in v0.1.15.dev7 (since v0.1.15.dev6)

### Release infrastructure

- improvement: switch verifiers PyPI publish to OIDC trusted publishing (#1386)

### Evals

- Add per-eval names for duplicate environment runs (#1384)
- Document eval name labels in skill (#1388)

### Docs

- Add AGENTS.local guidance to Lab workspace docs (#1383)

**Full Changelog**: https://github.com/PrimeIntellect-ai/verifiers/compare/v0.1.15.dev6...v0.1.15.dev7
