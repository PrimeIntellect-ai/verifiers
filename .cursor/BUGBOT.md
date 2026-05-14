# BugBot Instructions

## Releasable State

Every PR should leave Verifiers in a releasable state. Treat the merge commit as something that could be built, packaged, and published immediately after CI passes.

Request changes when a PR:

- Leaves known failing tests, lint, type checks, packaging checks, or release workflows unresolved
- Adds or weakens skips, xfails, broad exception handling, or CI conditionals to hide broken behavior
- Requires manual cleanup, local-only files, unpublished artifacts, or untracked generated files before release
- Leaves version metadata, release notes, package configuration, docs, or lockfiles inconsistent with the changed behavior

If validation is limited by missing credentials or external services, the PR should state the exact limitation while still preserving the normal releasable checks for code, packaging, and docs.

## Dependency Sources

Do not accept new or modified dependency declarations that pin packages to GitHub sources. Verifiers dependency surfaces should resolve from package indexes or other stable release channels, not repository snapshots.

Request changes for dependency specs such as:

- `git+https://github.com/...`
- `git@github.com:...`
- `package @ https://github.com/...`
- `[tool.uv.sources]` entries that point at GitHub repositories

When a GitHub-only upstream is unavoidable for an environment or integration, ask the author to make an explicit release-path decision instead of pinning the GitHub source directly in Verifiers.

## Documentation Updates

Any PR that adds or modifies core user-facing functionality as described in `docs/` must update the relevant documentation. This includes changes classes and APIs described in:

- `docs/overview.md`
- `docs/environments.md`
- `docs/evaluation.md`
- `docs/training.md`
- `docs/reference.md`
- `docs/faqs.md`

Notable information which should be available for reference, but does not neatly map to a specific documentation section, should be mentioned in `docs/faqs.md`.

If such changes are detected without a corresponding documentation update, request that the author add an entry.

## Example Environments Updates

Any PR that adds or removes an environment from the `environments/` folder must update `environments/README.md` to reflect the change. The README should:

- List the new environment under the appropriate category/pattern section
- Remove references to deleted environments
- Update the "What to look at for each pattern" section if applicable

If an environment is added or removed without a corresponding `environments/README.md` update, request that the author add the necessary changes.

## Skills Updates

Any PR that changes user-facing Prime or Verifiers workflows for environment development, browsing, review, evaluation, GEPA optimization, or RL training must update the corresponding skills under `skills/`.

This includes changes to command contracts, defaults, or behavior in:

- `docs/overview.md`
- `docs/environments.md`
- `docs/evaluation.md`
- `docs/training.md`
- `docs/faqs.md`
- `docs/prime_cli_verifiers_unification_design.md`
- `verifiers/scripts/*.py`
- `verifiers/cli/plugins/prime.py`

When these files change, verify and update any affected skill files:

- `skills/create-environments/SKILL.md`
- `skills/browse-environments/SKILL.md`
- `skills/review-environments/SKILL.md`
- `skills/evaluate-environments/SKILL.md`
- `skills/optimize-with-environments/SKILL.md`
- `skills/train-with-environments/SKILL.md`
- `skills/brainstorm/SKILL.md`

If workflow-relevant changes are detected without matching skill updates, request that the author update the impacted skills before merge.

## Environment Rollout Logic

Do not request library utilities solely because two or more environments contain similar message, state, or rollout-loop data manipulation. A few explicit lines inside an environment are often the clearest and most discoverable implementation.

In particular, do not suggest moving small helpers for selecting messages, extracting text from `state`, or juggling rollout-local fields into hidden library modules. Buried helpers are not easily discoverable by end users, clutter the public API when promoted, and make the docs responsible for enumerating every three-line convenience function.

Prefer explicit environment-local code unless the repeated logic is a framework contract, fixes a correctness bug at the boundary, or is already part of documented user-facing API. Do not ask authors to create one-off private helpers for simple rollout logic; if a few lines are used once, they should usually stay inline at the call site.

Helpers are acceptable when the logic is reused in multiple places, is a taskset-bound object that forms part of the environment contract, or is complex enough to deserve a named secondary module. Excess reliance on buried rollout-loop helpers should be treated as non-idiomatic and a code smell.
