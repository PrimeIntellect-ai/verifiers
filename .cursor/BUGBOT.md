# BugBot Instructions

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
