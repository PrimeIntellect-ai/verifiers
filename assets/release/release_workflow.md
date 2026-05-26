# Release workflow

`verifiers` ships two kinds of releases, each owned by its own workflow:

| Kind | Tag shape | Workflow | What it produces |
| --- | --- | --- | --- |
| Dev | `vX.Y.Z.devN` | `.github/workflows/devx_tag.yml` | git tag + PyPI upload |
| Stable | `vX.Y.Z` | `.github/workflows/tag-and-release.yml` | git tag + PyPI upload + GitHub Release with `assets/release/RELEASE_<tag>.md` |

Both workflows publish to PyPI via Trusted Publisher (`pypi-prod`
environment, OIDC). Trusted Publisher entries on PyPI must exist for
**both** workflow file names.

## Automatic dev releases (`devx_tag.yml`)

`devx_tag.yml` runs on every push to `main`. For each push it:

1. Computes the next dev tag from the most recent `v*` tag. If the latest
   tag is stable (`vX.Y.Z`) the next dev tag is `vX.Y.(Z+1).dev1`. If the
   latest tag is already a dev release (`vX.Y.Z.devN`) the next dev tag
   is `vX.Y.Z.dev(N+1)`.
2. Creates a synthetic commit on top of the current `main` HEAD that
   bumps `verifiers/__init__.py` to the new version. The commit is the
   parent of the annotated tag — it is **not** pushed back to `main`.
3. Pushes the tag, then checks out the tagged commit, runs `uv build`,
   and publishes the wheel + sdist to PyPI.

Dev releases intentionally do **not** create a GitHub Release and do
**not** require a hand-curated release-notes file. They are best
consumed via `pip install verifiers==X.Y.Z.devN` or
`pip install verifiers --pre`.

`devx_tag.yml` defers to the stable flow when a push already modifies
`verifiers/__init__.py`, so a release-prep PR never produces two tags
for the same commit.

To trigger a dev release manually (for example after a workflow rerun),
use **Actions → DevX Tag → Run workflow** against `main`.

## Manual stable releases (`tag-and-release.yml`)

`tag-and-release.yml` only handles non-dev tags. It rejects `v*.dev*`
tags both on push and on `workflow_dispatch`.

For a `vX.Y.Z` cut, prepare a release-prep PR on `main` with:

- `verifiers/__init__.py` set to the final version (for example
  `0.1.16`, no `.dev` suffix).
- Hand-curated release notes at `assets/release/RELEASE_vX.Y.Z.md`.
- Any ancillary artifacts or documentation updates that belong with the
  release.

Verify CI is green and confirm the `verifiers` project on PyPI has a
Trusted Publisher configured for repository
`PrimeIntellect-ai/verifiers`, workflow `tag-and-release.yml`,
environment `pypi-prod`.

When the PR is merged, `tag-and-release.yml`'s `auto-tag-on-main` job
detects the version change in `verifiers/__init__.py`, creates and
pushes `vX.Y.Z`, runs `uv build`, publishes to PyPI, and creates the
GitHub Release from `assets/release/RELEASE_vX.Y.Z.md`. `devx_tag.yml`
skips this push (because `__init__.py` changed) and resumes producing
`vX.Y.(Z+1).devN` tags on subsequent pushes.

> Optional: To republish an existing tag without pushing again, start
> **Actions → Tag and Release** manually and provide the existing tag
> (for example `v0.1.4`). The job checks out that tag and performs the
> same build and publish steps.

## After a stable release

- Verify the new version appears on PyPI and that the GitHub Release
  contains the built `dist/` artifacts.
- Draft follow-up communication (blog post, changelog announcement) if
  needed.
- No follow-up bump PR is required — the next push to `main` will
  create the first `vX.Y.(Z+1).devN` tag automatically.

## Troubleshooting

- **`devx_tag.yml` failed before pushing the tag**: re-run the workflow
  from the Actions UI. The job is idempotent: it always recomputes the
  next dev number against the live tag list.
- **`devx_tag.yml` pushed the tag but PyPI publish failed**: re-run the
  failed job from the Actions UI. PyPI rejects duplicate uploads, so if
  the wheel actually made it through, delete the partial upload first.
- **`tag-and-release.yml` reported "Dev tags are not accepted by this
  workflow"**: this is expected — use `devx_tag.yml` for dev releases.
- **`tag-and-release.yml` reported "verifiers/__init__.py bumped to dev
  version"**: never bump `__init__.py` to a `.devN` value manually; the
  auto-dev flow owns dev versions.
- **Version mismatch error**: ensure the tag you pushed (e.g. `v0.1.4`)
  matches the version string committed to `verifiers/__init__.py` at
  the tag.
