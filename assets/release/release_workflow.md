# Release workflow

`verifiers` ships two kinds of releases:

- **Automatic dev releases** (`vX.Y.Z.devN`) — published from every push to
  `main` by the `DevX Tag` workflow (`.github/workflows/devx_tag.yml`).
- **Manual stable releases** (`vX.Y.Z`) — prepared by a maintainer via a
  release-prep PR and published by `Tag and Release`
  (`.github/workflows/tag-and-release.yml`).

Both flows ultimately go through `tag-and-release.yml`, which runs the
`uv build` → `pypa/gh-action-pypi-publish` (Trusted Publisher, OIDC,
environment `pypi-prod`) → `softprops/action-gh-release` pipeline.

## Automatic dev releases

`devx_tag.yml` runs on every push to `main`. For each push it:

1. Computes the next dev tag from the most recent `v*` tag. If the latest
   tag is stable (`vX.Y.Z`) the next dev tag is `vX.Y.(Z+1).dev1`. If the
   latest tag is already a dev release (`vX.Y.Z.devN`) the next dev tag is
   `vX.Y.Z.dev(N+1)`.
2. Creates a synthetic commit on top of the current `main` HEAD that bumps
   `verifiers/__init__.py` to the new version and writes a stub release
   notes file at `assets/release/RELEASE_<tag>.md`.
3. Annotates and pushes that tag (the bump commit only ever lives as the
   tag's target — it is **not** pushed back to `main`).
4. Dispatches `tag-and-release.yml` with the new tag as input. The
   build/publish/GitHub-release jobs check out the tag, validate that
   `verifiers/__init__.py` matches the tag, build the wheel + sdist,
   publish to PyPI, and create the GitHub Release using the auto-generated
   notes file.

`devx_tag.yml` defers to the manual flow when a push already modifies
`verifiers/__init__.py`, so a release-prep PR never produces two tags for
the same commit.

To trigger a dev release manually (for example after a workflow rerun),
use **Actions → DevX Tag → Run workflow** against `main`.

## Manual stable releases

For a `vX.Y.Z` cut, prepare a release-prep PR on `main` with:

- `verifiers/__init__.py` set to the final version (for example `0.1.16`,
  no `.dev` suffix).
- Hand-curated release notes at `assets/release/RELEASE_vX.Y.Z.md`.
- Any ancillary artifacts or documentation updates that belong with the
  release.

Verify CI is green and confirm the `verifiers` project on PyPI has a
Trusted Publisher configured for repository `PrimeIntellect-ai/verifiers`,
workflow `tag-and-release.yml`, environment `pypi-prod`.

When the PR is merged, `tag-and-release.yml`'s `auto-tag-on-main` job
detects the version change in `verifiers/__init__.py`, creates and pushes
`vX.Y.Z`, and runs build → PyPI publish → GitHub Release. The
`devx_tag.yml` workflow skips this push (because `__init__.py` changed)
and resumes producing `vX.Y.(Z+1).devN` tags on subsequent pushes.

> Optional: To republish an existing tag without pushing again, start
> **Actions → Tag and Release** manually and provide the existing tag
> (for example `v0.1.4`). The job checks out that tag and performs the
> same build and publish steps.

## After a stable release

- Verify the new version appears on PyPI and that the GitHub Release
  contains the built `dist/` artifacts.
- Draft follow-up communication (blog post, changelog announcement) if
  needed.
- No follow-up bump PR is required — the next push to `main` will create
  the first `vX.Y.(Z+1).devN` tag automatically.

## Troubleshooting

- **`devx_tag.yml` failed before pushing the tag**: re-run the workflow
  from the Actions UI. The job is idempotent: it always recomputes the
  next dev number against the live tag list.
- **`devx_tag.yml` succeeded but the dispatched `Tag and Release` run
  failed**: fix the underlying issue and trigger
  **Actions → Tag and Release → Run workflow** manually with the same tag.
- **PyPI publish failed**: address the error locally, then re-run the
  workflow manually with the same tag once the issue is resolved. PyPI
  will reject duplicate uploads, so delete the partially uploaded files
  (if any) from the failed run before retrying.
- **Version mismatch error**: ensure the tag you pushed (e.g. `v0.1.4`)
  matches the version string committed to `verifiers/__init__.py` at the
  tag.
