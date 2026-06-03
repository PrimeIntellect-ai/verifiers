# Release workflow

The `Publish verifiers` GitHub Actions workflow (`.github/workflows/publish-verifiers.yml`) publishes `verifiers` to
PyPI. Versions are **git-driven** via `hatch-vcs` — there is no version string to hand-edit.

- **Dev pre-releases** happen automatically: every push to `main` builds and publishes `0.1.X.dev<commits-since-last-tag>`
  (installable with `pip install --pre verifiers`). No action and no per-commit tags are required.
- **Stable releases** happen when a maintainer pushes a `vX.Y.Z` tag. Building that tag yields exactly `X.Y.Z`.

## Before cutting a stable release

- Land a PR on `main` with the release notes in `assets/release/RELEASE_vX.Y.Z.md` (see
  [Writing the release notes](#writing-the-release-notes)) and any ancillary artifacts/documentation for the release.
- Verify CI is green on the commit you intend to tag.
- Confirm the `verifiers` project on PyPI has a Trusted Publisher configured for repository
  `PrimeIntellect-ai/verifiers`, workflow `publish-verifiers.yml`, environment `pypi-prod`. The publish job authenticates
  via OIDC — no PyPI token is required for `verifiers`.

## Writing the release notes

GitHub Releases are only created for **stable `vX.Y.Z` tags**, and the body is read verbatim from a file in this
directory by the workflow's `github-release-tag` job:

```yaml
body_path: assets/release/RELEASE_${tag}.md
```

So the notes are part of the repo, reviewed in the release-prep PR, and must exist on the tagged commit.

Rules:

- **Filename must match the tag exactly.** Tag `v0.1.15` -> `assets/release/RELEASE_v0.1.15.md`. If the file is missing
  when the tag is built, the `github-release-tag` job fails.
- **Dev pre-releases do not need a notes file.** Pushes to `main` publish `0.1.X.dev<N>` to PyPI but do **not** create a
  GitHub Release, so no `RELEASE_*.dev*.md` is required. (The older `RELEASE_*.dev*.md` files predate this workflow and
  are kept only as history; don't add new ones.)
- Author the notes in the release-prep PR, before tagging, so they are reviewed alongside the release.

### Format

Follow the structure used by recent releases (e.g. `RELEASE_v0.1.14.md`):

```markdown
# Verifiers vX.Y.Z Release Notes

*Date:* MM/DD/YYYY

## Highlights since <previous-release>

- **Short theme.** One or two sentences describing the most user-visible changes.
- **Another theme.** Group related PRs into a narrative rather than listing every commit.

## Changes included in vX.Y.Z (since <previous-release>)

### Features and enhancements

- Short description (#PR)

### Fixes and maintenance

- Short description (#PR)

**Full Changelog**: https://github.com/PrimeIntellect-ai/verifiers/compare/<previous-release>...vX.Y.Z
```

- `<previous-release>` is the last stable tag (e.g. `v0.1.14`). The compare link gives reviewers the full diff.
- Section headings are flexible — group changes in whatever buckets read clearly (e.g. `### v1 runtime`,
  `### Harnesses and packaging`). The `### Features and enhancements` / `### Fixes and maintenance` split is the common
  default.
- To draft the change list quickly, generate the merged-PR log between tags and edit it down:

  ```bash
  git log --pretty='- %s' v0.1.14..HEAD
  ```

## Cutting a stable release

1. From the commit you want to release, create an annotated tag matching the version (for example
   `git tag -a v0.1.15 -m "Release v0.1.15"`).
2. Push the tag with `git push origin v0.1.15`. The pushed tag is the trigger, so each version runs exactly once.
3. Watch the **Actions → Publish verifiers** run to confirm `uv build`, the PyPI publish (via
   `pypa/gh-action-pypi-publish` using OIDC), and the GitHub Release creation succeed. The publish jobs run in the
   `pypi-prod` environment.

> Optional: To republish an existing tag, start **Actions → Publish verifiers** manually and provide the existing tag
> (for example `v0.1.15`). The job checks out that tag and performs the same build and publish steps.

## After the release

- Verify the new version appears on PyPI and that the GitHub Release contains the built `dist/` artifacts.
- Dev pre-releases automatically resume on the next push to `main` as `0.1.16.dev<N>` (the next release guessed from the
  new tag). No development-version bump PR is needed.

## Troubleshooting

- **Workflow failed before publishing to PyPI**: fix the underlying issue and re-run the failed job from the Actions UI.
  The rerun builds from the same ref.
- **PyPI publish failed**: address the error locally, then re-run the workflow. PyPI rejects duplicate uploads; the dev
  job uses `skip-existing`, but for a stable tag delete any partially uploaded files from the failed run before retrying.
- **OIDC / trusted publishing rejected**: confirm the PyPI Trusted Publisher entry names the current workflow file
  (`publish-verifiers.yml`) and the `pypi-prod` environment.
