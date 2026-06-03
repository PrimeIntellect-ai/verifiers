# Release workflow

The `Publish verifiers` GitHub Actions workflow (`.github/workflows/publish-verifiers.yml`) publishes `verifiers` to
PyPI. Versions are **git-driven** via `hatch-vcs` — there is no version string to hand-edit.

- **Dev pre-releases** happen automatically: every push to `main` builds and publishes `0.1.X.dev<commits-since-last-tag>`
  (installable with `pip install --pre verifiers`). No action and no per-commit tags are required.
- **Stable releases** happen when a maintainer pushes a `vX.Y.Z` tag. Building that tag yields exactly `X.Y.Z`.

## Before cutting a stable release

- Verify CI is green on the commit you intend to tag.
- Confirm the `verifiers` project on PyPI has a Trusted Publisher configured for repository
  `PrimeIntellect-ai/verifiers`, workflow `publish-verifiers.yml`, environment `pypi-prod`. The publish job authenticates
  via OIDC — no PyPI token is required for `verifiers`.

## Release notes

Release notes are a **GitHub concern, not a repo concern** — they are not stored in this repository. The
`github-release-tag` job creates the GitHub Release with `generate_release_notes: true`, so GitHub builds the notes
from the merged PRs since the previous tag (a "What's Changed" list, new contributors, and a full-changelog compare
link). Clean PR titles drive the quality of this list.

After the workflow publishes the release, a maintainer may curate the generated notes (e.g. add a short "Highlights"
section) directly on the GitHub Release before announcing it — for example with `gh release edit <tag> --notes-file -`.
There is no `RELEASE_*.md` file to author or review in a PR.

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
