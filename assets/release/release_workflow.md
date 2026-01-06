# Release workflow

The `Tag and Release` GitHub Actions workflow (`.github/workflows/tag-and-release.yml`) publishes a new `verifiers`
version whenever a maintainer pushes a `v*` tag. Follow the checklist below to guarantee the workflow only runs once per
version and finishes cleanly.

## Before triggering a release

- Land a release prep PR on `main` with:
  - `verifiers/__init__.py` set to the final version (for example `0.1.4`).
  - Updated release notes in `notes/RELEASE_vX.Y.Z.md`.
  - Any ancillary artifacts or documentation updates that belong with the release.
- Verify CI is green on the commit you intend to tag.
- Confirm the organization secret `PYPI_TOKEN` is configured with publish permissions for the project.

## Triggering the workflow

1. From `main`, create an annotated tag that matches the version string (for example `git tag -a v0.1.4 -m "Release v0.1.4"`).
2. Push the tag to GitHub with `git push origin v0.1.4`. The push is the only automatic trigger for the workflow, so each
   version runs exactly once.
3. Watch the **Actions → Tag and Release** run to confirm `uv build`, `uv publish`, and the GitHub Release creation succeed.

> Optional: To republish an older tag without pushing again, start **Actions → Tag and Release** manually and provide the
> existing tag (for example `v0.1.4`). The job checks out that tag and performs the same build and publish steps.

## After the release

- Verify the new version appears on PyPI and that the GitHub Release contains the built `dist/` artifacts.
- Draft follow-up communication (blog post, changelog announcement) if needed.
- Open a short PR that bumps `verifiers/__init__.py` to the next development version (for example `0.1.5.dev0`).

## Troubleshooting

- **Workflow failed before publishing to PyPI**: fix the underlying issue and re-run the failed job from the Actions UI. The
  rerun builds from the same tag.
- **PyPI publish failed**: address the error locally, then re-run the workflow manually with the same tag once the issue is
  resolved. PyPI will reject duplicate uploads, so delete the partially uploaded files (if any) from the failed run before
  retrying.
- **Version mismatch error**: ensure the tag you pushed (e.g. `v0.1.4`) matches the version string committed to
  `verifiers/__init__.py` before retrying.
