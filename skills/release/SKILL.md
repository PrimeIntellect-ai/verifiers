---
name: release
description: Release a stable verifiers version from main to PyPI and GitHub.
---

# Release Verifiers

By default, bump to the next minor `vX.Y.Z` version unless the user specifies a different target version, and only run the release steps the user has authorized.

## 1. Prepare main

Run the release from a clean checkout of `main`:

```bash
git switch main
git pull --ff-only origin main
git status --short --branch
VERSION=vX.Y.Z
```

Confirm that the worktree has no local changes and that `$VERSION` has not already been published to GitHub or PyPI. Check `pyproject.toml` to verify all dependencies are pinned to published stable versions. If any use `.dev`, alpha, beta, or release-candidate versions, update them with `uv` (so `uv.lock` updates too), land the change through a normal PR, and pull `main` again before tagging.

## 2. Tag latest main

```bash
git tag "$VERSION"
git push origin "refs/tags/$VERSION"
```

Confirm that the remote tag points to the current `main` commit.

## 3. Publish

Manually dispatch the stable release workflow:

```bash
gh workflow run publish.yml \
  --repo PrimeIntellect-ai/verifiers \
  --ref main \
  -f tag="$VERSION"
```

Watch the workflow run to completion with `gh run watch <run-id> --exit-status`.

## 4. Update the release notes

Once the workflow creates the public GitHub Release, summarize the changes since the previous stable release using the same release-note format, then update the new release:

```bash
gh release edit "$VERSION" \
  --repo PrimeIntellect-ai/verifiers \
  --notes-file <release-notes.md>
```

## 5. Verify

Confirm that:

- PyPI lists `${VERSION#v}` with both a wheel and a source archive.
- The GitHub Release is public with both files attached.
- The release notes match the format of prior releases.
