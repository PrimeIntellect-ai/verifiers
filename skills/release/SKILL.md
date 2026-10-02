---
name: release
description: Release a stable verifiers version from main to PyPI and GitHub.
---

# Release Verifiers

Use the next minor version by default. Use a different `vX.Y.Z` version when the
user specifies one. Complete only the release steps that the user authorizes.

## 1. Prepare main

Run the release from the main repository:

```bash
git switch main
git pull --ff-only origin main
git status --short --branch
VERSION=vX.Y.Z
```

Make sure there are no local changes and `$VERSION` has not been released on
GitHub or PyPI. Check that dependencies in `pyproject.toml` use published stable
versions. Replace any `.dev`, alpha, beta, or release-candidate versions using
`uv`, which also updates `uv.lock`. Merge dependency changes through the normal
PR process. Pull `main` again and check for local changes before tagging.

## 2. Tag latest main

```bash
git tag "$VERSION"
git push origin "refs/tags/$VERSION"
```

Confirm that the remote tag points to the current `main` commit.

## 3. Publish

Start the stable release workflow:

```bash
gh workflow run publish-verifiers.yml \
  --repo PrimeIntellect-ai/verifiers \
  --ref main \
  -f tag="$VERSION"
```

Wait for it to finish with `gh run watch <run-id> --exit-status`.

## 4. Update the release notes

After the workflow creates the public GitHub Release, read the previous stable
release and use the same release-note format. Describe the changes since that
release, then update the new release:

```bash
gh release edit "$VERSION" \
  --repo PrimeIntellect-ai/verifiers \
  --notes-file <release-notes.md>
```

## 5. Verify

Confirm that:

- PyPI lists `${VERSION#v}` with a wheel and source archive.
- The GitHub Release is public with both files attached.
- The release notes match the prior release format.
