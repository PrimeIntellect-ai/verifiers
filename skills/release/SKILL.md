---
name: release
description: Prepare and publish stable verifiers releases to PyPI and GitHub. Use when checking release readiness, creating a stable vX.Y.Z tag and draft release, dispatching publish-verifiers.yml, monitoring publication, or recovering release metadata.
---

# Release Verifiers

## Goal

Publish one stable `verifiers` version from a reviewed commit on `main`.

Releases use Git tags and `hatch-vcs`. Do not add or bump a project version in
`pyproject.toml`. Every push to `main` publishes the next `.dev<N>` build. A
stable release requires an existing `vX.Y.Z` tag and a manual dispatch of
`.github/workflows/publish-verifiers.yml`.

Treat these as separate external changes:

1. Push the stable tag.
2. Create the draft GitHub Release.
3. Dispatch the PyPI workflow.

Do only the changes the user authorized.

## 1. Select the release

Work from the main repository or a clean release worktree. Refresh remote state:

```bash
git fetch origin main --tags
gh release list --repo PrimeIntellect-ai/verifiers --limit 10
git log -1 --format='%H%n%s' origin/main
```

Choose a SemVer patch, minor, or major version. Use the `vX.Y.Z` tag form. Check
that neither the tag nor the PyPI version exists:

```bash
NEW=v0.3.1
git ls-remote --tags origin "refs/tags/$NEW"
curl -sS -o /dev/null -w '%{http_code}\n' \
  "https://pypi.org/pypi/verifiers/${NEW#v}/json"
```

Expect no tag output and an HTTP `404`. Stop if either version already exists.

## 2. Check release prerequisites

Update `main` without merging:

```bash
git switch main
git pull --ff-only origin main
git status --short --branch
```

Require a clean worktree. Record the exact release commit:

```bash
RELEASE_SHA=$(git rev-parse HEAD)
```

Check that `pyproject.toml` uses stable dependency versions:

```bash
uv run --no-sync python \
  skills/release/scripts/check_stable_dependencies.py
```

This check rejects explicit prerelease versions such as `.dev`, `a`, `b`, and
`rc`. It also rejects direct Git and URL dependencies. Replace each rejected
dependency with a published stable version. Update `uv.lock` before release.
Do not tag while this check fails.

Review the checks for `RELEASE_SHA`. The publish workflow does not wait for the
test and style workflows:

```bash
gh api \
  "repos/PrimeIntellect-ai/verifiers/commits/$RELEASE_SHA/check-runs" \
  --jq '[.check_runs[] | {name, status, conclusion, html_url}]'
```

Resolve failed checks before release. If the user explicitly accepts a known
failure, record the exact accepted commit and failure in the handoff.

## 3. Draft the release notes

Use the previous stable release as the style reference. Review the actual diff
and PR bodies. Do not infer breaking changes only from titles:

```bash
PREV=$(gh release list --repo PrimeIntellect-ai/verifiers --limit 1 \
  --json tagName --jq '.[0].tagName')
gh release view "$PREV" --repo PrimeIntellect-ai/verifiers \
  --json body --jq .body
git log --first-parent --reverse --oneline "$PREV".."$RELEASE_SHA"
```

Use this release-note structure:

```markdown
## Highlights

- **Feature name.** User-facing summary with PR references.

## Breaking

### Area

- **Break.** State the replacement or migration.

## Changelog
<!-- GitHub-generated PR list -->

## New Contributors
<!-- GitHub-generated contributor list, when present -->

**Full Changelog**: <comparison URL>
```

Generate the complete PR and contributor list for the exact range:

```bash
gh api --method POST \
  repos/PrimeIntellect-ai/verifiers/releases/generate-notes \
  -f tag_name="$NEW" \
  -f previous_tag_name="$PREV" \
  -f target_commitish="$RELEASE_SHA" \
  --jq .body
```

Rename the generated `## What's Changed` heading to `## Changelog`. Put curated
highlights and breaking changes before it. Include only the net change at the
release commit. Keep reverted changes in the generated changelog, but do not
describe them as highlights.

## 4. Push the stable tag

Verifiers uses lightweight stable tags. Tag only the recorded commit:

```bash
git tag "$NEW" "$RELEASE_SHA"
git push origin "refs/tags/$NEW"
git ls-remote --tags origin "refs/tags/$NEW"
```

Pushing the tag triggers `.github/workflows/sync-docs.yml`. Check that workflow
separately. A docs-sync failure does not mean the package publish failed, but it
must be reported.

Do not move a pushed stable tag. Stop and ask before deleting or replacing one.

## 5. Create the draft GitHub Release

Create the full notes in a temporary file, then create a draft for the existing
tag:

```bash
gh release create "$NEW" \
  --repo PrimeIntellect-ai/verifiers \
  --draft \
  --verify-tag \
  --title "$NEW" \
  --notes-file <release-notes.md>

gh release view "$NEW" --repo PrimeIntellect-ai/verifiers \
  --json tagName,name,isDraft,targetCommitish,url \
  --jq .
```

Expect `tagName` to equal `NEW` and `isDraft` to be `true`. Draft URLs can use an
internal `untagged-*` path. That URL changes to `/tag/$NEW` when published.

Use `gh release edit` for later note changes. Avoid a body-only REST patch on a
draft because it can detach the visible tag association.

Stop here unless the user also authorized PyPI publication.

## 6. Dispatch and monitor publication

The stable workflow accepts only an existing tag that matches `vX.Y.Z`:

```bash
gh workflow run publish-verifiers.yml \
  --repo PrimeIntellect-ai/verifiers \
  --ref main \
  -f tag="$NEW"
```

The command prints the run URL. Monitor that run through completion:

```bash
gh run watch <run-id> \
  --repo PrimeIntellect-ai/verifiers \
  --interval 5 \
  --exit-status
```

The successful job order is:

1. `build-tag` checks out the tag and runs `uv build`.
2. `publish-tag` publishes the wheel and source archive to PyPI with trusted
   publishing.
3. `github-release-tag` uploads both files and publishes the GitHub Release.

The workflow appends generated notes when it publishes an existing draft. This
can add a second changelog and contributor section. Inspect the published body.
If duplicate sections exist, remove only the appended duplicate with
`gh release edit`. Preserve curated notes and the first complete changelog.

## 7. Verify the public release

Verify PyPI metadata and both distributions:

```bash
curl -fsS "https://pypi.org/pypi/verifiers/${NEW#v}/json" | jq '{
  version: .info.version,
  files: [.urls[] | {filename, size, sha256: .digests.sha256}]
}'
```

Verify the GitHub Release, assets, notes, and tag:

```bash
gh release view "$NEW" --repo PrimeIntellect-ai/verifiers \
  --json tagName,isDraft,isPrerelease,publishedAt,url,assets,body
git ls-remote --tags origin "refs/tags/$NEW"
```

The release is complete only when:

- the workflow succeeded;
- PyPI shows the stable version, wheel, and source archive;
- the GitHub Release is public and has both assets;
- the GitHub asset SHA-256 digests match PyPI;
- the release notes have one changelog and the correct comparison link; and
- the docs-sync result is checked and reported separately.

## Recovery

- If the build fails before PyPI publication, diagnose the tagged source. Do not
  move the tag without explicit approval.
- If PyPI succeeds but `github-release-tag` fails, do not rerun the complete
  workflow. Stable publication does not use `skip-existing`. Rerun failed jobs
  only, or repair the GitHub Release directly.
- If the release body contains duplicate generated notes, edit the body after
  publication and keep the curated first copy.
- If docs sync reports `Bad credentials`, fix `DOCS_SYNC_PAT` before rerunning
  `.github/workflows/sync-docs.yml`.
