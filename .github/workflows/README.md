# GitHub Actions Workflows

This directory contains automated workflows for the verifiers project.

## Workflows

### 1. Style (`style.yml`)
**Purpose**: Code style checking using ruff, ty, and Semgrep policy rules.

**Triggers**:
- Pull requests (opened, synchronized, reopened)
- Pushes to `main` branch

**What it does**:
- Runs ruff for linting and formatting checks
- Runs ty type checks with `uv run ty check verifiers`
- Runs Semgrep policy checks through pre-commit's isolated hook environment.
- Uses configuration from `pyproject.toml`, `.pre-commit-config.yaml`, and `.semgrep/verifiers.yml`

### 2. DevX Tag (`devx_tag.yml`)
**Purpose**: Automatically publish a `vX.Y.Z.devN` release to PyPI on every push to `main`.

**Triggers**:
- Pushes to `main`
- Manual dispatch

**What it does**:
- Computes the next dev tag from the most recent `v*` tag.
- Builds a synthetic commit on top of `main` HEAD that bumps `verifiers/__init__.py` to the new version, then annotates and pushes the tag (the bump commit is never pushed back to `main`).
- Checks out the tag, runs `uv build`, and publishes the wheel + sdist to PyPI via Trusted Publisher (`pypi-prod` environment, OIDC).
- Does **not** create a GitHub Release and does **not** require a hand-curated `assets/release/RELEASE_<tag>.md` file — those are reserved for stable releases.
- Skips itself when the push already modifies `verifiers/__init__.py` (deferring to `tag-and-release.yml`'s `auto-tag-on-main` job for stable release-prep PRs).

See `assets/release/release_workflow.md` for the full release workflow.

### 3. Tag and Release (`tag-and-release.yml`)
**Purpose**: Publish a stable `vX.Y.Z` release to PyPI and create a GitHub Release.

**Triggers**:
- Pushes to `main` that change `verifiers/__init__.py` to a non-dev version (auto-tags `vX.Y.Z`).
- Pushes of `v*` tags **excluding** `v*.dev*`.
- Manual dispatch with an existing stable tag.

**What it does**:
- Validates that the tag does not contain `.dev` (dev releases are handled by `devx_tag.yml`).
- Runs `uv build`, publishes to PyPI via Trusted Publisher (`pypi-prod` environment, OIDC), and creates a GitHub Release using `assets/release/RELEASE_<tag>.md` plus the built `dist/` artifacts.

### 4. Test (`test.yml`)
**Purpose**: Comprehensive testing with coverage reports.

**Triggers**:
- Pull requests affecting Python files, dependencies, or workflow files
- Pushes to `main`, `master`, or `develop` branches with the same file changes

**What it does**:
- Runs tests on multiple Python versions (3.12, 3.13)
- Generates coverage reports (XML, HTML, and terminal output)
- Uploads coverage to Codecov (requires `CODECOV_TOKEN` secret)
- Uploads HTML coverage reports as artifacts
- Comments on PRs with test results

## Setting Up

### Branch Protection
It's recommended to set up branch protection rules for your main branch:
1. Go to Settings → Branches
2. Add a rule for your main branch
3. Enable "Require status checks to pass before merging"
4. Select the CI jobs you want to require

## Running Checks Locally

To run checks locally the same way they run in CI:

```bash
# Ty parity with CI (Python 3.13 target configured in `pyproject.toml`)
uv run ty check verifiers

# Verifiers-specific policy lint
env PYTHONWARNINGS=ignore::SyntaxWarning uv run pre-commit run semgrep-v1-policy --all-files

# Tests
uv sync
uv run pytest tests/ -v
uv run pytest tests/ -v --cov=verifiers --cov-report=html
```
Tip: install pre-push hooks to block pushes when Ty fails:

```bash
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
```

## Customization

### Adding New Python Versions
Edit the `matrix.python-version` in the workflow files to test on additional Python versions.

### Changing Trigger Conditions
Modify the `on:` section in the workflow files to change when workflows run.

### Adding More Checks
You can extend the workflows to include:
- Type checking with mypy
- Security scanning
- Documentation building
- Package building and publishing
