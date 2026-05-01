# GitHub Actions Workflows

This directory contains automated workflows for the verifiers project.

## Workflows

### 1. Style (`style.yml`)
**Purpose**: Code style checking using ruff and ty.

**Triggers**:
- Pull requests (opened, synchronized, reopened)
- Pushes to `main` branch

**What it does**:
- Runs ruff for linting and formatting checks
- Runs ty type checks with `uv run ty check verifiers`
- Uses configuration from `pyproject.toml`

### 2. DevX Tag (`devx_tag.yml`)
**Purpose**: Auto-create rolling `vX.Y.Z.devN` pre-release tags from the latest stable release.

**Triggers**:
- Pushes to `main`
- Manual `workflow_dispatch`

**What it does**:
- Reads the latest published GitHub Release tag (e.g. `v0.1.13`).
- Bumps the patch number to form a base (e.g. `v0.1.14`).
- Picks the next free `.devN` suffix and creates `vX.Y.Z.devN` at the current SHA.
- The created tag fires `tag-and-release.yml`, which publishes the dev build to PyPI (no GitHub Release is cut for dev tags).

### 3. Test (`test.yml`)
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
