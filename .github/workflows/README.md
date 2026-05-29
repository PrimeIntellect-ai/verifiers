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
- Runs ty type checks with `uv run ty check verifiers packages/tasksets/tasksets packages/harnesses/harnesses`
- Runs Semgrep policy checks from the isolated `policy` dependency group.
- Uses configuration from `pyproject.toml`, `.pre-commit-config.yaml`, and `.semgrep/verifiers.yml`

### 2. Test (`test.yml`)
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

### 3. Package Publishing
**Purpose**: Build and publish PyPI packages.

**Workflows**:
- `tag-and-release.yml` publishes `verifiers` from `v*` tags with trusted publishing.
- `publish-tasksets.yml` publishes `tasksets` from `tasksets-v*` tags with `PYPI_TASKSETS_TOKEN`. On `main`, it creates `tasksets-v<version>` when the current package version has no matching remote tag and is not published on PyPI.
- `publish-harnesses.yml` publishes `harnesses` from `harnesses-v*` tags with `PYPI_HARNESSES_TOKEN`. On `main`, it creates `harnesses-v<version>` when the current package version has no matching remote tag and is not published on PyPI.
- `publish-verifiers-rl.yml` publishes `verifiers-rl` from `verifiers-rl-v*` tags.

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
uv run ty check verifiers packages/tasksets/tasksets packages/harnesses/harnesses

# Verifiers-specific policy lint
env PYTHONWARNINGS=ignore::SyntaxWarning uv run --no-dev --group policy semgrep --metrics=off --disable-version-check --config .semgrep/verifiers.yml --error --quiet

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
