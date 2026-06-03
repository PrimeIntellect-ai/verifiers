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
- `publish-verifiers.yml` publishes `verifiers` with trusted publishing. On every push to `main` it builds and publishes a pre-release (`skip-existing`), and on a pushed `vX.Y.Z` tag (or manual dispatch) it publishes that stable release and creates a GitHub release.
- `publish-tasksets.yml` publishes `tasksets` from `tasksets-v*` tags with trusted publishing. On `main`, it creates `tasksets-v<version>` when `packages/tasksets/tasksets/__init__.py` defines `__version__` and the matching remote tag does not already exist, then builds and publishes from that tag in the same workflow run.
- `publish-harnesses.yml` publishes `harnesses` from `harnesses-v*` tags with trusted publishing. On `main`, it creates `harnesses-v<version>` when `packages/harnesses/harnesses/__init__.py` defines `__version__` and the matching remote tag does not already exist, then builds and publishes from that tag in the same workflow run.
- `publish-verifiers-rl.yml` publishes `verifiers-rl` from `verifiers-rl-v*` tags.

#### `verifiers` versioning convention
`verifiers` versions are **git-driven** via `hatch-vcs` (setuptools-scm); there is no hand-edited version literal. `verifiers/__init__.py` reads `__version__` from installed package metadata (`importlib.metadata`).

- The version is computed from `git describe` against core `vX.Y.Z` release tags (the describe filter ignores `.dev`/`rc` tags and the `tasksets-`/`harnesses-`/`verifiers-rl-` namespaces). `no-local-version` strips the `+g<sha>` segment that PyPI rejects.
- Any commit that is not exactly on a release tag builds the next release as a pre-release, e.g. `0.1.15.dev<commits-since-v0.1.14>`. Every push to `main` publishes one automatically — no per-commit tags are created. Install the latest with `pip install --pre verifiers`.
- To cut a stable release, push a release tag: `git tag v0.1.15 && git push origin v0.1.15`. Building that tag yields exactly `0.1.15`, which `publish-verifiers.yml` publishes to PyPI alongside a GitHub release.

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
