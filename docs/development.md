# Development & Testing

This guide covers setup, testing, and contributing to the verifiers package.

## Table of Contents

- [Setup](#setup)
- [Project Structure](#project-structure)
- [Prime CLI Plugin Export](#prime-cli-plugin-export)
- [Running Tests](#running-tests)
- [Writing Tests](#writing-tests)
- [Contributing](#contributing)
- [Contributor Practices](#contributor-practices)
- [Common Issues](#common-issues)
- [Environment Development](#environment-development)
- [Quick Reference](#quick-reference)

## Setup

### Prerequisites
- Python 3.13 recommended for CI parity with Ty checks
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
# Clone and install for development
git clone https://github.com/PrimeIntellect-ai/verifiers.git
cd verifiers

# CPU-only development:
uv sync

# GPU-based trainer development:
uv sync --all-extras

# Install pre-commit hooks (including pre-push Ty gate):
uv run pre-commit install
```

## Project Structure

```
verifiers/
├── verifiers/          # Main package
│   ├── envs/           # Environment classes
│   │   ├── integrations/   # Third-party wrappers (TextArena, ReasoningGym)
│   │   └── experimental/   # Newer environments (MCP, Harbor, etc.)
│   ├── parsers/        # Parser classes  
│   ├── rubrics/        # Rubric classes
│   ├── rl/             # Training infrastructure
│   │   ├── inference/  # vLLM server utilities
│   │   └── trainer/    # Trainer implementation
│   ├── cli/            # Prime-facing CLI modules and plugin exports
│   ├── scripts/        # Compatibility wrappers around verifiers/cli commands
│   └── utils/          # Utilities
├── environments/       # Installable environment modules
├── configs/            # Example training configurations
├── tests/              # Test suite
└── docs/               # Documentation
```

## Prime CLI Plugin Export

Verifiers exports a plugin consumed by `prime` so command behavior is sourced from verifiers modules.

Entry point:

```python
from verifiers.cli.plugins.prime import get_plugin

plugin = get_plugin()
```

The plugin exposes:

- `api_version` (current: `1`)
- command modules:
  - `eval_module` (`verifiers.cli.commands.eval`)
  - `gepa_module` (`verifiers.cli.commands.gepa`)
  - `install_module` (`verifiers.cli.commands.install`)
  - `init_module` (`verifiers.cli.commands.init`)
  - `setup_module` (`verifiers.cli.commands.setup`)
  - `build_module` (`verifiers.cli.commands.build`)
- `build_module_command(module_name, args)` to construct subprocess invocation for a command module

Contributor guidance:

- Add new prime-facing command logic under `verifiers/cli/commands/`.
- Export new command modules through `PrimeCLIPlugin` in `verifiers/cli/plugins/prime.py`.
- Keep `verifiers/scripts/*` as thin compatibility wrappers that call into `verifiers/cli`.

## Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run with coverage
uv run pytest tests/ --cov=verifiers --cov-report=html

# Run specific test file
uv run pytest tests/test_parser.py

# Stop on first failure with verbose output
uv run pytest tests/ -xvs

# Run tests matching a pattern
uv run pytest tests/ -k "xml_parser"

# Run environment tests
uv run pytest tests/test_envs.py -vv

# Run environment tests across all CPU cores
uv run pytest -n auto tests/test_envs.py -vv

# Run specific environment tests
uv run pytest tests/test_envs.py -k math_python
```

The test suite includes 380+ tests covering parsers, rubrics, environments, and utilities.

## Writing Tests

### Test Structure

```python
class TestFeature:
    """Test the feature functionality."""
    
    def test_basic_functionality(self):
        """Test normal operation."""
        # Arrange
        feature = Feature()
        
        # Act
        result = feature.process("input")
        
        # Assert
        assert result == "expected"
    
    def test_error_handling(self):
        """Test error cases."""
        with pytest.raises(ValueError):
            Feature().process(invalid_input)
```

### Using Mocks

The test suite provides a `MockClient` in `conftest.py` that implements the `Client` interface:

```python
def test_with_mock(mock_client):
    mock_client.set_default_responses(chat_response="test answer")
    env = vf.SingleTurnEnv(client=mock_client, model="test", ...)
    # Test without real API calls
```

### Guidelines

1. **Test both success and failure cases**
2. **Use descriptive test names** that explain what's being tested
3. **Leverage existing fixtures** from `conftest.py`
4. **Group related tests** in test classes
5. **Keep tests fast** - use mocks instead of real API calls

## Contributing

### Workflow

1. **Fork** the repository
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make changes** following existing patterns
4. **Add tests** for new functionality
5. **Run tests**: `uv run pytest tests/`
6. **Install hooks once per clone**: `uv run pre-commit install`
7. **Commit and push** (hooks run automatically on each commit/push)
8. **Update docs** if adding/changing public APIs
9. **Submit PR** with clear description

### Code Style

- Strict `ruff` enforcement via pre-commit hooks
- `ty` runs in the pre-push hook via `uv run --python 3.13 ty check verifiers`
- Use type hints for function parameters and returns
- Write docstrings for public functions/classes
- Keep functions focused and modular
- Fail fast, fail loud - no defensive programming or silent fallbacks

### PR Checklist

- [ ] Tests pass locally (`uv run pytest tests/`)
- [ ] Pre-commit and pre-push hooks pass on latest commit/push
- [ ] Added tests for new functionality
- [ ] Updated documentation if needed

## Contributor Practices

### Public Surface

Treat public config, docs, starter examples, skills, and generated agent
guidance as one surface. If a behavior changes for users, update all matching
surfaces in the same patch.

For TOML config, keep one shape across eval, GEPA, RL, and Hosted Training.
Normalize old or alternate inputs at the loader boundary, then keep examples on
the current golden path.

For v1 Taskset/Harness environments, put task data, task-owned tools, user
behavior, metrics, rewards, and task-specific configuration on the `Taskset`.
Use the base `vf.Harness` unless the harness owns a reusable execution adapter
such as a CLI, framework program, sandboxed program, or nested harness flow.

### Validation By Change Type

- Core runtime or shared config parsing: run the focused unit tests plus `uv run pre-commit run --all-files`.
- Example environment behavior: run the focused tests and a real `prime eval run` smoke when credentials and endpoint access are available.
- Environment packaging: exercise `tests/test_envs.py` for the changed environment so a fresh venv installs the environment package and its dependencies.
- Docs or generated agent guidance: run `uv run python scripts/sync.py` and include the regenerated files.
- Release prep: verify the version source, release notes commit range, `uv build`, and final worktree status.
- PR/CI follow-up: inspect the live review thread, check run, or log before patching, then rerun the smallest check that proves the fix.

### Downstream Checks

Before changing dependencies, optional extras, lockfiles, exported config fields,
or upload/eval metadata, trace the consumers in `prime-cli`, `prime-rl`, Hosted
Training, and public docs when they are in scope. Update the consumer or document
the compatibility boundary rather than assuming transitive behavior remains
safe.

## Common Issues

### Import Errors
```bash
# Ensure package is installed in development mode
uv sync
```

### Integration Tests
```bash
# Install optional dependencies for specific integrations
uv sync --extra ta   # for TextArenaEnv
uv sync --extra rg   # for ReasoningGymEnv
```

### Test Failures
```bash
# Debug specific test
uv run pytest tests/test_file.py::test_name -vvs --pdb
```

## Environment Development

### Creating a New Environment Module

```bash
# Initialize template
prime env init my-environment

# Include an explicit harness loader when needed
prime env init my-environment --with-harness

# Install locally for testing
prime env install my-environment

# Test your environment
prime eval run my-environment -m openai/gpt-4.1-mini -n 5
```

### Environment Module Structure

```python
# my_environment.py
import verifiers as vf

def load_environment(**kwargs):
    """Load the environment."""
    dataset = vf.load_example_dataset("dataset_name")
    parser = vf.XMLParser(fields=["reasoning", "answer"])
    
    def reward_func(parser, completion, answer, **kwargs):
        return 1.0 if parser.parse_answer(completion) == answer else 0.0
    
    rubric = vf.Rubric(
        funcs=[reward_func, parser.get_format_reward_func()],
        weights=[1.0, 0.2],
        parser=parser
    )
    
    return vf.SingleTurnEnv(
        dataset=dataset,
        parser=parser,
        rubric=rubric,
        **kwargs
    )
```

## Quick Reference

### Essential Commands

```bash
# Development setup
uv sync                               # CPU-only
uv sync --all-extras                  # With RL/training extras
uv run pre-commit install             # One-time per clone (installs pre-commit + pre-push)

# Run tests
uv run pytest tests/                  # All tests
uv run pytest tests/ -xvs             # Debug mode
uv run pytest tests/ --cov=verifiers  # With coverage

# Run environment tests
uv run pytest tests/test_envs.py -vv              # All environments
uv run pytest tests/test_envs.py -k math_python   # Specific environment

# Linting
uv run ruff check --fix .             # Fix lint errors
uv run ruff format --check verifiers tests  # Verify Python formatting
uv run ty check verifiers             # Type check (matches CI Ty target)

# Environment tools
prime env init new-env                       # Create taskset-first v1 environment
prime env install new-env                    # Install environment
prime eval run new-env -m openai/gpt-4.1-mini -n 5  # Test environment
prime eval view                              # Browse evals in the tree browser
```

### CLI Tools

 | Command | Description |
|---------|-------------|
| `prime eval run` | Run evaluations on environments |
| `prime env init` | Initialize new environment from template |
| `prime env install` | Install environment module |
| `prime lab setup` | Set up training workspace |
| `prime eval view` | Terminal UI for browsing evals and rollout details |
| `prime rl run` | Launch Hosted Training |

### Project Guidelines

- **Environments**: Installable modules with `load_environment()` function
- **Parsers**: Extract structured data from model outputs
- **Rubrics**: Define multi-criteria evaluation functions
- **Tests**: Comprehensive coverage with mocks for external dependencies
