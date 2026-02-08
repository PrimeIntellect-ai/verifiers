# Development & Testing

This guide covers setup, testing, and contributing to the verifiers package.

## Table of Contents

- [Setup](#setup)
- [Project Structure](#project-structure)
- [Running Tests](#running-tests)
- [Writing Tests](#writing-tests)
- [Contributing](#contributing)
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
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
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
│   ├── scripts/        # CLI entry points
│   └── utils/          # Utilities
├── environments/       # Installable environment modules
├── configs/            # Example training configurations
├── tests/              # Test suite
└── docs/               # Documentation
```

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

The test suite provides mock OpenAI clients:

```python
from tests.mock_openai_client import MockOpenAIClient

def test_with_mock(mock_client):
    env = vf.SingleTurnEnv(client=mock_client)
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
6. **Run linting/format checks**: `uv run ruff check --fix . && uv run ruff format --check verifiers tests`
7. **Run CI-parity type checks**: `uv run ty check verifiers`
8. **Update docs** if adding/changing public APIs
9. **Submit PR** with clear description

### Code Style

- Strict `ruff` enforcement - all PRs must pass `ruff check --fix .` and `ruff format --check verifiers tests`
- `ty` must pass via `uv run ty check verifiers` to mirror CI setup (Python 3.13 target)
- Use type hints for function parameters and returns
- Write docstrings for public functions/classes
- Keep functions focused and modular
- Fail fast, fail loud - no defensive programming or silent fallbacks

### PR Checklist

- [ ] Tests pass locally (`uv run pytest tests/`)
- [ ] Linting/format checks pass (`uv run ruff check --fix . && uv run ruff format --check verifiers tests`)
- [ ] Type checks pass (`uv run ty check verifiers`)
- [ ] Pre-commit hooks pass (`uv run pre-commit run --all-files`)
- [ ] Added tests for new functionality
- [ ] Updated documentation if needed

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

# Install locally for testing
prime env install my-environment

# Test your environment
prime eval run my-environment -m gpt-4.1-mini -n 5
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
uv run ty check verifiers               # Type check (matches CI Ty target)
uv run pre-commit run --all-files     # Run all pre-commit hooks

# Environment tools
prime env init new-env                       # Create environment
prime env install new-env                    # Install environment
prime eval run new-env -m gpt-4.1-mini -n 5  # Test environment
prime eval tui                               # Browse eval results
```

### CLI Tools

 | Command | Description |
|---------|-------------|
| `prime eval run` | Run evaluations on environments |
| `prime env init` | Initialize new environment from template |
| `prime env install` | Install environment module |
| `prime lab setup` | Set up training workspace |
| `prime eval tui` | Terminal UI for browsing eval results |
| `prime rl run` | Launch Hosted Training |
| `uv run prime-rl` | Launch prime-rl training |

### Project Guidelines

- **Environments**: Installable modules with `load_environment()` function
- **Parsers**: Extract structured data from model outputs
- **Rubrics**: Define multi-criteria evaluation functions
- **Tests**: Comprehensive coverage with mocks for external dependencies
