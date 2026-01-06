# Contributing & Testing

This guide covers setup, testing, and contributing to the verifiers package.

## Setup

### Prerequisites
- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
# Clone and install for development
git clone https://github.com/PrimeIntellect-ai/verifiers.git
cd verifiers
uv sync --all-extras
uv run pre-commit install
```

## Project Structure

```
verifiers/
├── verifiers/          # Main package
│   ├── envs/           # Environment classes
│   ├── parsers/        # Parser classes  
│   ├── rubrics/        # Rubric classes
│   └── utils/          # Utilities
├── environments/       # Installable environment modules
├── examples/           # Usage examples
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
uv run pytest tests/test_parsers.py

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

The test suite includes 130+ tests covering parsers, rubrics, and environments. The test suite does not currently cover example environments or the trainer. If you require robust performance guarantees for training, you will likely want to use [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl).

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
6. **Update docs** if adding/changing public APIs
7. **Submit PR** with clear description

### Code Style

- Follow existing conventions in the codebase
- Use type hints for function parameters and returns
- Write docstrings for public functions/classes
- Keep functions focused and modular

### PR Checklist

- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Updated documentation if needed
- [ ] No breaking changes (or clearly documented)

## Common Issues

### Import Errors
```bash
# Ensure package is installed in development mode
uv pip install -e .
```

### Async Test Issues
```bash
# May need nest-asyncio for some environments
uv add nest-asyncio
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
vf-init my-environment

# Install locally for testing
vf-install my-environment

# Test your environment
vf-eval my-environment -m gpt-4.1-mini -n 5
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
uv sync --all-extras

# Run tests
uv run pytest tests/                   # All tests
uv run pytest tests/ -xvs              # Debug mode
uv run pytest tests/ --cov=verifiers   # With coverage

# Run environment tests
uv run pytest tests/test_envs.py -vv              # All environments
uv run pytest tests/test_envs.py -k math_python   # Specific environment

# Environment tools
vf-init new-env                        # Create environment
vf-install new-env                     # Install environment
vf-eval new-env                        # Test environment
vf-tui                                 # Browse eval results in your terminal

# Documentation
cd docs && make html                   # Build docs
```

### Project Guidelines

- **Environments**: Installable modules with `load_environment()` function
- **Parsers**: Extract structured data from model outputs
- **Rubrics**: Define multi-criteria evaluation functions
- **Tests**: Comprehensive coverage with mocks for external dependencies

For more details, see the full documentation at [readthedocs](https://verifiers.readthedocs.io).