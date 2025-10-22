# Verifiers Test Suite

This directory contains the test suite for the verifiers package.

## Setup

**Prerequisites:**
- Install `uv` package manager (https://docs.astral.sh/uv/getting-started/installation/)
- Ensure Python 3.11+ is available

Install test dependencies:

```bash
uv sync
```

## Running Tests

Run all tests:

```bash
uv run pytest
```

Run specific test files:

```bash
uv run pytest tests/test_parser.py
uv run pytest tests/test_xml_parser.py
uv run pytest tests/test_think_parser.py
```

Run with coverage:

```bash
uv run pytest --cov=verifiers
```

Run only unit tests:

```bash
uv run pytest -m unit
```

## Test Structure

- `conftest.py` - Pytest configuration and shared fixtures
- `test_parser.py` - Tests for the base Parser class
- `test_xml_parser.py` - Tests for the XMLParser class  
- `test_think_parser.py` - Tests for the ThinkParser class
- `test_environment.py` - Tests for the base Environment class
- `test_singleturn_env.py` - Tests for the SingleTurnEnv class
- `test_multiturn_env.py` - Tests for the MultiTurnEnv class

## Test Markers

- `unit` - Fast unit tests (default for all current tests)
- `integration` - Integration tests
- `slow` - Slow-running tests
- `asyncio` - Async tests

## Async Testing & Mocking

The test suite includes comprehensive support for testing async Environment classes:

### AsyncOpenAI Client Mocking
- `mock_openai_client` fixture provides a fully mocked AsyncOpenAI client
- Supports both chat completions and regular completions
- No actual API calls are made during testing

### Test Datasets
- `sample_dataset` - Basic question/answer dataset
- `sample_chat_dataset` - Pre-formatted chat messages
- Custom datasets can be created using `Dataset.from_dict()`

### Async Test Examples
```python
@pytest.mark.asyncio
async def test_my_async_function(mock_openai_client):
    env = SingleTurnEnv(client=mock_openai_client, model="test", ...)
    result = await env.rollout(...)
    assert result[0] == expected_completion

# MultiTurnEnv testing
@pytest.mark.asyncio  
async def test_multiturn_conversation(mock_multiturn_env):
    # Configure sequential responses
    responses = ["response1", "response2", "final DONE"]
    mock_multiturn_env.client.chat.completions.create.side_effect = [
        create_mock_response(resp) for resp in responses
    ]
    
    completion, state = await mock_multiturn_env.rollout(...)
    assert len(completion) > 1  # Multiple turns
```

### Environment Testing
- **SingleTurnEnv**: Simple request-response testing
- **MultiTurnEnv**: Complex multi-turn conversation testing with:
  - Turn-by-turn conversation flow
  - Max turns limiting
  - Environment response integration
  - Completion detection logic
  - State management across turns
- Tests cover both chat and completion message formats
- Mocked responses simulate real OpenAI API behavior
- Error handling and edge cases are tested
- No real LLM requests are made

## Adding New Tests

1. Create test files following the `test_*.py` naming convention
2. Use the fixtures from `conftest.py` for common instances
3. Add appropriate test markers (`@pytest.mark.asyncio` for async tests)
4. Use `mock_openai_client` for Environment testing
5. Follow the existing test structure and naming conventions