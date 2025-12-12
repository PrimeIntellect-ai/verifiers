# Verbatim Copy Environment

Tests the ability of models to accurately reproduce text verbatim. Designed to compare RLM (Recursive Language Model) against standard LLM approaches.

## Key Design

The text to copy is included in the **prompt** for both RLM and standard modes. This ensures both models must actually write out the text character by character, making it a fair comparison.

- **RLM mode**: Uses `RLMEnv` - the model can use Python to write to `answer["content"]`, inspect what it wrote, and make corrections before finalizing
- **Standard mode**: Uses `SingleTurnEnv` - the model generates a one-shot response with no ability to self-correct

## Installation

```bash
uv run vf-install verbatim_copy
```

## Usage

### Basic evaluation

```bash
# RLM mode (default)
uv run vf-eval -s verbatim_copy -m gpt-4.1 -n 20

# Standard LLM mode
uv run vf-eval -s verbatim_copy -m gpt-4.1 -n 20 --env-args '{"use_rlm": false}'
```

### Programmatic usage

```python
from verbatim_copy import load_environment

# RLM mode
env = load_environment(use_rlm=True, num_samples=50)

# Standard mode
env = load_environment(use_rlm=False, num_samples=50)

# Specific difficulty
env = load_environment(difficulty="hard", num_samples=50)
```

## Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `num_samples` | int | 100 | Number of samples to generate |
| `difficulty` | str | "all" | Difficulty level: "easy", "medium", "hard", "mixed", or "all" |
| `seed` | int | 42 | Random seed for data generation |
| `use_rlm` | bool | True | Use RLMEnv (True) or SingleTurnEnv (False) |
| `max_iterations` | int | 30 | Maximum REPL iterations (RLM mode only) |
| `max_output_length` | int | 8192 | Max code output length (RLM mode only) |

## Difficulty Levels

| Level | Content Type | Description |
|-------|--------------|-------------|
| easy | Word sequences | Random common English words, familiar patterns |
| medium | Structured data | JSON records, CSV data with numbers and special chars |
| hard | Alphanumeric codes | UUIDs, short codes, no semantic cues |
| mixed | Combined | Multiple types in one sample |

The default "all" distribution: 25% easy, 35% medium, 25% hard, 15% mixed.

## Reward Functions

| Function | Weight | Description |
|----------|--------|-------------|
| `exact_match` | 1.0 | 1.0 if perfect match, 0.0 otherwise |
| `char_accuracy` | 0.0 | Proportion of characters matching at each position |
| `levenshtein_similarity` | 0.0 | 1 - (edit_distance / max_length) |

## Data Generation

Data is synthetically generated using:
- **Faker**: Realistic structured data (names, emails, addresses, etc.)
- **UUID**: Unique identifiers for hard difficulty
- **Random word sequences**: From a curated list of unambiguous words

This ensures:
1. **Novelty**: Text is not in model training data
2. **Reproducibility**: Same seed = same dataset
3. **Controlled difficulty**: Precise control over content types
