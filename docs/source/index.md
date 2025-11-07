# Verifiers Documentation

Welcome to Verifiers! This library provides a flexible framework for creating RL environments and evaluations with custom multi-turn interaction protocols.

```{toctree}
:maxdepth: 2
:hidden:

overview
environments
components
training
development
api_reference
```

## What is Verifiers?

Verifiers enables you to:
- Define custom interaction protocols between models and environments
- Build agents, multi-turn conversations, tool-augmented reasoning, and interactive games
- Create reusable evaluation environments with multi-criteria reward functions
- Train models with the included RL trainer (via `vf-rl`) or integrate with other RL frameworks

Key features:
- **First-class OpenAI-compatibility** for `ChatCompletions` and `Completions`
- **Extensible multi-turn interactions** via `MultiTurnEnv` 
- **Native tool calling** support with `ToolEnv`
- **Modular reward functions** through `Rubric` classes
- **End-to-end async compatibility** with sync support where you want it
- **Full-spectrum scaling** from CPU evaluations in Jupyter to multi-node GPU RL training
- **Environments as Python modules** for easy installation, sharing, and reuse

## Installation

### Basic Installation

For evaluation and API model usage:
```bash
uv add verifiers
```

### Training Support

For RL training with the included trainer:
```bash
uv add 'verifiers[rl]'
```

To use the latest `main` with RL extras:
```bash
uv add 'verifiers[rl] @ git+https://github.com/PrimeIntellect-ai/verifiers.git@main'
```

### Latest Development Version

To use the latest `main` branch:
```bash
uv add verifiers@git+https://github.com/PrimeIntellect-ai/verifiers.git
```

### Development Setup

For contributing to verifiers:
```bash
git clone https://github.com/PrimeIntellect-ai/verifiers.git
cd verifiers
uv sync --all-extras && uv pip install flash-attn --no-build-isolation
uv run pre-commit install
```

### Integration with prime-rl

For large-scale FSDP training, see [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl).

### Integration with Prime Intellect Environments Hub

Coming soon.

## Documentation

### Getting Started

**[Overview](overview.md)** — Core concepts and architecture. Start here if you're new to Verifiers to understand how environments orchestrate interactions.

**[Environments](environments.md)** — Creating custom interaction protocols with `MultiTurnEnv`, `ToolEnv`, and basic rubrics.

### Advanced Usage

**[Components](components.md)** — Advanced rubrics, tools, parsers, with practical examples. Covers judge rubrics, tool design, and complex workflows.

**[Training](training.md)** — RL training and hyperparameter tuning. Read this when you're ready to train models with your environments.

### Reference

**[Development](development.md)** — Contributing to verifiers

**[Type Reference](api_reference.md)** — Understanding data structures
