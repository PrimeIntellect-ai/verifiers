# Verifiers Documentation

This directory contains source files for the `verifiers` documentation.

## Building the Documentation

### Prerequisites

```bash
# Install documentation dependencies
pip install sphinx sphinx-rtd-theme myst-parser

# Or using uv
uv add sphinx sphinx-rtd-theme myst-parser
```

### Build Commands

```bash
# Build HTML documentation
cd docs/
make html

# Or using uv from project root
cd docs/
uv run make html

# View the documentation
open build/html/index.html  # macOS
xdg-open build/html/index.html  # Linux
```