# scicode

### Overview
- **Environment ID**: `scicode`
- **Short description**: SciCode scientific Python coding tasks from research problems decomposed into executable substeps.
- **Tags**: scicode, science, coding, python, research, single-turn, train, eval

### Task
Each sample prompts the model with a SciCode subproblem, prior step context, required dependencies, and the exact function/class header to implement. The response should be one fenced Python code block containing background comments and the implementation for the requested step only.

### Quickstart
```bash
prime env install environments/scicode
prime eval run scicode -n 1 -r 1
```

### Rubric
The reward combines static checks that are available without the external numeric HDF5 answer file:

- syntactically valid Python after stripping dependency imports,
- expected function/class name is implemented,
- response is in a fenced Python block,
- no top-level examples, tests, prints, or asserts,
- includes a `# Background:` comment,
- includes at least one return statement.

The environment loads `SciCode1/SciCode` from Hugging Face and falls back to a small local sample for offline smoke tests.

### References
- Algora bounty: https://algora.io/PrimeIntellect-ai/bounties/AG9a7bN3dkaFcVL3
- Source benchmark: https://github.com/scicode-bench/SciCode
- Dataset: https://huggingface.co/datasets/SciCode1/SciCode
