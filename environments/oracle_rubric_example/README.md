# oracle-rubric-example

<a href="https://github.com/PrimeIntellect-ai/verifiers/tree/main/environments/oracle_rubric_example">
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="Source Code">
</a>

### Overview
- **Environment ID**: `oracle-rubric-example`
- **Short description**: Demonstrates `OracleRubric` by scoring generations with an API-style oracle that returns a similarity property.
- **Tags**: oracle, rubric, single-turn, scoring, api

### Task
- **Type**: single-turn
- **Rubric overview**:
  - Parses assistant output.
  - Sends parsed text to an oracle (`SimilarityAPIServer`).
  - Extracts a numeric property (`similarity`).
  - Compares property against `answer.target` with tolerance `answer.threshold`.

### Quickstart
```bash
prime eval run oracle-rubric-example
```

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `system_prompt` | str \| None | "Answer with one short sentence that addresses the question directly." | Optional system prompt |

### Notes
- This example is self-contained and does not require external APIs.
- To use a real neural network or HTTP API, replace `SimilarityAPIServer` and keep the same `oracle_fn` / `property_extractor` pattern.
