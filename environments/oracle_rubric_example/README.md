# oracle-rubric-example

<a href="https://github.com/PrimeIntellect-ai/verifiers/tree/main/environments/oracle_rubric_example">
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="Source Code">
</a>

### Overview
- **Environment ID**: `oracle-rubric-example`
- **Short description**: Demonstrates `OracleRubric` on a SMILES editing task: make minimal edits to move solubility up or down.
- **Tags**: oracle, rubric, single-turn, chemistry, smiles, api

### Task
- **Type**: single-turn
- **Rubric overview**:
  - Parses assistant output and extracts a SMILES candidate.
  - Rubric 1 (normal `Rubric`): similarity reward from direct string comparison (no oracle call).
  - Rubric 2 (`OracleRubric`): calls a predict endpoint client (`SolubilityPredictClient`) and returns oracle output.
  - `score_function` (`solubility_modification_func`) reads `answer` directly and computes directional reward.
  - No target/threshold/property extractor hooks are required in this simplified configuration.
  - Final reward is the sum of both rubric scores.

### Quickstart
```bash
prime eval run oracle-rubric-example
```

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `system_prompt` | str \| None | "You are a medicinal chemistry assistant. Make minimal edits and return only one SMILES string." | Optional system prompt |

### Notes
- This example is self-contained and does not require external APIs.
- Intended remote API contract:
  - `POST http://localhost:0000/predict`
  - request body: `{"smiles": "<edited_smiles>"}`
  - response body: `{"edited_solubility": <float>}`
- The example uses `use_mock=True` so it runs offline, but keeps the endpoint-oriented structure.
- Logged metrics in this example are concise by default:
  - `similarity_reward_func`
  - `solubility_modification_func`
  - `num_turns`
