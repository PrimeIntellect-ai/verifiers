# Solubility Expert

<a href="https://github.com/PrimeIntellect-ai/verifiers/tree/main/environments/solubility_expert">
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="Source Code">
</a>

### Overview
- **Environment ID**: `solubility_expert`
- **Short description**: Solubility Expert environment to test experimental `OracleRubric` on a SMILES editing task with a mock oracle or a real Rowan solubility workflow backend.
- **Tags**: oracle, rubric, single-turn, chemistry, smiles, api

### Task
- **Type**: single-turn
- **Rubric overview**:
  - Parses assistant output and extracts a SMILES candidate.
  - Rubric 1 (normal `Rubric`): similarity reward from direct string comparison (no oracle call).
  - Rubric 2 (`OracleRubric`): `oracle` is the backend client (`SolubilityPredictClient`), `oracle_fn` calls the backend.
  - Scoring registered with `add_reward_func(...)` — same pattern as `JudgeRubric`.
  - `solubility_modification_func` receives `oracle` injected and calls `await oracle(prompt, completion, answer, state)` directly.
  - Oracle backend can run in mock mode or submit Rowan `submit_solubility_workflow(...)` jobs.
  - Final reward is the sum of both rubric scores.

### Quickstart
```bash
prime eval run solubility_expert
```

### Real Rowan Backend
Install Rowan's SDK:

```bash
pip install rowan-python
```

Set your API key secret (default var used by this env is `rowan_key`):

```bash
export rowan_key="<your_rowan_api_key>"
```

Run with Rowan backend enabled:

```bash
prime eval run solubility_expert --env-args '{"use_rowan_api": true}'
```

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `system_prompt` | str \| None | "You are a Solubility Expert. Make minimal edits and return only one SMILES string." | Optional system prompt |
| `use_rowan_api` | bool | `false` | If true, uses Rowan `submit_solubility_workflow` instead of mock predictions. |
| `rowan_api_key_var` | str | `"rowan_key"` | Environment variable name that stores Rowan API key. |
| `rowan_solubility_method` | str | `"fastsolv"` | Solubility model passed to Rowan `submit_solubility_workflow`. |
| `rowan_solvents` | list[str] \| None | `["CO"]` | Solvent list passed to Rowan `submit_solubility_workflow`. Default is one solvent (one request). |
| `rowan_temperatures` | list[float] \| None | `[293.15]` | Temperatures passed to Rowan `submit_solubility_workflow`. |
| `rowan_max_credits` | int \| None | `null` | Optional Rowan max credits per workflow. |

### Notes
- By default this environment uses mock predictions and runs fully offline.
- When `use_rowan_api=true`, each oracle call submits Rowan `submit_solubility_workflow(...)` and waits for completion.
- Default Rowan config uses exactly one solvent (`["CO"]`) to avoid multi-solvent fanout.
- You can override solvent/temperature at runtime, for example:
  - `prime eval run solubility_expert --env-args '{"use_rowan_api": true, "rowan_solvents": ["CS(=O)C"], "rowan_temperatures": [293.15]}'`
- API key validation is enforced in `load_environment` via `vf.ensure_keys([rowan_api_key_var])`.
- Logged metrics in this example are concise by default:
  - `similarity_reward_func`
  - `solubility_modification_func`
  - `num_turns`
