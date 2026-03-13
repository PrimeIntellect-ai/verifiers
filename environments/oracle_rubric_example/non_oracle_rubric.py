import re
from difflib import SequenceMatcher

from datasets import Dataset

import verifiers as vf

from oracle_rubric_example import SolubilityPredictClient


def _extract_smiles_candidate(response: str) -> str | None:
    if not response:
        return None

    tokens = re.findall(r"[A-Za-z0-9@+\-\[\]\(\)=#$\\/%\.]+", response)
    candidates = [tok for tok in tokens if any(ch.isalpha() for ch in tok) and len(tok) >= 3]
    if not candidates:
        return None
    return max(candidates, key=len)


def _completion_to_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        content = completion.get("content", "")
        if isinstance(content, str):
            return content
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            content = last.get("content", "")
            if isinstance(content, str):
                return content
    return str(completion)


def similarity_reward_func(parser, completion, answer, state, **kwargs):
    start_smiles = str(answer.get("start_smiles", "")) if isinstance(answer, dict) else ""
    if not start_smiles:
        return 0.0

    completion_text = _completion_to_text(completion)
    edited_smiles = _extract_smiles_candidate(completion_text)
    if not edited_smiles:
        return 0.0

    similarity = SequenceMatcher(None, start_smiles, edited_smiles).ratio()
    min_similarity = (
        float(answer.get("min_similarity", 0.55))
        if isinstance(answer, dict)
        else 0.55
    )
    if similarity < min_similarity:
        return 0.0
    return max(0.0, min(1.0, similarity))


async def solubility_modification_func(parser, completion, answer, state, **kwargs):
    """Same scoring behavior as the OracleRubric example, implemented directly in Rubric.

    This function manually does the pipeline that OracleRubric abstracts:
    1) parse model output
    2) build oracle input
    3) call oracle backend
    4) extract property from oracle response if dict
    5) compare against answer-derived task targets/thresholds
    """
    completion_text = _completion_to_text(completion)
    edited_smiles = _extract_smiles_candidate(completion_text)
    if edited_smiles is None:
        return 0.0

    cache_key = repr((edited_smiles, answer))
    cache = state.get("non_oracle_cache") if state else None
    cached = cache.get(cache_key) if isinstance(cache, dict) else None

    if cached is not None:
        oracle_output = cached
    else:
        oracle = state.get("solubility_oracle") if state else None
        if oracle is None:
            oracle = SolubilityPredictClient(
                endpoint_url="http://localhost:0000/predict",
                use_mock=True,
            )
            if state is not None:
                state["solubility_oracle"] = oracle
        try:
            oracle_output = await oracle.predict(edited_smiles=edited_smiles)
        except Exception:
            return 0.0

        if state is not None:
            if not isinstance(cache, dict):
                cache = {}
            cache[cache_key] = oracle_output
            state["non_oracle_cache"] = cache

    # Optional extractor behavior: only extract if oracle output is dict.
    if isinstance(oracle_output, dict):
        if not bool(oracle_output.get("valid_predict_call", True)):
            return 0.0
        property_value = oracle_output.get("edited_solubility", 0.0)
    else:
        property_value = oracle_output

    try:
        edited_solubility = float(property_value)
    except (TypeError, ValueError):
        return 0.0

    target_data = answer if isinstance(answer, dict) else {}
    direction = str(target_data.get("direction", "")).lower()
    if direction not in {"higher", "lower"}:
        return 0.0

    try:
        start_solubility = float(target_data.get("start_solubility", 0.0))
        min_delta = float(target_data.get("min_delta", 0.02))
    except (TypeError, ValueError):
        return 0.0

    delta = edited_solubility - start_solubility
    desired_delta = delta if direction == "higher" else -delta

    if desired_delta <= 0.0:
        return 0.0
    if desired_delta < min_delta:
        score = desired_delta / min_delta
    else:
        score = min(1.0, desired_delta / 0.25)

    if state is not None:
        state["oracle_response"] = oracle_output
        state["oracle_property_value"] = property_value

    return float(score)


COMPARISON = {
    "oracle_rubric": {
        "what_it_gives": [
            "Built-in pipeline for oracle input, backend call, optional extraction, and scoring",
            "Built-in measurement cache and state wiring",
            "Cleaner environment files via composition",
        ],
        "best_for": [
            "Multiple tasks using the same oracle pattern",
            "Consistent behavior across environments",
            "Lower maintenance as custom logic grows",
        ],
    },
    "plain_rubric": {
        "what_it_gives": [
            "Maximum flexibility in one function",
            "No OracleRubric dependency",
            "Can be quicker for one-off experiments",
        ],
        "tradeoffs": [
            "More boilerplate per environment",
            "Manual cache/state/error handling",
            "Higher risk of divergence across tasks",
        ],
    },
}


def load_environment(
    system_prompt: str
    | None = "You are a medicinal chemistry assistant. Make minimal edits and return only one SMILES string.",
):
    dataset = Dataset.from_list(
        [
            {
                "question": (
                    "Starting SMILES: CCO\n"
                    "Goal: make it LESS soluble with minimal edits.\n"
                    "Return only the edited SMILES."
                ),
                "answer": {
                    "start_smiles": "CCO",
                    "direction": "lower",
                    "min_delta": 0.02,
                    "min_similarity": 0.55,
                    "start_solubility": SolubilityPredictClient.estimate_solubility("CCO"),
                },
                "task": "non-oracle-rubric-example",
                "info": {},
            },
            {
                "question": (
                    "Starting SMILES: c1ccccc1\n"
                    "Goal: make it MORE soluble with minimal edits.\n"
                    "Return only the edited SMILES."
                ),
                "answer": {
                    "start_smiles": "c1ccccc1",
                    "direction": "higher",
                    "min_delta": 0.03,
                    "min_similarity": 0.55,
                    "start_solubility": SolubilityPredictClient.estimate_solubility("c1ccccc1"),
                },
                "task": "non-oracle-rubric-example",
                "info": {},
            },
        ]
    )

    rubric = vf.Rubric()
    rubric.add_reward_func(similarity_reward_func)
    rubric.add_reward_func(solubility_modification_func)

    env = vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt=system_prompt,
        parser=rubric.parser,
        rubric=rubric,
    )
    return env
