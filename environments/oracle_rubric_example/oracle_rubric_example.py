import re
from difflib import SequenceMatcher

from datasets import Dataset

import verifiers as vf


class SolubilityPredictClient:
    """Example NN-service client.

    Intended contract (remote service):
    POST http://localhost:0000/predict
    body: {"smiles": "..."}
    response: {"edited_solubility": <float>}

    For this example we keep a local mock fallback so the environment runs
    without an external server.
    """

    def __init__(self, endpoint_url: str = "http://localhost:0000/predict", use_mock: bool = True):
        self.endpoint_url = endpoint_url
        self.use_mock = use_mock

    @staticmethod
    def estimate_solubility(smiles: str) -> float:
        """String-based toy proxy for aqueous solubility in [0, 1]."""
        hetero_count = sum(smiles.count(ch) for ch in ("O", "N", "S", "P"))
        ionic_markers = smiles.count("+") + smiles.count("-")
        aromatic_marker = smiles.count("c")
        carbon_count = smiles.count("C") + smiles.count("c")

        raw_score = (
            0.35
            + 0.09 * hetero_count
            + 0.10 * ionic_markers
            - 0.02 * carbon_count
            - 0.015 * aromatic_marker
        )
        return max(0.0, min(1.0, raw_score))

    async def predict(self, edited_smiles: str) -> dict[str, float]:
        if not edited_smiles:
            return {"valid_predict_call": False}

        # In production, this is where an HTTP call to /predict would happen.
        # Example-only mock mode keeps the environment runnable offline.
        if self.use_mock:
            return {
                "edited_solubility": self.estimate_solubility(edited_smiles),
                "valid_predict_call": True,
            }

        return {"valid_predict_call": False}


def _extract_smiles_candidate(response: str) -> str | None:
    if not response:
        return None

    tokens = re.findall(r"[A-Za-z0-9@+\-\[\]\(\)=#$\\/%\.]+", response)
    candidates = [tok for tok in tokens if any(ch.isalpha() for ch in tok) and len(tok) >= 3]
    if not candidates:
        return None
    return max(candidates, key=len)


def _completion_to_text(completion) -> str:
    """Extract plain text from common completion formats."""
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


async def call_solubility_oracle(
    oracle=None,
    oracle_backend=None,
    oracle_input=None,
    **kwargs,
):
    backend = oracle_backend or oracle
    if backend is None or not isinstance(oracle_input, dict):
        return {"valid_predict_call": False}

    try:
        response_text = str(oracle_input.get("response", ""))
        edited_smiles = _extract_smiles_candidate(response_text)
        if edited_smiles is None:
            return {"valid_predict_call": False}

        return await backend.predict(edited_smiles=edited_smiles) #Output: {"edited_solubility": <float>, "valid_predict_call": True}
    except Exception:
        return {"valid_predict_call": False}


def similarity_reward_func(parser, completion, answer, state, **kwargs):
    """Normal (non-oracle) rubric: reward minimal edits via SMILES similarity."""
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


def solubility_modification_func(property_value, answer, **kwargs):
    # Support both direct scalar outputs and dict-shaped outputs from oracle_fn.
    if isinstance(property_value, dict):
        if not bool(property_value.get("valid_predict_call", True)):
            return 0.0
        property_value = property_value.get("edited_solubility", 0.0)

    try:
        edited_solubility = float(property_value)
    except (TypeError, ValueError):
        return 0.0

    try:
        target_data = answer if isinstance(answer, dict) else {}
        direction = str(target_data.get("direction", "")).lower()
        start_solubility = float(target_data.get("start_solubility", 0.0))
        min_delta = float(target_data.get("min_delta", 0.02))
    except (TypeError, ValueError):
        return 0.0

    if direction not in {"higher", "lower"}:
        return 0.0

    delta = edited_solubility - start_solubility

    desired_delta = delta if direction == "higher" else -delta
    if desired_delta <= 0.0:
        return 0.0
    if desired_delta < min_delta:
        return desired_delta / min_delta
    return min(1.0, desired_delta / 0.25)


def load_environment(
    system_prompt: str
    | None = "You are a medicinal chemistry assistant. Make minimal edits and return only one SMILES string.",
):
    """OracleRubric example for directional solubility editing with minimal changes."""

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
                "task": "oracle-rubric-example",
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
                "task": "oracle-rubric-example",
                "info": {},
            },
            {
                "question": (
                    "Starting SMILES: CC(=O)N\n"
                    "Goal: make it MORE soluble with minimal edits.\n"
                    "Return only the edited SMILES."
                ),
                "answer": {
                    "start_smiles": "CC(=O)N",
                    "direction": "higher",
                    "min_delta": 0.02,
                    "min_similarity": 0.55,
                    "start_solubility": SolubilityPredictClient.estimate_solubility("CC(=O)N"),
                },
                "task": "oracle-rubric-example",
                "info": {},
            },
        ]
    )

    #our mock service client with the example oracle function as fallback
    oracle = SolubilityPredictClient(endpoint_url="http://localhost:0000/predict", use_mock=True)

    shared_input = lambda response, **kwargs: {
        "response": response,
    }

    similarity_rubric = vf.Rubric()
    similarity_rubric.add_reward_func(similarity_reward_func)

    directional_solubility_rubric = vf.OracleRubric(
        oracle=oracle,
        oracle_input_fn=shared_input,
        oracle_fn=call_solubility_oracle,
        score_function=solubility_modification_func,
    )

    rubric = vf.RubricGroup([similarity_rubric, directional_solubility_rubric])

    return vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt=system_prompt,
        parser=similarity_rubric.parser,
        rubric=rubric,
    )
