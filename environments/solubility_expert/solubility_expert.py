import re
import os
import asyncio
import logging
import importlib
from difflib import SequenceMatcher
from typing import Any

from datasets import Dataset

import verifiers as vf


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SolubilityPredictClient:
    """Example NN-service client.

    Intended contract (remote service):
    POST http://localhost:0000/predict
    body: {"smiles": "..."}
    response: {"edited_pka": <float>}

    For this example we keep a local mock fallback so the environment runs
    without an external server, and optionally route to Rowan's API.
    """

    def __init__(
        self,
        endpoint_url: str = "http://localhost:0000/predict",
        use_mock: bool = True,
        use_rowan_api: bool = False,
        rowan_api_key_var: str = "rowan_key",
        rowan_solubility_method: str = "fastsolv",
        rowan_solvents: list[str] | None = None,
        rowan_temperatures: list[float] | None = None,
        rowan_max_credits: int | None = None,
    ):
        self.endpoint_url = endpoint_url
        self.use_mock = use_mock
        self.use_rowan_api = use_rowan_api
        self.rowan_api_key_var = rowan_api_key_var
        self.rowan_solubility_method = rowan_solubility_method
        # Default to exactly one solvent to keep one external request per rollout.
        self.rowan_solvents = rowan_solvents if rowan_solvents is not None else ["CO"]
        self.rowan_temperatures = (
            rowan_temperatures if rowan_temperatures is not None else [293.15]
        )
        self.rowan_max_credits = rowan_max_credits

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

    def _extract_rowan_solubility(self, data: Any) -> float | None:
        if isinstance(data, (int, float)) and not isinstance(data, bool):
            return float(data)
        if isinstance(data, dict):
            for key in (
                "solubility",
                "aqueous_solubility",
                "log_s",
                "logS",
            ):
                if key in data:
                    val = self._extract_rowan_solubility(data[key])
                    if val is not None:
                        return val
            for value in data.values():
                val = self._extract_rowan_solubility(value)
                if val is not None:
                    return val
        if isinstance(data, list):
            for item in data:
                val = self._extract_rowan_solubility(item)
                if val is not None:
                    return val
        return None

    async def _predict_with_rowan(self, edited_smiles: str) -> dict[str, Any]:
        api_key = os.getenv(self.rowan_api_key_var) or os.getenv("ROWAN_API_KEY")
        if not api_key:
            logger.error(
                "Rowan API key missing. Set %s (or ROWAN_API_KEY).",
                self.rowan_api_key_var,
            )
            return {
                "edited_solubility": None,
                "valid_predict_call": False,
                "oracle_error": "missing_api_key",
            }

        def _submit_and_wait() -> dict[str, Any]:
            rowan: Any = importlib.import_module("rowan")

            rowan.api_key = api_key
            logger.info(
                "Rowan API call started via submit_solubility_workflow (method=%s, solvents=%s, temperatures=%s)",
                self.rowan_solubility_method,
                self.rowan_solvents,
                self.rowan_temperatures,
            )
            workflow = rowan.submit_solubility_workflow(
                initial_smiles=edited_smiles,
                solubility_method=self.rowan_solubility_method,
                solvents=self.rowan_solvents,
                temperatures=self.rowan_temperatures,
                name="OracleRubric Solubility Workflow",
                max_credits=self.rowan_max_credits,
            )
            workflow = workflow.wait_for_result().fetch_latest(in_place=True)
            data = getattr(workflow, "data", {}) or {}
            solubility_value = self._extract_rowan_solubility(data)
            logger.info(
                "Rowan API call completed (workflow_uuid=%s, has_solubility=%s)",
                getattr(workflow, "uuid", None),
                solubility_value is not None,
            )
            return {
                "edited_solubility": solubility_value,
                "valid_predict_call": solubility_value is not None,
                "workflow_uuid": getattr(workflow, "uuid", None),
                "workflow_status": getattr(workflow, "status", None),
            }

        try:
            return await asyncio.to_thread(_submit_and_wait)
        except ModuleNotFoundError:
            logger.exception(
                "Missing Rowan SDK. Install with `pip install rowan-python` to use use_rowan_api=True"
            )
            return {
                "edited_solubility": None,
                "valid_predict_call": False,
                "oracle_error": "missing_rowan_sdk",
            }
        except Exception as exc:
            logger.exception("Rowan solubility workflow call failed")
            return {
                "edited_solubility": None,
                "valid_predict_call": False,
                "oracle_error": f"rowan_exception:{type(exc).__name__}:{exc}",
            }

    async def predict(self, edited_smiles: str) -> dict[str, Any]:
        if not edited_smiles:
            return {"valid_predict_call": False}

        if self.use_mock:
            logger.info("Mock oracle call used (no external API)")
            return {
                "edited_solubility": self.estimate_solubility(edited_smiles),
                "valid_predict_call": True,
            }

        if self.use_rowan_api:
            logger.info("Real Rowan API mode enabled")
            return await self._predict_with_rowan(edited_smiles)

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


async def call_solubility_oracle(oracle, response, **kwargs):
    """oracle is the SolubilityPredictClient; response is the parsed completion text."""
    edited_smiles = _extract_smiles_candidate(response)
    if edited_smiles is None:
        logger.debug("No SMILES candidate extracted from response")
        return {"edited_solubility": None}

    try:
        logger.info(
            "Calling backend from call_solubility_oracle (edited_smiles=%s, use_rowan_api=%s, use_mock=%s)",
            edited_smiles,
            getattr(oracle, "use_rowan_api", None),
            getattr(oracle, "use_mock", None),
        )
        result = await oracle.predict(edited_smiles=edited_smiles)
        return result
    except Exception:
        logger.exception("Solubility backend call failed")
        return {
            "edited_solubility": None,
            "valid_predict_call": False,
            "oracle_error": "backend_call_exception",
        }


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


async def solubility_modification_func(
    oracle,
    prompt,
    completion,
    answer,
    state,
    **kwargs,
):
    oracle_output = await oracle(prompt, completion, answer, state)

    try:
        solubility_val = (
            oracle_output.get("edited_solubility")
            if isinstance(oracle_output, dict)
            else oracle_output
        )
        if solubility_val is None:
            logger.debug("edited_solubility missing in oracle output")
            return 0.0
        edited_solubility = float(solubility_val)
        logger.debug("Parsed edited_solubility=%s", edited_solubility)
    except (TypeError, ValueError):
        logger.debug("Could not parse edited_solubility from oracle output")
        return 0.0

    try:
        target_data = answer if isinstance(answer, dict) else {}
        direction = str(target_data.get("direction", "")).lower()
        start_solubility = float(target_data.get("start_solubility", 0.0))
        min_delta = float(target_data.get("min_delta", 0.02))
        logger.debug(
            "Scoring inputs direction=%s start_solubility=%s min_delta=%s",
            direction,
            start_solubility,
            min_delta,
        )
    except (TypeError, ValueError):
        logger.debug("Invalid answer payload for directional scoring")
        return 0.0

    if direction not in {"higher", "lower"}:
        logger.debug("Direction is not supported: %s", direction)
        return 0.0

    delta = edited_solubility - start_solubility

    desired_delta = delta if direction == "higher" else -delta
    logger.debug("Computed delta=%s desired_delta=%s", delta, desired_delta)
    if desired_delta <= 0.0:
        logger.info("Directional objective not met")
        return 0.0
    if desired_delta < min_delta:
        score = desired_delta / min_delta
        logger.info("Directional objective partially met with score=%s", score)
        return score
    score = min(1.0, desired_delta / 0.25)
    logger.info("Directional objective met with score=%s", score)
    return score


class OracleRubricExampleEnv(vf.SingleTurnEnv):
    """SingleTurnEnv wrapper that propagates runtime kwargs into oracle client."""

    def __init__(self, oracle_client: SolubilityPredictClient, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.oracle_client = oracle_client

    def set_use_rowan_api(self, value: bool) -> None:
        self.oracle_client.use_rowan_api = bool(value)
        self.oracle_client.use_mock = not bool(value)
        logger.info(
            "OracleRubricExampleEnv updated use_rowan_api=%s use_mock=%s",
            self.oracle_client.use_rowan_api,
            self.oracle_client.use_mock,
        )

    def set_rowan_api_key_var(self, value: str) -> None:
        self.oracle_client.rowan_api_key_var = str(value)
        logger.info("OracleRubricExampleEnv updated rowan_api_key_var=%s", value)

    def set_rowan_solubility_method(self, value: str) -> None:
        self.oracle_client.rowan_solubility_method = str(value)
        logger.info(
            "OracleRubricExampleEnv updated rowan_solubility_method=%s",
            value,
        )

    def set_rowan_solvents(self, value: list[str] | None) -> None:
        normalized = value if value is not None else ["CO"]
        self.oracle_client.rowan_solvents = normalized
        logger.info("OracleRubricExampleEnv updated rowan_solvents=%s", normalized)

    def set_rowan_temperatures(self, value: list[float] | None) -> None:
        normalized = value if value is not None else [293.15]
        self.oracle_client.rowan_temperatures = normalized
        logger.info(
            "OracleRubricExampleEnv updated rowan_temperatures=%s",
            normalized,
        )

    def set_rowan_max_credits(self, value: int | None) -> None:
        self.oracle_client.rowan_max_credits = None if value is None else int(value)
        logger.info(
            "OracleRubricExampleEnv updated rowan_max_credits=%s",
            self.oracle_client.rowan_max_credits,
        )


def load_environment(
    system_prompt: str
    | None = "You are a Solubility Expert. Make minimal edits and return only one SMILES string.",
    use_rowan_api: bool = False,
    rowan_api_key_var: str = "rowan_key",
    rowan_solubility_method: str = "fastsolv",
    rowan_solvents: list[str] | None = None,
    rowan_temperatures: list[float] | None = None,
    rowan_max_credits: int | None = None,
):
    """Solubility Expert environment for directional solubility editing.

    Set use_rowan_api=True to use Rowan's submit_solubility_workflow backend.
    API key is read from rowan_api_key_var (default: rowan_key).
    """

    if use_rowan_api:
        vf.ensure_keys([rowan_api_key_var])

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
                    "start_solubility": 0.610785,
                },
                "task": "solubility_expert",
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
                    "start_solubility": 0.699303,
                },
                "task": "solubility_expert",
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
                    "start_solubility": -0.422196,
                },
                "task": "solubility_expert",
                "info": {},
            },
        ]
    )

    #our mock service client with the example oracle function as fallback
    oracle = SolubilityPredictClient(
        endpoint_url="http://localhost:0000/predict",
        use_mock=not use_rowan_api,
        use_rowan_api=use_rowan_api,
        rowan_api_key_var=rowan_api_key_var,
        rowan_solubility_method=rowan_solubility_method,
        rowan_solvents=rowan_solvents,
        rowan_temperatures=rowan_temperatures,
        rowan_max_credits=rowan_max_credits,
    )

    similarity_rubric = vf.Rubric()
    similarity_rubric.add_reward_func(similarity_reward_func)

    directional_solubility_rubric = vf.OracleRubric(
        oracle=oracle,
        oracle_fn=call_solubility_oracle,
    )
    directional_solubility_rubric.add_reward_func(solubility_modification_func)

    rubric = vf.RubricGroup([similarity_rubric, directional_solubility_rubric])

    return OracleRubricExampleEnv(
        oracle_client=oracle,
        dataset=dataset,
        system_prompt=system_prompt,
        parser=similarity_rubric.parser,
        rubric=rubric,
    )
