from datasets import Dataset

import verifiers as vf


class SimilarityAPIServer:
    """Example API-style oracle that returns overlap similarity in [0, 1]."""

    async def score(self, response: str, reference: str) -> dict[str, float]:
        response_tokens = set(response.lower().split())
        reference_tokens = set(reference.lower().split())
        if not reference_tokens:
            return {"similarity": 0.0}
        overlap = len(response_tokens & reference_tokens)
        return {"similarity": overlap / len(reference_tokens)}


def load_environment(
    system_prompt: str
    | None = "Answer with one short sentence that addresses the question directly.",
):
    """Example environment showing how to use OracleRubric with an API server."""

    dataset = Dataset.from_list(
        [
            {
                "question": "Describe why regular exercise is useful.",
                "answer": {
                    "reference": "regular exercise improves health",
                    "target": 1.0,
                    "threshold": 0.34,
                },
                "task": "oracle-rubric-example",
                "info": {},
            },
            {
                "question": "Give one benefit of drinking water.",
                "answer": {
                    "reference": "drinking water supports hydration",
                    "target": 1.0,
                    "threshold": 0.34,
                },
                "task": "oracle-rubric-example",
                "info": {},
            },
            {
                "question": "Why does sleep matter?",
                "answer": {
                    "reference": "sleep helps recovery and focus",
                    "target": 1.0,
                    "threshold": 0.34,
                },
                "task": "oracle-rubric-example",
                "info": {},
            },
        ]
    )

    oracle = SimilarityAPIServer()

    rubric = vf.OracleRubric(
        oracle=oracle,
        oracle_input_fn=lambda response, answer, **kwargs: {
            "response": response,
            "reference": answer["reference"],
        },
        oracle_fn=lambda oracle, oracle_input, **kwargs: oracle.score(
            response=oracle_input["response"],
            reference=oracle_input["reference"],
        ),
        property_extractor=lambda oracle_output, **kwargs: oracle_output["similarity"],
    )

    return vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt=system_prompt,
        parser=rubric.parser,
        rubric=rubric,
    )
