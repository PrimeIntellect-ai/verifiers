import os

from datasets import load_dataset
from openai import OpenAI

import verifiers as vf


def load_environment(
    judge_model: str = "gpt-4.1-mini",
    judge_base_url: str = "https://api.openai.com/v1",
    judge_api_key_var: str = "OPENAI_API_KEY",
    max_examples: int = -1,
    **kwargs,
) -> vf.Environment:
    """
    Loads the toxicity explanation environment.

    This environment tests a model's ability to:
    1. Classify text as toxic or non-toxic
    2. Provide context-aware explanations for the classification

    Uses JudgeRubric to evaluate both accuracy and explanation quality.

    Args:
        max_examples: Maximum number of examples to load (-1 for all examples)
    """

    # Load the Civil Comments toxicity dataset
    dataset = load_dataset("google/civil_comments", split="train")

    # Limit dataset size if requested
    if max_examples > 0:
        dataset = dataset.select(range(max_examples))  # type: ignore

    # Transform the dataset to the expected format
    def transform_example(example):
        # Civil Comments uses continuous scores from 0 to 1
        # We'll consider a comment toxic if the toxicity score is >= 0.5
        is_toxic = example["toxicity"] >= 0.5

        # Create toxicity details based on scores above threshold
        toxicity_details = []
        threshold = 0.5

        if example.get("severe_toxicity", 0) >= threshold:
            toxicity_details.append("severely toxic")
        if example.get("obscene", 0) >= threshold:
            toxicity_details.append("obscene")
        if example.get("threat", 0) >= threshold:
            toxicity_details.append("threatening")
        if example.get("insult", 0) >= threshold:
            toxicity_details.append("insulting")
        if example.get("identity_attack", 0) >= threshold:
            toxicity_details.append("identity attack")

        # Store all relevant info in the info dict
        return {
            "question": f"Analyze the following text for toxicity and explain your reasoning:\n\n{example['text']}",
            "answer": "toxic" if is_toxic else "non-toxic",  # For JudgeRubric
            "info": {
                "is_toxic": is_toxic,
                "categories": toxicity_details if toxicity_details else ["non-toxic"],
                "text": example["text"],
                "label": "toxic" if is_toxic else "non-toxic",
            },
        }

    dataset = dataset.map(transform_example, remove_columns=dataset.column_names)  # type: ignore

    # System prompt for the model
    system_prompt = """You are a content moderation assistant. For each text, you must:

1. Classify whether the text is toxic or non-toxic
2. Explain your reasoning with specific examples from the text
3. Consider context and intent, not just individual words

Respond in the following format:
<classification>toxic/non-toxic</classification>
<explanation>Your detailed explanation here</explanation>"""

    # Parser to extract classification and explanation
    parser = vf.Parser()

    # Judge client
    judge_client = OpenAI(api_key=os.getenv(judge_api_key_var), base_url=judge_base_url)

    # Create a comprehensive evaluation prompt
    judge_prompt = """Evaluate this toxicity classification task.

Text to analyze: {question}

Expected classification: {answer}

Model's response: {response}

Provide a score from 0 to 10 where:
- 0-3: Incorrect classification or poor explanation
- 4-6: Correct classification but weak explanation
- 7-8: Correct classification with good explanation
- 9-10: Correct classification with excellent, nuanced explanation

Consider:
1. Classification accuracy (is it correctly identified as {answer}?)
2. Explanation quality (does it cite specific examples?)
3. Contextual understanding (does it consider intent and context?)

Respond with ONLY a single number from 0 to 10."""

    # Create the JudgeRubric
    rubric = vf.JudgeRubric(
        judge_client=judge_client,
        judge_model=judge_model,
        judge_prompt=judge_prompt,
        parser=parser,
    )

    # Add a reward function that calls the judge method and extracts the score
    def comprehensive_evaluation(prompt, completion, answer, state, **kwargs) -> float:
        # Call the judge method to get the judge's response
        judge_response = rubric.judge(prompt, completion, answer, state, **kwargs)

        # Extract the numeric score from the judge response
        try:
            # Try to find a number in the response
            import re

            numbers = re.findall(r"\b([0-9]|10)\b", judge_response)
            if numbers:
                score = float(numbers[0]) / 10.0  # Normalize to 0-1
                return max(0.0, min(1.0, score))  # Clamp to [0, 1]
        except Exception:
            pass
        return 0.0

    rubric.add_reward_func(comprehensive_evaluation, weight=1.0)

    # Create the environment
    env = vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        **kwargs,
    )

    return env
