import random
from copy import deepcopy
from difflib import SequenceMatcher
from typing import List

from datasets import Dataset, load_dataset

import verifiers as vf
from verifiers.types import (
    Messages,
    State,
)


def get_sentences(paragraph: str) -> List[str]:
    """Get first sentences of paragraph (up to 5)."""
    sentences = paragraph.replace('." ', '". ').split(". ")
    return sentences[:5]


def filter_sentences(sentences: List[str]) -> bool:
    """Filter paragraphs by number of sentences and sentence length."""
    if len(sentences) < 5:
        return False
    for sentence in sentences:
        if len(sentence) < 10 or len(sentence) > 500:
            return False
    return True


index = [("first", 0), ("second", 1), ("third", 2), ("fourth", 3), ("fifth", 4)]


def get_sentence_questions(x):
    sentences = x["sentences"]
    shuffled_index = deepcopy(index)
    random.shuffle(shuffled_index)
    questions_answers = [
        (
            f"What's the {shuffled_index[0][0]} sentence of the paragraph?",
            f'The {shuffled_index[0][0]} sentence of the paragraph is: "{sentences[shuffled_index[0][1]]}"',
        ),
        (
            f"OK how about the {shuffled_index[1][0]} sentence?",
            f'The {shuffled_index[1][0]} sentence of the paragraph is: "{sentences[shuffled_index[1][1]]}"',
        ),
        (
            f"And the {shuffled_index[2][0]} sentence?",
            f'The {shuffled_index[2][0]} sentence of the paragraph is: "{sentences[shuffled_index[2][1]]}"',
        ),
        (
            f"What's the {shuffled_index[3][0]} sentence?",
            f"The {shuffled_index[3][0]} sentence of the paragraph is: {sentences[shuffled_index[3][1]]}",
        ),
        (
            f"OK last one. What's the {shuffled_index[4][0]} sentence?",
            f"The {shuffled_index[4][0]} sentence of the paragraph is: {sentences[shuffled_index[4][1]]}",
        ),
    ]
    x["task"] = "sentence-repeater"
    x["info"] = {}
    x["info"]["questions"] = [q for q, _ in questions_answers]
    x["info"]["answers"] = [a for _, a in questions_answers]
    paragraph = ". ".join(sentences)
    x["prompt"] = [
        {
            "role": "user",
            "content": f'Answer questions about the following paragraph:\n\n"{paragraph}"\n\n{questions_answers[0][0]}',
        }
    ]
    return x


class SentenceRepeaterEnv(vf.MultiTurnEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @vf.stop
    async def all_questions_answered(self, state: State) -> bool:
        return len(state["trajectory"]) >= len(state["info"]["questions"])

    async def env_response(
        self, messages: Messages, state: State, **kwargs
    ) -> Messages:
        num_turn = len(state["trajectory"])
        return [
            {
                "role": "user",
                "content": state["info"]["questions"][num_turn],
            }
        ]


def load_environment(**kwargs) -> vf.Environment:
    dataset: Dataset = load_dataset("agentlans/wikipedia-paragraphs", split="train")  # type: ignore
    dataset = dataset.map(lambda x: {"sentences": get_sentences(x["text"])})
    dataset = dataset.filter(lambda x: len(x["sentences"]) == 5)
    dataset = dataset.map(get_sentence_questions)
    parser = vf.Parser()

    def compare_answers_reward_func(parser, completion, info, **kwargs) -> float:
        assert isinstance(completion, list)
        model_answers = [
            m["content"] for m in parser.get_assistant_messages(completion)
        ]
        answers = info["answers"]
        reward = 0
        if len(model_answers) != len(answers):
            return 0

        def similarity(a, b):
            return SequenceMatcher(None, a, b).ratio()

        for model_answer, answer in zip(model_answers[: len(answers)], answers):
            reward += similarity(model_answer, answer) / len(answers)
        return reward

    rubric = vf.Rubric(parser=parser, funcs=[compare_answers_reward_func])

    vf_env = SentenceRepeaterEnv(
        dataset=dataset, parser=parser, rubric=rubric, **kwargs
    )

    return vf_env
