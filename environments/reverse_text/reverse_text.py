from datasets import load_dataset

import verifiers as vf


def load_environment(num_train_examples=2000, num_eval_examples=200, **kwargs):
    dataset = load_dataset("agentlans/wikipedia-paragraphs", split="train").map(
        lambda x: {"question": x["text"], "answer": x["text"][::-1]}
    )
    train_dataset = dataset.select(range(num_train_examples))  # type: ignore
    eval_dataset = dataset.select(  # type: ignore
        range(num_train_examples, num_train_examples + num_eval_examples)
    )

    parser = vf.XMLParser(["think", "answer"], answer_field="answer")
    system_prompt = f"""Reverse the given text.

    Respond in the following format:
    {parser.get_format_str()}"""

    def lcs_reward_func(completion, answer, **kwargs) -> float:
        """
        LCS ratio of the reversed prompt and the parsed completion.
        """

        def lcs_ratio(x: str, y: str) -> float:
            """
            Return the longest common subsequence ratio of x and y.
            """
            from difflib import SequenceMatcher

            return SequenceMatcher(None, x, y).ratio()

        response = parser.parse_answer(completion) or ""
        return lcs_ratio(response, answer)

    rubric = vf.Rubric(
        funcs=[
            lcs_reward_func,
            parser.get_format_reward_func(),
        ],
        weights=[1.0, 0.2],
    )

    vf_env = vf.SingleTurnEnv(
        dataset=train_dataset,  # type: ignore
        eval_dataset=eval_dataset,  # type: ignore
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
    )
    return vf_env
