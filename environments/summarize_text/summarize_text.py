from datasets import load_dataset

import verifiers as vf


def load_environment(**kwargs) -> vf.Environment:
    dataset = load_dataset("agentlans/wikipedia-paragraphs", split="train")
    dataset = dataset.map(lambda x: {"question": x["text"], "answer": x["text"]})

    parser = vf.XMLParser(["think", "answer"], answer_field="answer")
    system_prompt = f"""Respond in the following format:
    {parser.get_format_str()}

    Summarize the given text in 3 sentences."""

    def sentence_reward_func(completion, **kwargs) -> float:
        """
        Count the number of sentences in the completion.
        """
        response = parser.parse_answer(completion) or ""
        return 1.0 if len(response.split(".")) == 3 else 0.0

    def lcs_reward_func(completion, answer, **kwargs) -> float:
        """
        LCS ratio of the prompt and the parsed completion.
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
            sentence_reward_func,
            lcs_reward_func,
            parser.get_format_reward_func(),
        ],
        weights=[1.0, 0.2, 0.2],
    )

    vf_env = vf.SingleTurnEnv(
        eval_dataset=dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        max_concurrent=10,
    )
    return vf_env
