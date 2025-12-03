from datasets import Dataset, load_dataset
import verifiers as vf
import re
import os
import pandas as pd
import textwrap
from random import random
from typing import List

"""
inference:
CUDA_VISIBLE_DEVICES=0 uv run vf-vllm --model willcb/Qwen3-0.6B

training:
CUDA_VISIBLE_DEVICES=1 uv run accelerate launch --num-processes 1 --config-file configs/deepspeed/zero3.yaml verifiers/examples/j1_reward.py
"""


parser = vf.XMLParser(['think', 'answer'], answer_field='answer')

def judge_system_prompt() -> str:
    return textwrap.dedent(
        """
    You are an expert XML wrangler. You must respond in the following format, regardless of the input:
    
    <specific_criteria>
    ...
    </specific_criteria>
    <analysis>
    ...
    </analysis>
    <scores>
    \\boxed{{..., ...}}
    </scores>

    Please only respond in English.
    """
    ).strip()


judge_prompt_template = textwrap.dedent(
    """
    You are a skilled little expert at scoring responses. You should evaluate given responses based on the given judging criteria.
    Given the context of the conversation (the last round is the User's query) and multiple responses from the Assistant, you need to refer to the [General Evaluation Criteria] to score the responses. Based on the general evaluation criteria, state potential other specific criteria to the query, the weights of different criteria, and then provide an overall comprehensive score upon them.
    Each score is an integer between 1 and 10, with a higher score indicating that the response meets the relevant criteria more closely. For example, a score of 1 means the response does not meet the criteria at all, a score of 6 means the response meets only some parts, and a score of 10 means the response perfectly meets the evaluation criteria.
    Before scoring, please analyze step by step. Your scoring needs to be as strict as possible.

    #### Evaluation Criteria ####
    1. Instruction Adherence:
    - Fully Adhered (9-10 points): The response fully complies with all instructions and requirements of the question.
    - Partially Adhered (6-8 points): The response meets most of the instructions but has some omissions or misunderstandings.
    - Basically Adhered (3-5 points): The response meets some instructions, but the main requirements are not fulfilled.
    - Not Adhered (1-2 points): The response does not meet any instructions.
    Example: If the question requires three examples and the response provides only one, it falls under "Partially Adhered."
    2. Usefulness:
    - Highly Useful (9-10 points): The response provides comprehensive and accurate information, fully addressing the issue.
    - Useful but Incomplete (6-8 points): The response provides some useful information, but lacks details or accuracy.
    - Limited Usefulness (3-5 points): The response offers little useful information, with most content being irrelevant or incorrect.
    - Useless or Incorrect (1-2 points): The response is completely irrelevant or incorrect.
    Example: If there are factual errors in the response but the overall direction is correct, it falls under "Useful but Incomplete."
    3. Level of Detail:
    - Very Detailed (9-10 points): The response includes ample details covering all aspects of the issue.
    - Detailed but Slightly Lacking (6-8 points): The response is fairly detailed but misses some important details.
    - Basically Detailed (3-5 points): The response provides some details but is not thorough enough overall.
    - Not Detailed (1-2 points): The response is very brief and lacks necessary details.
    Example: If the response provides only a simple conclusion without an explanation, it falls under "Not Detailed."
    4. Relevance:
    - Highly Relevant (9-10 points): The response is highly relevant to the question, with information closely aligned with the topic.
    - Generally Relevant (6-8 points): The response is generally relevant but includes some unnecessary information.
    - Partially Relevant (3-5 points): The response has a lot of content that deviates from the topic.
    - Not Relevant (1-2 points): The response is completely irrelevant.
    Example: If the response strays from the topic but still provides some relevant information, it falls under "Partially Relevant."

    #### Conversation Context ####
    {conversation_context_query}
    #### Responses to be Scored ####
    [The Begin of Response A]
    {response_a}
    [The End of Response A]
    [The Begin of Response B]
    {response_b}
    [The End of Response B]
    #### Output Format Requirements ####

    Output with three lines
    <specific_criteria>
    [Other potential criteria specific to the query and the context, and the weights of each criteria.]
    </specific_criteria>
    <analysis>
    [Compare different responses based on given Criteria.]
    </analysis>
    <scores>
    [The overall comprehensive score of all responses in order, separate by comma in the boxed, e.g., \\boxed{{x, x}} if there exists 2 responses.]
    </scores>
    """
).strip()


def judge_prompt_format(
    conversation_context_query: str, response_a: str, response_b: str
) -> str:
    """
    See page 40 of https://arxiv.org/abs/2504.02495
    """

    return judge_prompt_template.format(
        conversation_context_query=conversation_context_query,
        response_a=response_a,
        response_b=response_b,
    )


def format_inputs(
    row: dict[str, str], use_system_prompt: bool = True
) -> list[dict[str, str]]:
    messages = []
    if use_system_prompt:
        messages.append({"role": "system", "content": judge_system_prompt()})
    input = row["sky_input"]
    if row["chosen_positions"] == "a":
        response_a = row["chosen"]
        response_b = row["rejected"]
    else:
        response_a = row["rejected"]
        response_b = row["chosen"]
    judge_prompt = judge_prompt_format(
        conversation_context_query=input,
        response_a=response_a,
        response_b=response_b,
    )
    messages.append({"role": "user", "content": judge_prompt})
    return messages


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the dataframe into TRL-friendly format.
    """
    df["sky_input"] = df["chosen"].apply(
        lambda x: x[0]["content"]
    )
    df["sky_chosen"] = df["chosen"].apply(
        lambda x: x[1]["content"]
    )
    df["sky_rejected"] = df["rejected"].apply(
        lambda x: x[1]["content"]
    )
    df["chosen_positions"] = [
        "a" if random() < 0.5 else "b"
        for _ in range(len(df))
    ]
    return df


def get_skywork_dataset(
    file_name: str,
    split: str = "train",
    ds_name: str = "Skywork/Skywork-Reward-Preference-80K-v0.2",
) -> Dataset:
    """
    See https://huggingface.co/datasets/Skywork/Skywork-Reward-Preference-80K-v0.2
    """

    assert split in [
        "train",
        "test",
    ], f"Invalid `split` argument: {split}. Expected 'train' or 'test'."

    assert (
        ".csv" in file_name
    ), f"Invalid `file_name` argument: {file_name}. Expected a .csv file."

    if os.path.exists(file_name):
        assert (
            split in file_name
        ), f"Invalid split `{split}` for file_name: `{file_name}`"
        df = pd.read_csv(file_name)

    else:
        ds = load_dataset(ds_name)
        ds = ds["train"].train_test_split(test_size=0.2)
        train_df = process_df(ds["train"].to_pandas())
        test_df = process_df(ds["test"].to_pandas())
        train_df.to_csv(file_name, index=False)
        test_df.to_csv(file_name, index=False)

        if split == "train":
            df = train_df
        elif split == "test":
            df = test_df

    df["prompt"] = df.apply(
        lambda row: format_inputs(row, use_system_prompt=True),
        axis=1,
    )
    df = df.drop(
        columns=[
            "source",
            "chosen",
            "rejected",
        ]
    )
    return Dataset.from_pandas(df)

class FormatError(Exception):
    pass


def extract_scores(raw_response: str) -> List[float]:
    """
    Extract Judge scores from the raw response.
    Expects the following format:
        ---
        <specific_criteria>...</specific_criteria>
        <analysis>...</analysis>
        <scores>\boxed{x, y}</scores>
        ---
    """
    match = re.search(r"<scores>(.*?)</scores>", raw_response, re.DOTALL)
    if not match:
        raise FormatError("No Judge scores found in response")

    boxed_match = re.search(r"\\boxed{([\d.]+),\s*([\d.]+)}", match.group(1))
    if not boxed_match:
        raise FormatError("No boxed scores found in scores tag")

    try:
        return [float(boxed_match.group(1)), float(boxed_match.group(2))]
    except ValueError:
        raise FormatError("Invalid score format in boxed response")


def format_reward_func(completions: List[dict[str, str]], **kwargs) -> List[float]:
    """
    Judge format reward for the n=2 (pairwise preference) case

    > Range: [0, 0.2]
    > Boxed contributes 1/4 of the total score
    > Each of the 6 tags contributes 1/8 of the total score
    """
    scores = []
    for completion in completions:
        boxed_fmted = True
        raw_response = completion[0]["content"]
        try:
            extract_scores(raw_response)
        except FormatError:
            boxed_fmted = False
        finally:
            required_tags = [
                "<specific_criteria>",
                "</specific_criteria>",
                "<analysis>",
                "</analysis>",
                "<scores>",
                "</scores>",
            ]
            denominator = len(required_tags) + 2
            total_score = 0.0
            for tag in required_tags:
                if tag in raw_response:
                    total_score += 1.0 / denominator

            if boxed_fmted:
                total_score += 2.0 / denominator

            scores.append(total_score / 5)

    return scores


def argmax_reward_func(
    # completion: str, answer: List[str], **kwargs
    completion: str, answer: str, **kwargs
) -> float:
    """
    For the n=2 (pairwise preference) case, the `chosen` response should score higher than the `rejected` response.

    > Range: [0, 1]
    > completion: rollout
    > answers: list of positions (A or B) of the `chosen` response; corresponds to `chosen_positions` in original interface
    """
    # breakpoint()
    # Hardcoded config values
    
    raw_response = completion[0]["content"]
    try:
        extracted_score_box = extract_scores(raw_response)
    except FormatError:
        return 0.0
    else:
        position_to_score = {
            "a": lambda scores: scores[0] > scores[1],
            "b": lambda scores: scores[1] > scores[0],
        }
        if answer not in position_to_score:
            raise ValueError(f"Invalid chosen position: {answer}")
        return 1.0 if position_to_score[answer](extracted_score_box) else 0.0


model_name = 'willcb/Qwen3-0.6B'
# model_name = 'willcb/Qwen3-1.7B'

dataset = get_skywork_dataset('skywork_train.csv', split='train')
# breakpoint()
dataset = dataset.map(lambda x: {'question': x['prompt'], 'answer': x['chosen_positions']})
dataset = dataset.shuffle(seed=42)
# breakpoint()

# evaluate on the first 32 examples, train on the rest
eval_dataset = dataset.select(range(len(dataset)-32, len(dataset))) # type: ignore
train_dataset = dataset.select(range(0, len(dataset)-32)) # type: ignore


rubric = vf.Rubric(funcs=[
	argmax_reward_func,
], weights=[1.0])

vf_env = vf.SingleTurnEnv(
    dataset=train_dataset, # type: ignore
    eval_dataset=eval_dataset, # type: ignore
    system_prompt=judge_system_prompt(),
    parser=parser,
    rubric=rubric
)
run_name = 'j1_reward_0.6b'
args = vf.grpo_defaults(run_name=run_name)
args.num_iterations = 2
# args.num_iterations = 1
# args.per_device_train_batch_size = 10
args.per_device_train_batch_size = 4
# args.per_device_train_batch_size = 2
# args.num_generations = 10
# args.num_generations = 2
args.num_generations = 8
# args.num_generations = 4
args.max_prompt_length = 4096
# args.max_prompt_length = 2048
# args.gradient_accumulation_steps = 4
# args.gradient_accumulation_steps = 16
args.gradient_accumulation_steps = 64
args.eval_strategy = "steps"
# args.eval_steps = 10
# args.eval_steps = 2
args.eval_steps = 10
# args.max_steps = 100
# args.max_steps = 10
args.max_steps = -1
args.num_train_epochs = 1
# args.num_train_epochs = 8
# args.log_completions = False

model, tokenizer = vf.get_model_and_tokenizer(model_name)
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=args
)
trainer.train()