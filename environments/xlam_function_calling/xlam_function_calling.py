import json

from datasets import load_dataset

import verifiers as vf


def load_environment() -> vf.Environment:
    def format_sys_prompt(tools: list[dict]) -> str:
        tool_str = json.dumps(tools, indent=2)
        return f"""You are a helpful assistant that can use tools to answer questions and perform tasks. You have the following tools:
{tool_str}

Think step-by-step inside <think>...</think> tags, then call one or more tools inside <tool>...</tool> tags \
as a JSON array with 'name' and 'arguments' keys for each tool call."""

    def process_example(x):
        prompt = [
            {"role": "system", "content": format_sys_prompt(json.loads(x["tools"]))},
            {"role": "user", "content": x["query"]},
        ]
        return {
            "prompt": prompt,
            "answer": x["answers"],
            "task": "xlam-function-calling",
        }

    dataset = load_dataset("Salesforce/xlam-function-calling-60k", split="train")
    dataset = dataset.map(process_example, remove_columns=dataset.column_names)  # type: ignore

    parser = vf.XMLParser(fields=["think", "tool"], answer_field="tool")

    def check_tools_reward_func(completion, answer) -> float:
        try:
            called_tools = json.loads(parser.parse_answer(completion) or "[]")
            target_tools = json.loads(answer)
            for called_tool in called_tools:
                if called_tool not in target_tools:
                    return 0
            for target_tool in target_tools:
                if target_tool not in called_tools:
                    return 0
            return 1
        except Exception:
            return 0

    rubric = vf.Rubric(funcs=[check_tools_reward_func])

    vf_env = vf.SingleTurnEnv(
        dataset=dataset,
        parser=parser,
        rubric=rubric,
    )
    return vf_env
