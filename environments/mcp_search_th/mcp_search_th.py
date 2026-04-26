import sys
from pathlib import Path

from datasets import Dataset

import verifiers as vf


def load_taskset() -> vf.Taskset:
    parser = vf.XMLParser(fields=["answer"], answer_field="answer")

    def contains_answer(parser, completion, answer) -> float:
        response = (parser.parse_answer(completion) or "").lower()
        return 1.0 if answer.lower() in response else 0.0

    def source() -> Dataset:
        return Dataset.from_list(
            [
                {
                    "question": (
                        "Use the MCP document tools to answer this question: "
                        "what does the taskset/harness refactor compose? "
                        "Put the short answer in <answer> tags."
                    ),
                    "answer": "tasksets with reusable harnesses",
                }
            ]
        )

    server_path = Path(__file__).with_name("mcp_server.py")
    return vf.Taskset(
        source=source,
        rubric=vf.Rubric(funcs=[contains_answer], weights=[1.0], parser=parser),
        tools=[
            vf.MCPTool(
                name="documents",
                command=sys.executable,
                args=[str(server_path)],
                description="Local document search MCP server.",
            )
        ],
        name="mcp-search-th",
    )


def load_harness(
    system_prompt: str | None = (
        "Use the available MCP document tools before answering. "
        "Return the final answer in <answer> tags."
    ),
    max_turns: int = 4,
) -> vf.Harness:
    return vf.Harness(
        system_prompt=system_prompt,
        run=vf.RunConfig(max_turns=max_turns),
    )


def load_environment(
    system_prompt: str | None = (
        "Use the available MCP document tools before answering. "
        "Return the final answer in <answer> tags."
    ),
    max_turns: int = 4,
) -> vf.Environment:
    return vf.Env(
        taskset=load_taskset(),
        harness=load_harness(system_prompt=system_prompt, max_turns=max_turns),
    )
