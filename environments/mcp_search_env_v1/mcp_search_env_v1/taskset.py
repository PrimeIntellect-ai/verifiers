from collections.abc import Iterable

import verifiers.v1 as vf

from .servers.toolset import SearchToolsetConfig

SYSTEM_PROMPT = "Use the available MCP tools to answer the question."

DEFAULT_EXAMPLES = [
    {
        "query": "ceramic battery recycling",
        "question": "Use the MCP tools to find the record about ceramic battery recycling. What is the record title?",
        "answer": "Kiln Battery Loop",
    },
    {
        "query": "ocean drone algae bloom",
        "question": "Use the MCP tools to find the record about ocean drones and algae blooms. What is the record title?",
        "answer": "Tide Scout",
    },
    {
        "query": "library robot sorting",
        "question": "Use the MCP tools to find the record about library robot sorting. What is the record title?",
        "answer": "Stacks Navigator",
    },
    {
        "query": "green roof insulation",
        "question": "Use the MCP tools to find the record about green roof insulation. What is the record title?",
        "answer": "Moss Blanket",
    },
    {
        "query": "satellite wildfire mapping",
        "question": "Use the MCP tools to find the record about satellite wildfire mapping. What is the record title?",
        "answer": "Ember Atlas",
    },
    {
        "query": "fermentation sensor brewery",
        "question": "Use the MCP tools to find the record about fermentation sensors in a brewery. What is the record title?",
        "answer": "Yeast Whisper",
    },
    {
        "query": "rail tunnel airflow",
        "question": "Use the MCP tools to find the record about airflow in rail tunnels. What is the record title?",
        "answer": "Tunnel Pulse",
    },
    {
        "query": "museum climate microgrid",
        "question": "Use the MCP tools to find the record about museum climate control and microgrids. What is the record title?",
        "answer": "Gallery Grid",
    },
    {
        "query": "orchard frost prediction",
        "question": "Use the MCP tools to find the record about orchard frost prediction. What is the record title?",
        "answer": "Frost Lantern",
    },
    {
        "query": "city curb delivery",
        "question": "Use the MCP tools to find the record about city curb delivery routing. What is the record title?",
        "answer": "Curb Queue",
    },
]


class MCPSearchTasksetConfig(vf.TasksetConfig):
    max_turns: int = 6
    examples: list[vf.JsonData] | None = None
    toolsets: list[vf.ToolsetConfig] = [SearchToolsetConfig()]


class MCPSearchTask(vf.Task):
    query: str
    question: str
    answer: str


class MCPSearchTaskset(vf.Taskset[MCPSearchTasksetConfig]):
    task_type = MCPSearchTask

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        return load_tasks(
            examples=self.config.examples, max_turns=self.config.max_turns
        )

    def load_system_prompt(self, config: MCPSearchTasksetConfig) -> vf.SystemPrompt:
        _ = config
        return SYSTEM_PROMPT

    @vf.reward(weight=1.0)
    async def exact_title_reward(self, task: MCPSearchTask, state: vf.State) -> float:
        completion = state.completion
        messages = vf.get_messages(completion, role="assistant")
        response = str(messages[-1].content or "") if messages else ""
        return float(task.answer.lower() in response.lower())


def load_tasks(
    examples: Iterable[vf.JsonData] | None = None,
    *,
    max_turns: int = 6,
):
    records = examples if examples is not None else DEFAULT_EXAMPLES
    for index, record in enumerate(records):
        question = str(record["question"])
        yield {
            **dict(record),
            "example_id": index,
            "max_turns": max_turns,
            "prompt": [{"role": "user", "content": question}],
        }


def load_taskset(config: MCPSearchTasksetConfig) -> MCPSearchTaskset:
    return MCPSearchTaskset(config=config)
