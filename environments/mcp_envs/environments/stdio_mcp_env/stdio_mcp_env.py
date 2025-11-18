import verifiers as vf
from datasets import Dataset

def load_environment(**kwargs):
    ds = Dataset.from_dict(
        {
            "question": [
                "Find out what Prime Intellect's newest announcement was from their website, give me the headline in 2 words. Their url is primeintellect.ai",
            ],
            "answer": ["ENVIRONMENTS HUB"],
        }
    )

    rub = vf.JudgeRubric(judge_model="gpt-4.1-mini")

    async def judge_reward(judge, prompt, completion, answer, state):
        judge_response = await judge(prompt, completion, answer, state)
        return 1.0 if "yes" in judge_response.lower() else 0.0

    env = vf.MCPEnv(
        mcp_servers=[
            {"name": "fetch", "command": "uvx", "args": ["mcp-server-fetch"]}
        ],
        transport_type="stdio",
        connection_scope="rollout",
        dataset=ds,
        rubric=rub,
        max_turns=10
    )

    return env
