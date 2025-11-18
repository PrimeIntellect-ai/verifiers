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
            {
                "name": "database", 
                "command": "npx", 
                "args": [
                    "-y",
                    "mongodb-mcp-server@latest",
                    "--transport",
                    "http"
                ]
            }
        ],
        transport_type="http",
        http_urls={"database": "http://localhost:3000"},
        connection_scope="session", # we will reuse the connection 
        http_timeout=60.0,
        dataset=ds,
        rubric=rub,
    )

    return env
