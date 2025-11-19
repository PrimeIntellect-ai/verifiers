import verifiers as vf
from datasets import Dataset
import os
from dotenv import load_dotenv
from urllib.parse import urlencode

load_dotenv()


def get_remote_url():
    API_KEY = os.getenv("SMITHERY_API_KEY")
    PROFILE = os.getenv("SMITHERY_PROFILE")
    base_url = "https://server.smithery.ai/@smithery-ai/fetch/mcp"
    params = {"api_key": API_KEY, "profile": PROFILE}
    smithery_url = f"{base_url}?{urlencode(params)}"
    return smithery_url


def load_environment(**kwargs):
    remote_url = get_remote_url()
    ds = Dataset.from_dict(
        {
            "question": [
                "Find out what Prime Intellect's newest announcement was from their website, give me the headline in 2 words. Their url is primeintellect.ai",
            ],
            "answer": ["ENVIRONMENTS HUB"],  # Or whatever the actual top result is
        }
    )

    rub = vf.JudgeRubric(judge_model="gpt-4.1-mini")

    async def judge_reward(judge, prompt, completion, answer, state):
        judge_response = await judge(prompt, completion, answer, state)
        return 1.0 if "yes" in judge_response.lower() else 0.0

    rub.add_reward_func(judge_reward, weight=1.0)

    env = vf.MCPEnv(
        mcp_servers=[
            {
                "name": "fetch-mcp-server-http",
                "command": "node",  # Not used for HTTP transport, but required by MCPServerConfig
                "args": [],
            }
        ],
        transport_type="http",
        http_urls={
            "fetch-mcp-server-http": remote_url
        },
        connection_scope="session",  # Reuse connection across rollouts
        http_timeout=60.0,
        http_max_retries=3,
        dataset=ds,
        rubric=rub,
        max_turns=10,
        **kwargs
    )

    return env
