import verifiers as vf
from datasets import Dataset

# idea here is to create a bunch of sandboxes
# and each sandbox will run jupyter and a streaming http jupyter mcp server
# this allows each rollout and agent to have a completely stateful private workspace
# i imagine this would be perfect for implementing 
# notebook bench or jupyter bench or whatever its called

# the below is the beginning of an extension of mcpenv
# in which we add a data loading function
# so when the rollout starts the agent has the notebook mcp with data to work with

#class NotebookEnv(vf.MCPEnv):
#    def __init__(self):
#        super().__init__()
#
#    async def setup_state(self, state: State, **kwargs):
#        state = super().setup_state(state, **kwargs)
#        sandbox = state["sandbox"]
#        await load_data(sandbox)


def load_environment(**kwargs):
    ds = Dataset.from_dict(
        {
            "question": [
                "",
            ],
            "answer": [""],
        }
    )

    rub = vf.JudgeRubric(judge_model="gpt-4.1-mini")

    async def judge_reward(judge, prompt, completion, answer, state):
        judge_response = await judge(prompt, completion, answer, state)
        return 1.0 if "yes" in judge_response.lower() else 0.0

    env = vf.MCPEnv(
        mcp_servers=[
            {
                "name": "notebook-mcp", 
                "command": "uvx", 
                "args": [
                    "jupyter-mcp-serveral@latest",
                ],
                "env": {
                    "JUPYTER_URL": "http://localhost:8888",
                    "JUPYTER_TOKEN": "MY_TOKEN",
                    "ALLOW_IMG_OUTPUT": "true"
                }
            }
        ],
        transport_type="sandbox",
        connection_scope="rollout",
        sandbox_image="python:3.11-slim",
        sandbox_cpu_cores=2,
        sandbox_memory_gb=4,
        dataset=ds,
        rubric=rub,
    )

    return env
