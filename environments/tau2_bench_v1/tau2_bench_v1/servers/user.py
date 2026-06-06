from mcp.server.fastmcp import FastMCP
from tau2.data_model.tasks import Task as TauTask

mcp = FastMCP("tau2-user")


@mcp.tool()
def respond(task: dict, state: dict, transcript: list[dict]) -> dict:
    _ = transcript
    task_info = task.get("tau2_task")
    tau_task = TauTask.model_validate(task_info)
    raw_completion = state["completion"]
    if not isinstance(raw_completion, list):
        raise TypeError("User server state.completion must be a list.")
    completion = [
        message
        for message in raw_completion
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    if not completion:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": str(tau_task.user_scenario),
                }
            ],
            "scratch": {
                "tau2": {
                    "task_id": tau_task.id,
                    "step_count": 0,
                    "num_errors": 0,
                    "reward": 0.0,
                }
            },
        }
    final_answer = str(completion[-1].get("content") or "")
    reward = float(bool(final_answer.strip()))
    return {
        "messages": [],
        "scratch": {
            "tau2": {
                "task_id": tau_task.id,
                "step_count": len(completion),
                "num_errors": 0,
                "reward": reward,
            }
        },
        "stop_condition": "tau2_user_done",
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
