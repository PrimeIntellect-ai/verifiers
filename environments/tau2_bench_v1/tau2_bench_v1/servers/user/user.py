from tau2.data_model.tasks import Task as TauTask
import verifiers.v1 as vf

from .config import UserConfig


class User(vf.User[UserConfig]):
    @vf.user(
        args={
            "tau2_task": "task.tau2_task",
            "completion": "state.completion",
        },
        sets={
            "tau2": "state.extras.tau2",
            "stop_condition": "state.stop_condition",
        },
    )
    def respond(self, tau2_task: dict, completion: list[dict]) -> dict:
        tau_task = TauTask.model_validate(tau2_task)
        completion = [
            message
            for message in completion
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
                "tau2": {
                    "task_id": tau_task.id,
                    "step_count": 0,
                    "num_errors": 0,
                    "reward": 0.0,
                },
            }
        final_answer = str(completion[-1].get("content") or "")
        reward = float(bool(final_answer.strip()))
        return {
            "messages": [],
            "tau2": {
                "task_id": tau_task.id,
                "step_count": len(completion),
                "num_errors": 0,
                "reward": reward,
            },
            "stop_condition": "tau2_user_done",
        }
