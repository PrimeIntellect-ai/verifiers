"""One isolated invocation of Tau's official runner and evaluator."""

import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from verifiers.v1.tasksets.tau.data import (
        prepare_data,  # noqa: TC004 - embedded by bundle_program
    )


def main():
    request = json.loads(sys.argv[1])
    data = request["data"]
    os.environ["TAU2_DATA_DIR"] = str(
        prepare_data(data["repository"], data["revision"])
    )

    from tau2.data_model.tasks import Task  # ty: ignore[unresolved-import]
    from tau2.evaluator import evaluator_nl_assertions  # ty: ignore[unresolved-import]
    from tau2.evaluator.evaluator import EvaluationType  # ty: ignore[unresolved-import]

    connections = request["connections"]
    evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS = connections["judge"]["model"]
    evaluator_nl_assertions.DEFAULT_LLM_NL_ASSERTIONS_ARGS = connections["judge"][
        "args"
    ]
    arguments = {
        "domain": data["domain"],
        "agent": "llm_agent",
        "user": "user_simulator",
        "llm_agent": connections["assistant"]["model"],
        "llm_args_agent": connections["assistant"]["args"],
        "llm_user": connections["user"]["model"],
        "llm_args_user": connections["user"]["args"],
    }
    if data["max_steps"] is not None:
        arguments["max_steps"] = data["max_steps"]
    task = Task.model_validate(data["tau_task"])
    if data["runner"] == "synth":
        from tau2.run import run_task  # ty: ignore[unresolved-import]

        result = run_task(task=task, evaluation_type=EvaluationType.ALL, **arguments)
    else:
        from tau2 import TextRunConfig  # ty: ignore[unresolved-import]
        from tau2.runner import run_single_task  # ty: ignore[unresolved-import]

        config = TextRunConfig(**arguments, retrieval_config=data["retrieval_config"])
        result = run_single_task(config, task, evaluation_type=EvaluationType.ALL)
    Path(request["output"]).write_text(result.model_dump_json())


if __name__ == "__main__":
    main()
