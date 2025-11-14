import base64
import textwrap

from datasets import Dataset

import verifiers as vf
from verifiers.envs.cli_agent_env import CliAgentEnv

_AGENT_SCRIPT = textwrap.dedent(
    """
    import os
    import sys
    from openai import OpenAI

    base_url = os.environ.get("OPENAI_BASE_URL")
    model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

    if not base_url:
        print("ERROR: OPENAI_BASE_URL not set", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=base_url, api_key="dummy-key")

    env_responses = [
        "Environment: Starting task",
        "Environment: Processing step 1",
        "Environment: Processing step 2",
        "Environment: Task complete",
    ]

    messages = []
    for i, env_msg in enumerate(env_responses):
        print(f"[ENV] {env_msg}", flush=True)
        
        user_msg = {"role": "user", "content": f"Step {i+1}: {env_msg}"}
        messages.append(user_msg)
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=50,
            )
            assistant_msg = response.choices[0].message
            messages.append(assistant_msg)
            print(f"[AGENT] Response: {assistant_msg.content}", flush=True)
        except Exception as e:
            print(f"[AGENT] Error: {e}", file=sys.stderr, flush=True)
            sys.exit(1)

    with open("/tmp/vf_complete", "w") as f:
        f.write("done")
    print("[AGENT] Completion marker written", flush=True)
    """
)

_START_COMMAND_TEMPLATE = textwrap.dedent(
    """
    bash -lc '
    set -euo pipefail

    # Install openai package if not present
    pip install -q openai || true

    agent_path="/tmp/agent_script.py"

    python - <<'PY'
import base64
from pathlib import Path

Path("{agent_path}").write_bytes(base64.b64decode("{agent_b64}"))
PY

    # Run agent script in background, then keep container alive
    python -u "$agent_path" &
    tail -f /dev/null
    '
    """
)


class DummyCliAgentEnv(CliAgentEnv):
    async def post_rollout(self, state: vf.State):
        """Check for completion marker file and set reward before sandbox destruction"""
        sandbox_id = state.get("sandbox_id")
        if not sandbox_id:
            state["reward"] = 0.0
            return

        try:
            from prime_sandboxes import AsyncSandboxClient

            sandbox_client = AsyncSandboxClient()
            result = await sandbox_client.execute_command(
                sandbox_id,
                "test -f /tmp/vf_complete && echo 'done' || echo 'not_done'",
            )
            state["reward"] = 1.0 if result.stdout.strip() == "done" else 0.0
        except Exception as e:
            self.logger.debug(f"Error checking completion marker: {e}")
            state["reward"] = 0.0


def load_environment(**kwargs) -> vf.Environment:
    agent_b64 = base64.b64encode(_AGENT_SCRIPT.encode("utf-8")).decode("utf-8")
    start_command = _START_COMMAND_TEMPLATE.format(
        agent_path="/tmp/agent_script.py",
        agent_b64=agent_b64,
    )

    dataset_rows = [
        {
            "example_id": i,
            "prompt": [{"role": "user", "content": "Run the agent script"}],
            "task": "dummy-cli-agent",
        }
        for i in range(5)
    ]
    dataset = Dataset.from_list(dataset_rows)

    def reward_func(state: vf.State) -> float:
        reward = state.get("reward")
        return 0.0 if reward is None else float(reward)

    rubric = vf.Rubric(
        funcs=[reward_func],
    )

    env = DummyCliAgentEnv(
        max_turns=10,
        timeout_seconds=300.0,
        request_timeout=60.0,
        start_command=start_command,
        dataset=dataset,
        rubric=rubric,
        interception_host=kwargs.get("interception_host"),
    )

    return env
