import asyncio
import json

import verifiers as vf
from verifiers.v1.types import ConfigMap
from verifiers.v1.utils.judge_utils import (
    clamp_float,
    parse_judge_json,
    truncate_command_record,
    truncate_text,
)

SYSTEM_PROMPT = """You are a careful sandbox operator.

You are running inside a sandboxed harness program. Use the callable bash tool
to complete the requested file task inside that same primary program sandbox.
Before answering, write:

- `/tmp/answer.txt` with the exact requested answer text;
- `/tmp/worklog.md` with a short note about what you did.

Then reply with the final answer text only.
"""

FILE_AUDIT_SYSTEM_PROMPT = """You are a file-state auditor.

You are running inside the same sandbox as the answer rollout and have a
callable bash tool.
Call bash to inspect `/tmp/answer.txt` and `/tmp/worklog.md`, then report
whether the files exist and what they contain. Do not assign a numeric score.
"""

COMMAND_AUDIT_SYSTEM_PROMPT = """You are a process auditor.

You are running inside the same sandbox as the answer rollout and have a
callable bash tool.
Call bash to inspect `/tmp`, file metadata, and any useful command artifacts.
Report whether the sandbox state looks consistent with the task. Do not assign
a numeric score.
"""

REWARD_JUDGE_SYSTEM_PROMPT = """You are a scoring judge.

You are running inside the same sandbox as the answer rollout and have a
callable bash tool.
Call bash to inspect `/tmp/answer.txt` before scoring. Respond with compact JSON
only:

{"score": 0.0-1.0, "reason": "..."}
"""

TASKS: list[vf.ConfigData] = [
    {
        "task_id": "exact-token",
        "answer": "prime-v1-shared-sandbox",
        "instruction": (
            "Create `/tmp/answer.txt` containing exactly `prime-v1-shared-sandbox`."
        ),
    },
    {
        "task_id": "reverse-token",
        "answer": "xobdnas-derahs",
        "instruction": (
            "Create `/tmp/answer.txt` containing exactly the reverse of "
            "`shared-sandbox`."
        ),
    },
    {
        "task_id": "joined-words",
        "answer": "taskset-harness-runtime",
        "instruction": (
            "Create `/tmp/answer.txt` containing the words taskset, harness, "
            "and runtime joined by hyphens."
        ),
    },
    {
        "task_id": "uppercase-token",
        "answer": "SANDBOX",
        "instruction": "Create `/tmp/answer.txt` containing `sandbox` in uppercase.",
    },
    {
        "task_id": "lowercase-token",
        "answer": "runtime",
        "instruction": "Create `/tmp/answer.txt` containing `RUNTIME` in lowercase.",
    },
    {
        "task_id": "count-letters",
        "answer": "9",
        "instruction": (
            "Create `/tmp/answer.txt` containing the number of letters in `verifiers`."
        ),
    },
    {
        "task_id": "repeat-prefix",
        "answer": "v1-v1-v1",
        "instruction": (
            "Create `/tmp/answer.txt` containing `v1` repeated three times, "
            "joined by hyphens."
        ),
    },
    {
        "task_id": "basename",
        "answer": "answer.txt",
        "instruction": (
            "Create `/tmp/answer.txt` containing the basename of the path "
            "`/tmp/answer.txt`."
        ),
    },
    {
        "task_id": "math-sum",
        "answer": "42",
        "instruction": "Create `/tmp/answer.txt` containing the sum of 19 and 23.",
    },
    {
        "task_id": "sorted-words",
        "answer": "alpha,beta,gamma",
        "instruction": (
            "Create `/tmp/answer.txt` containing alpha, beta, and gamma in "
            "alphabetical order, separated by commas and no spaces."
        ),
    },
]


PROGRAM_SANDBOX = {
    "image": "python:3.11-slim",
    "scope": "rollout",
    "network_access": True,
    "timeout_minutes": 20,
    "command_timeout": 120,
}


class ParallelSandboxTasksetConfig(vf.TasksetConfig):
    num_examples: int = -1


class ParallelSandboxHarnessConfig(vf.HarnessConfig):
    turn_limit: int = 4


class ParallelSandboxEnvConfig(vf.EnvConfig):
    taskset: ParallelSandboxTasksetConfig
    harness: ParallelSandboxHarnessConfig


async def bash(command: str, sandbox, state) -> str:
    """Run a bash command in the active program sandbox."""
    result = await sandbox.execute(command, timeout=120, working_dir="/tmp")
    output = {
        "exit_code": int(getattr(result, "exit_code", 0)),
        "stdout": truncate_text(str(getattr(result, "stdout", "") or "")),
        "stderr": truncate_text(str(getattr(result, "stderr", "") or "")),
    }
    state.setdefault("bash_tool_outputs", []).append(output)
    return json.dumps(output, ensure_ascii=False)


@vf.update(priority=10)
async def parallel_sandbox_audit(task, state) -> None:
    messages = vf.get_messages(state.get("completion") or [], role="assistant")
    response = str(messages[-1].content or "") if messages else ""
    audit_specs = [
        (
            "file_audit",
            FILE_AUDIT_SYSTEM_PROMPT,
            file_audit_prompt(task, response),
        ),
        (
            "command_audit",
            COMMAND_AUDIT_SYSTEM_PROMPT,
            command_audit_prompt(task),
        ),
    ]

    async def run_audit(
        label: str, system_prompt: str, prompt: str
    ) -> tuple[str, vf.State]:
        audit_task = vf.Task(
            {
                "prompt": [{"role": "user", "content": prompt}],
                "max_turns": 2,
            }
        ).freeze()
        audit_state = state.for_task(
            audit_task,
            borrow=["model", "sandbox"],
            tools="bash",
            transcript="append",
        )
        audit_state = await vf.Harness(
            system_prompt=system_prompt,
            max_turns=2,
        ).run(audit_task, audit_state)
        return label, audit_state

    audit_states = await asyncio.gather(
        *(
            run_audit(label, system_prompt, prompt)
            for label, system_prompt, prompt in audit_specs
        )
    )
    state["parallel_audits"] = []
    for label, audit_state in audit_states:
        messages = vf.get_messages(
            audit_state.get("completion") or [], role="assistant"
        )
        findings = str(messages[-1].content or "") if messages else ""
        state["parallel_audits"].append(
            {
                "name": label,
                "findings": findings,
                "trajectory_id": audit_state.get("trajectory_id"),
            }
        )


@vf.reward(weight=1.0)
async def sandbox_stage_score(task, state) -> float:
    judge_task = vf.Task(
        {
            "prompt": [
                {
                    "role": "user",
                    "content": reward_prompt(task, state),
                }
            ],
            "max_turns": 2,
        }
    ).freeze()
    judge_state = state.for_task(judge_task, borrow=["model", "sandbox"], tools="bash")
    judge_state = await vf.Harness(
        system_prompt=REWARD_JUDGE_SYSTEM_PROMPT,
        max_turns=2,
    ).run(judge_task, judge_state)
    messages = vf.get_messages(judge_state.get("completion") or [], role="assistant")
    judge_text = str(messages[-1].content or "") if messages else ""
    parsed = parse_judge_json(judge_text)
    score = clamp_float(parsed.get("score", 0.0))
    state["reward_judge"] = {
        "score": score,
        "reason": str(parsed.get("reason", "")),
        "raw": judge_text,
    }
    return score


@vf.cleanup(priority=10)
async def collect_program_sandbox_commands(task, state) -> None:
    _ = task
    state["program_sandbox_commands"] = [
        truncate_command_record(record) for record in state.get("sandbox_commands", [])
    ]
    state.pop("sandbox_commands", None)


@vf.metric(priority=-10)
async def bash_calls(task, state) -> float:
    _ = task
    return float(len(state.get("sandbox_commands", [])))


@vf.metric
async def update_audits(task, state) -> float:
    _ = task
    audits = state.get("parallel_audits", [])
    return float(len(audits) if isinstance(audits, list) else 0)


def file_audit_prompt(task: ConfigMap, response: str) -> str:
    return (
        "Task instruction:\n"
        f"{task['instruction']}\n\n"
        "Expected answer text:\n"
        f"{task['answer']}\n\n"
        "Assistant final answer:\n"
        f"{response}\n\n"
        "Call bash to inspect the sandbox. A good command is:\n"
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "import json\n"
        "paths = ['/tmp/answer.txt', '/tmp/worklog.md']\n"
        "print(json.dumps({p: {\n"
        "    'exists': Path(p).exists(),\n"
        "    'text': Path(p).read_text(errors='replace') if Path(p).exists() else '',\n"
        "} for p in paths}))\n"
        "PY\n"
    )


def command_audit_prompt(task: ConfigMap) -> str:
    return (
        "Task instruction:\n"
        f"{task['instruction']}\n\n"
        "Call bash to inspect the sandbox file layout and metadata. A good "
        "command is:\n"
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "import json\n"
        "payload = {\n"
        "    'tmp_files': sorted(str(p) for p in Path('/tmp').glob('*')),\n"
        "    'answer_size': Path('/tmp/answer.txt').stat().st_size if Path('/tmp/answer.txt').exists() else None,\n"
        "    'worklog_size': Path('/tmp/worklog.md').stat().st_size if Path('/tmp/worklog.md').exists() else None,\n"
        "}\n"
        "print(json.dumps(payload))\n"
        "PY\n"
    )


def reward_prompt(task: ConfigMap, state: ConfigMap) -> str:
    messages = vf.get_messages(state.get("completion") or [], role="assistant")
    response = str(messages[-1].content or "") if messages else ""
    return (
        "Task instruction:\n"
        f"{task['instruction']}\n\n"
        "Expected answer text:\n"
        f"{task['answer']}\n\n"
        "Assistant final answer:\n"
        f"{response}\n\n"
        "Update-stage audit findings:\n"
        f"{json.dumps(state.get('parallel_audits', []), indent=2)}\n\n"
        "Call bash to inspect `/tmp/answer.txt` directly, then score whether "
        "the sandbox state and final answer satisfy the task."
    )


def source(num_examples: int = -1):
    rows = TASKS if num_examples < 0 else TASKS[:num_examples]
    for index, row in enumerate(rows):
        yield {
            **row,
            "example_id": index,
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        f"{row['instruction']}\n\n"
                        "Use the bash tool for the file operation, write "
                        "`/tmp/worklog.md`, then answer with the requested text only."
                    ),
                }
            ],
            "max_turns": 4,
        }


def load_taskset(config: ParallelSandboxTasksetConfig) -> vf.Taskset:
    def load_rows():
        return source(num_examples=config.num_examples)

    return vf.Taskset(
        source=load_rows,
        system_prompt=SYSTEM_PROMPT,
        toolsets=[vf.Toolset(tools=[bash], write=True, sandbox="program")],
        updates=[parallel_sandbox_audit],
        rewards=[sandbox_stage_score],
        metrics=[bash_calls, update_audits],
        cleanups=[collect_program_sandbox_commands],
        config=config,
    )


class ParallelSandboxHarness(vf.Harness):
    def __init__(self, config: ParallelSandboxHarnessConfig | None = None):
        config = ParallelSandboxHarnessConfig(config)
        super().__init__(
            program={"sandbox": True, "channels": "callable"},
            sandbox=PROGRAM_SANDBOX,
            max_turns=config.turn_limit,
            config=config,
        )


def load_harness(config: ParallelSandboxHarnessConfig) -> ParallelSandboxHarness:
    return ParallelSandboxHarness(config=config)


def load_environment(config: ParallelSandboxEnvConfig) -> vf.Env:
    return vf.Env(
        taskset=load_taskset(config=config.taskset),
        harness=load_harness(config=config.harness),
    )
