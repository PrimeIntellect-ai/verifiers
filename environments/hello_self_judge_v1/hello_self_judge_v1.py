from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast

import verifiers.v1 as vf
from verifiers.v1.utils.judge_utils import (
    clamp_float,
    completion_text,
    parse_judge_json,
    truncate_command_record,
    truncate_text,
)

SYSTEM_PROMPT = """You are a web evidence assistant running in an isolated sandbox.

Use the bash tool to fetch the requested public web pages and inspect their
contents. Before your final answer, write `/tmp/evidence.md` with:

- the source URL or URLs you used;
- short copied excerpts or facts from those sources;
- notes explaining how the evidence supports your answer.

Then give a concise answer with a `Sources:` line. Do not answer from memory
alone.
"""

UPDATE_JUDGE_SYSTEM_PROMPT = """You are a strict evidence reviewer.

Review whether the assistant answer is supported by the sandbox evidence. You
have a bash tool connected to the same sandbox used by the answering model.

First call bash to inspect `/tmp/evidence.md` and any other relevant sandbox
state. Then write plain-language findings for a later scoring judge. Mention
strengths, gaps, and any contradiction. Do not assign a numeric score.
"""

REWARD_JUDGE_SYSTEM_PROMPT = """You convert evidence-review findings into a score.

Grade whether the answer is supported by the sandbox evidence, not whether it
matches a hidden ground truth. Respond with compact JSON only:

{"score": 0.0-1.0, "reason": "..."}

Use these criteria:
- 1.0: answer is clearly supported by fetched evidence and cites sources;
- 0.7: mostly supported but missing a detail or source citation;
- 0.4: plausible but weakly supported by the sandbox state;
- 0.0: no meaningful evidence, no answer, or answer contradicts evidence.
"""


TASKS: list[dict[str, object]] = [
    {
        "task_id": "example-domains",
        "question": (
            "Fetch https://www.iana.org/domains/reserved and explain what the "
            "example domains are reserved for."
        ),
        "seed_urls": ["https://www.iana.org/domains/reserved"],
    },
    {
        "task_id": "rfc-9110-404",
        "question": (
            "Fetch https://www.rfc-editor.org/rfc/rfc9110.txt and explain what "
            "HTTP status code 404 means."
        ),
        "seed_urls": ["https://www.rfc-editor.org/rfc/rfc9110.txt"],
    },
    {
        "task_id": "python-venv",
        "question": (
            "Fetch the Python venv documentation at "
            "https://docs.python.org/3/library/venv.html and summarize what a "
            "virtual environment is used for."
        ),
        "seed_urls": ["https://docs.python.org/3/library/venv.html"],
    },
    {
        "task_id": "robots-txt",
        "question": (
            "Fetch https://www.wikipedia.org/robots.txt and report one rule or "
            "section that appears in the file."
        ),
        "seed_urls": ["https://www.wikipedia.org/robots.txt"],
    },
    {
        "task_id": "gnu-gpl",
        "question": (
            "Fetch https://www.gnu.org/licenses/gpl-3.0.txt and summarize one "
            "permission and one condition from the GPLv3 license text."
        ),
        "seed_urls": ["https://www.gnu.org/licenses/gpl-3.0.txt"],
    },
    {
        "task_id": "python-json",
        "question": (
            "Fetch https://docs.python.org/3/library/json.html and summarize "
            "what json.dumps does."
        ),
        "seed_urls": ["https://docs.python.org/3/library/json.html"],
    },
    {
        "task_id": "iana-time-zones",
        "question": (
            "Fetch https://www.iana.org/time-zones and explain what kind of "
            "resource the Time Zone Database is."
        ),
        "seed_urls": ["https://www.iana.org/time-zones"],
    },
    {
        "task_id": "mozilla-http",
        "question": (
            "Fetch https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/"
            "Overview and summarize what HTTP is used for."
        ),
        "seed_urls": [
            "https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/Overview"
        ],
    },
    {
        "task_id": "w3c-html",
        "question": (
            "Fetch https://www.w3.org/TR/html52/introduction.html and summarize "
            "what HTML is for."
        ),
        "seed_urls": ["https://www.w3.org/TR/html52/introduction.html"],
    },
    {
        "task_id": "sqlite-about",
        "question": (
            "Fetch https://www.sqlite.org/about.html and summarize two design "
            "properties SQLite claims about itself."
        ),
        "seed_urls": ["https://www.sqlite.org/about.html"],
    },
    {
        "task_id": "iana-root-zone",
        "question": (
            "Fetch https://www.iana.org/domains/root and explain what the root "
            "zone database contains."
        ),
        "seed_urls": ["https://www.iana.org/domains/root"],
    },
    {
        "task_id": "python-pathlib",
        "question": (
            "Fetch https://docs.python.org/3/library/pathlib.html and summarize "
            "why someone would use pathlib."
        ),
        "seed_urls": ["https://docs.python.org/3/library/pathlib.html"],
    },
]


async def bash(command: str, sandbox, state) -> str:
    """Run a bash command in the rollout sandbox and return stdout/stderr."""
    result = await sandbox.execute(command, timeout=120, working_dir="/tmp")
    output = {
        "exit_code": int(getattr(result, "exit_code", 0)),
        "stdout": truncate_text(str(getattr(result, "stdout", "") or "")),
        "stderr": truncate_text(str(getattr(result, "stderr", "") or "")),
    }
    state.setdefault("bash_tool_outputs", []).append(output)
    return json.dumps(output, ensure_ascii=False)


@vf.cleanup(priority=10)
async def collect_bash_commands(task, state) -> None:
    _ = task
    state["bash_commands"] = [
        truncate_command_record(record) for record in state.get("sandbox_commands", [])
    ]
    state.pop("sandbox_commands", None)


@vf.metric
async def bash_calls(task, state) -> float:
    _ = task
    return float(len(state.get("bash_tool_outputs", [])))


@vf.reward(weight=1.0)
async def self_consistency_score(task, state) -> float:
    updated = state.get("update_judge")
    if not isinstance(updated, Mapping):
        return 0.0
    updated = cast(Mapping[str, object], updated)
    findings = str(updated.get("findings") or "")
    if not findings:
        return 0.0

    judge_task = vf.Task(
        {
            "prompt": [
                {
                    "role": "user",
                    "content": score_prompt(task, findings),
                }
            ],
            "max_turns": 1,
        }
    ).freeze()
    judge_state = state.for_task(judge_task, borrow="model")
    judge_state = await vf.Harness(
        system_prompt=REWARD_JUDGE_SYSTEM_PROMPT,
        max_turns=1,
    ).run(judge_task, judge_state)

    judge_text = completion_text(judge_state.get("completion"))
    parsed = parse_judge_json(judge_text)
    score = clamp_float(parsed.get("score", 0.0))
    state["judge"] = {
        "score": score,
        "reason": str(parsed.get("reason", "")),
        "raw": judge_text,
    }
    return score


@vf.update(priority=10)
async def sandbox_judge(task, state) -> None:
    response = completion_text(state.get("completion"))
    judge_task = vf.Task(
        {
            "prompt": [
                {
                    "role": "user",
                    "content": update_prompt(task, response),
                }
            ],
            "max_turns": 3,
        }
    ).freeze()
    judge_state = state.for_task(
        judge_task,
        borrow="model",
        tools="bash",
        transcript="append",
    )
    bash_output_start = len(state.get("bash_tool_outputs", []))
    judge_state = await vf.Harness(
        system_prompt=UPDATE_JUDGE_SYSTEM_PROMPT,
        max_turns=3,
    ).run(judge_task, judge_state)
    judge_bash_outputs = state.get("bash_tool_outputs", [])[bash_output_start:]

    state["update_judge"] = {
        "findings": completion_text(judge_state.get("completion")),
        "trajectory_id": judge_state["trajectory_id"],
        "bash_calls": len(judge_bash_outputs),
    }
    state["sandbox_report"] = judge_bash_outputs


def update_prompt(task: Mapping[str, object], response: str) -> str:
    return (
        "Task:\n"
        f"{task['question']}\n\n"
        "Expected seed URLs:\n"
        f"{json.dumps(task.get('seed_urls', []), indent=2)}\n\n"
        "Assistant answer:\n"
        f"{response}\n\n"
        "Call bash before writing findings. A useful inspection command is:\n"
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "import json\n"
        "path = Path('/tmp/evidence.md')\n"
        "payload = {\n"
        "    'evidence_exists': path.exists(),\n"
        "    'evidence_bytes': path.stat().st_size if path.exists() else 0,\n"
        "    'evidence_preview': path.read_text(errors='replace')[:6000] if path.exists() else '',\n"
        "    'tmp_files': sorted(str(p) for p in Path('/tmp').glob('*'))[:100],\n"
        "}\n"
        "print(json.dumps(payload))\n"
        "PY\n\n"
        "After inspecting the sandbox, write findings in words. Do not output "
        "JSON and do not assign a score."
    )


def score_prompt(task: Mapping[str, object], findings: str) -> str:
    return (
        "Task:\n"
        f"{task['question']}\n\n"
        "Evidence-review findings from the update stage:\n"
        f"{findings}\n\n"
        "Convert the findings to a calibrated JSON score. You cannot inspect "
        "the sandbox directly; score only from the findings above."
    )


def source(num_examples: int = -1):
    rows = TASKS if num_examples < 0 else TASKS[:num_examples]
    for index, row in enumerate(rows):
        question = str(row["question"])
        yield {
            **row,
            "example_id": index,
            "prompt": [
                {
                    "role": "user",
                    "content": (
                        f"{question}\n\n"
                        "Use bash to fetch the source material. Save your "
                        "evidence and notes to `/tmp/evidence.md` before "
                        "answering."
                    ),
                }
            ],
            "max_turns": 8,
        }


def load_bash_toolset(config=None) -> vf.Toolset:
    return vf.Toolset(
        tools=[bash],
        write=True,
        scope="rollout",
        sandbox={
            "image": "python:3.11-slim",
            "scope": "rollout",
            "network_access": True,
            "timeout_minutes": 30,
            "command_timeout": 120,
        },
        cleanups=[collect_bash_commands],
        config=config,
    )


def load_taskset(num_examples: int = -1, config=None) -> vf.Taskset:
    def load_rows():
        return source(num_examples=num_examples)

    return vf.Taskset(
        source=load_rows,
        system_prompt=SYSTEM_PROMPT,
        toolsets=[load_bash_toolset()],
        updates=[sandbox_judge],
        rewards=[self_consistency_score],
        metrics=[bash_calls],
        config=config,
    )


def load_harness(max_turns: int = 8, config=None) -> vf.Harness:
    return vf.Harness(
        max_turns=max_turns,
        config=config,
    )


def load_environment(
    num_examples: int = -1,
    max_turns: int = 8,
    config=None,
) -> vf.Env:
    return vf.Env(
        taskset=load_taskset(
            num_examples=num_examples,
            config=getattr(config, "taskset", None),
        ),
        harness=load_harness(
            max_turns=max_turns,
            config=getattr(config, "harness", None),
        ),
    )


def load_v1_environment(
    num_examples: int = -1,
    max_turns: int = 8,
    config=None,
) -> vf.Env:
    return load_environment(
        num_examples=num_examples,
        max_turns=max_turns,
        config=config,
    )
