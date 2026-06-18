"""general-agent-synth: synthesize NEW tool-use tasks for the general-agent benchmark.

The other half of the self-growing loop (synthesizer -> solver). Each rollout asks a coding agent
(the `bash` harness, in a container) to author a brand-new task — its `tools.py` world + tools, an
initial `db.json`, an `instruction.md`, and a gold `gold.json` solution chain — under `out/<name>/`.
`setup` stages a tiny `general_agent` shim + a `validate_task.py` self-check into the container so the
agent can iterate until its task is valid; the reward then pulls the produced task back out and
gold-checks it (the same check the solver and `uv run validate` use). No pass-rate gating (that needs
running the solver per tier — see the source env); this just rewards a structurally-valid new task.

Needs a container: `--harness.id bash --harness.runtime.type docker` (or prime/modal).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import verifiers.v1 as vf

from general_agent_v1 import tools as ga_tools
from general_agent_v1.corpus import gold_check

# The general_agent tool base classes, staged into the container so the agent's task `tools.py`
# (`from general_agent.tools import DB, Tools, tool`) imports and runs there for self-testing.
_BASE_TOOLS_SRC = Path(ga_tools.__file__).read_text()

# A self-contained uv script the agent runs to validate its task before finishing (mirrors
# `corpus.gold_check`, but standalone so it needs nothing but pydantic + the staged general_agent).
_VALIDATE_SRC = '''\
# /// script
# requires-python = ">=3.10"
# dependencies = ["pydantic"]
# ///
"""Validate a synthesized general-agent task dir: gold replay must change the DB, and (if defined)
verify(initial)==0 and verify(gold)==1. Usage: uv run validate_task.py out/<name>"""
import importlib.util, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # find the staged `general_agent` package
task_dir = Path(sys.argv[1])


def fail(msg):
    print("FAIL:", msg)
    sys.exit(1)


spec = importlib.util.spec_from_file_location("task_tools", task_dir / "tools.py")
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except Exception as e:
    fail(f"tools.py import error: {type(e).__name__}: {e}")
TaskDB, TaskTools, verify = getattr(mod, "TaskDB", None), getattr(mod, "TaskTools", None), getattr(mod, "verify", None)
if TaskDB is None or TaskTools is None:
    fail("tools.py must define TaskDB and TaskTools")
for name in ("db.json", "gold.json"):
    if not (task_dir / name).exists():
        fail(f"missing {name}")
initial = TaskTools(TaskDB.load(task_dir / "db.json"))
gold = TaskTools(TaskDB.load(task_dir / "db.json"))
try:
    for tool_name, kwargs in json.loads((task_dir / "gold.json").read_text()):
        gold.call_tool(tool_name, **kwargs)
except Exception as e:
    fail(f"gold replay failed: {type(e).__name__}: {e}")
if initial.db.get_hash() == gold.db.get_hash():
    fail("gold solution did not change the DB")
if verify is not None:
    if verify(initial.db) != 0.0:
        fail("verify(initial_db) != 0.0")
    if verify(gold.db) != 1.0:
        fail("verify(gold_db) != 1.0")
print("PASS")
'''

_PROTOCOL = """\
You are authoring a NEW tool-use task for the "general-agent" benchmark, in this domain:

    {family}

A task is a directory of files. Create exactly one at `out/{name}/` (choose a short snake_case
`{name}` describing the scenario). Write these files with the bash tool:

1. `tools.py` — the task's world and the tools the solver will call:

       from general_agent.tools import DB, Tools, tool
       from pydantic import BaseModel  # for any nested record models

       class TaskDB(DB):
           # pydantic fields describing the initial world (e.g. lists of records)
           ...

       class TaskTools(Tools):
           db: TaskDB

           @tool
           def do_something(self, arg: str) -> str:
               \"\"\"One-line description shown to the solver.\"\"\"
               ...        # MUST mutate self.db and return a short result string

       def verify(db: TaskDB) -> float:
           # 1.0 iff the goal state holds, else 0.0
           ...

2. `db.json` — the initial DB state as JSON (must validate against TaskDB).
3. `instruction.md` — a concrete, natural-language request telling the solver what to accomplish.
4. `gold.json` — the canonical solution: a JSON list of `[tool_name, {{kwargs}}]` steps that, applied
   in order from the initial DB, reach the goal.
5. `task.toml` — minimal metadata:

       [metadata]
       name = "{name}"
       tier = 0

Hard requirements (the reward checks these):
- The gold chain must CHANGE the DB.
- `verify(initial_db)` must be 0.0 and `verify(gold_db)` must be 1.0.

Before finishing, VALIDATE your task and iterate until it passes:

    uv run validate_task.py out/{name}

It must print `PASS`. Then stop.
"""

_FAMILIES = [
    "a library book lending system",
    "a restaurant table reservation system",
    "a warehouse inventory tracker",
    "a personal todo / task manager",
    "a support-ticket queue",
    "a flight booking system",
    "a bank account ledger",
    "a music playlist manager",
]


class GeneralAgentSynthTask(vf.Task):
    family: str
    """The domain hint the agent synthesizes a task for."""
    name: str
    """A snake_case slug for the new task (also the suggested `out/<name>/` directory)."""


class GeneralAgentSynthConfig(vf.TasksetConfig):
    families: list[str] = _FAMILIES
    """Domain hints; one synthesis job per family (truncated/cycled to `num_tasks`)."""
    num_tasks: int | None = None
    """How many synthesis jobs to emit (default: one per family)."""


class GeneralAgentSynthTaskset(
    vf.Taskset[GeneralAgentSynthTask, GeneralAgentSynthConfig]
):
    NEEDS_CONTAINER = True  # the agent writes + self-tests files in a container

    def load_tasks(self) -> list[GeneralAgentSynthTask]:
        families = self.config.families
        n = self.config.num_tasks or len(families)
        tasks = []
        for i in range(n):
            family = families[i % len(families)]
            slug = "".join(c if c.isalnum() else "_" for c in family.split()[1]).lower()
            tasks.append(
                GeneralAgentSynthTask(
                    idx=i,
                    name=f"{slug}_{i}",
                    family=family,
                    prompt=_PROTOCOL.format(family=family, name=f"{slug}_{i}"),
                )
            )
        return tasks

    async def setup(self, task: GeneralAgentSynthTask, runtime: vf.Runtime) -> None:
        """Stage the `general_agent` shim + the `validate_task.py` self-check into the container."""
        await runtime.write("general_agent/__init__.py", b"")
        await runtime.write("general_agent/tools.py", _BASE_TOOLS_SRC.encode())
        await runtime.write("validate_task.py", _VALIDATE_SRC.encode())

    @vf.reward(weight=1.0)
    async def synthesized(
        self, task: GeneralAgentSynthTask, trace: vf.Trace, runtime: vf.Runtime
    ) -> float:
        """Pull each produced `out/<name>/` task off the container and gold-check it; reward 1.0 if
        any is valid."""
        ls = await runtime.run(["sh", "-c", "ls -1 out 2>/dev/null"], {})
        names = [n for n in ls.stdout.split() if n]
        best = 0.0
        for name in names:
            with tempfile.TemporaryDirectory() as td:
                d = Path(td) / name
                d.mkdir(parents=True)
                missing = False
                for fname in (
                    "tools.py",
                    "db.json",
                    "gold.json",
                    "instruction.md",
                    "task.toml",
                ):
                    try:
                        (d / fname).write_bytes(
                            await runtime.read(f"out/{name}/{fname}")
                        )
                    except Exception:
                        if fname in ("tools.py", "db.json", "gold.json"):
                            missing = True
                            break
                if missing:
                    continue
                ok, _ = gold_check(d)
                best = max(best, float(ok))
        return best


__all__ = ["GeneralAgentSynthTaskset"]
