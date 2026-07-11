"""Resume an interrupted eval by rerunning missing or incomplete graph invocations."""

import json
import tomllib
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from pydantic_core import from_json

from verifiers.v1.cli.output import CONFIG_FILE, TRACES_FILE
from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.rollout import Phase, Rollout
from verifiers.v1.task import Task
from verifiers.v1.topology import AgentGraph, graph_complete
from verifiers.v1.trace import Trace


def split_resume(argv: list[str]) -> tuple[Path | None, list[str]]:
    for i, arg in enumerate(argv):
        if arg == "--resume":
            if i + 1 >= len(argv):
                raise SystemExit(
                    "--resume needs an output dir: uv run eval --resume <dir>"
                )
            return Path(argv[i + 1]), argv[:i] + argv[i + 2 :]
        if arg.startswith("--resume="):
            return Path(arg.split("=", 1)[1]), argv[:i] + argv[i + 1 :]
    return None, argv


def load_resume_config(resume_dir: Path) -> EvalConfig:
    config_path = resume_dir / CONFIG_FILE
    if not config_path.exists():
        raise SystemExit(
            f"--resume: no config.toml in {resume_dir} - not an eval output dir"
        )
    config = EvalConfig.model_validate(tomllib.loads(config_path.read_text()))
    config.resume = resume_dir
    config.output_dir = resume_dir
    return config


class Finished(Rollout):
    def __init__(self, trace: Trace) -> None:
        self.trace = trace
        self.task = Task(trace.task.data)
        self.phase = Phase.DONE
        self.runtime = None


def load(
    resume_dir: Path,
    selected_idxs: list[int],
    num_rollouts: int,
    complete: Callable[[AgentGraph], bool] = graph_complete,
) -> tuple[list[AgentGraph], dict[int, int]]:
    """Keep the graph records the topology considers complete and return invocations
    still owed per seed task. `complete` is the topology's instance-validity verdict
    (`Topology.complete`) — an incomplete record is dropped and its invocation re-run;
    the default is the conservative rule, applied when no live topology is at hand
    (server-mode eval). Kept lines are rewritten verbatim (temp file + atomic rename),
    so an interrupted resume can't corrupt the prior good results."""
    path = resume_dir / TRACES_FILE
    selected = set(selected_idxs)
    good: dict[int, list[tuple[bytes, AgentGraph]]] = defaultdict(list)
    if path.exists():
        with path.open("rb") as results:
            for line in results:
                if not line.strip():
                    continue
                try:
                    row = from_json(line)
                except ValueError:
                    row = json.loads(line)
                idx = row["task"]["data"]["idx"]
                if idx not in selected or len(good[idx]) >= num_rollouts:
                    continue
                graph = AgentGraph.load(row)
                if complete(graph):
                    good[idx].append(
                        (line if line.endswith(b"\n") else line + b"\n", graph)
                    )
    keep: list[tuple[bytes, AgentGraph]] = []
    owed: dict[int, int] = {}
    for idx in selected_idxs:
        rows = good.get(idx, [])
        keep.extend(rows)
        if missing := num_rollouts - len(rows):
            owed[idx] = missing
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_bytes(b"".join(line for line, _ in keep))
    tmp.replace(path)
    return [graph for _, graph in keep], owed


def nothing_to_resume_msg(resume_dir: Path, num_tasks: int, num_rollouts: int) -> str:
    return f"nothing to resume in {resume_dir}: all {num_tasks}x{num_rollouts} invocations already complete"
