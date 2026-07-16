"""Resume an interrupted eval: reload its finished rollouts and run only the rest.

A run writes `config.toml` + `traces.jsonl` into its output dir. `--resume <dir>` reloads
that config verbatim (so it takes no other flags) and writes back into the same dir. `load`
brings the good saved rollouts back into memory and re-runs only what's still owed: the
*missing* rollouts (never written — the run was interrupted) and the *errored* ones (written
with an error; dropped and redone). The loaded traces rejoin the run everywhere — counted,
displayed, pushed, and printed alongside this session's — so a resumed run picks up exactly
where the interrupted one stopped.
"""

import json
import tomllib
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from pydantic_core import from_json

from verifiers.v1.cli.output import CONFIG_FILE, TRACES_FILE, sniff_record
from verifiers.v1.configs.eval import EvalConfig
from verifiers.v1.trace import RolloutRecord, WireRecord, WireTrace


def split_resume(argv: list[str]) -> tuple[Path | None, list[str]]:
    """Pull `--resume <dir>` / `--resume=<dir>` out of argv, returning (dir, the other args).
    The caller rejects any leftover args, since resume re-runs the saved config verbatim."""
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
    """Rebuild the run's `EvalConfig` from its saved `config.toml`, pointed back at its own
    output dir so the resumed rollouts append to the same `traces.jsonl`."""
    config_path = resume_dir / CONFIG_FILE
    if not config_path.exists():
        raise SystemExit(
            f"--resume: no config.toml in {resume_dir} - not an eval output dir"
        )
    config = EvalConfig.model_validate(tomllib.loads(config_path.read_text()))
    config.resume = resume_dir
    config.output_dir = resume_dir
    return config


def _good_row(row: dict) -> bool:
    """Whether a parsed traces.jsonl row is a keepable finished rollout. A record row
    is good iff it has no rollout-level errors (`errors`; `error` in the earliest
    record files) and none of its traces errored; a pre-record row (one bare trace) is
    good iff the trace didn't error."""
    if sniff_record(row):
        return (
            not row.get("error")
            and not row.get("errors")
            and not any(t.get("errors") for t in row.get("traces") or [])
        )
    return not row.get("errors")


def load(
    resume_dir: Path,
    selected_idxs: list[int],
    num_rollouts: int,
    complete: Callable[[RolloutRecord], bool] | None = None,
) -> tuple[list[RolloutRecord], dict[int, int]]:
    """Load the good saved rollouts back into memory as finished records and diff them
    against the run's target (`num_rollouts` per selected task): returns (the kept
    records, rollouts owed per task idx). A rollout is kept or redone as a unit — the
    record — so a multi-trace rollout interrupted mid-write is simply owed again.
    `complete` is the environment's verdict on a parsed record (`Environment.complete`)
    — an env that deliberately tolerates errored participants keeps the rollouts it
    accepted; without it (the server path, whose env lives in the workers) the default
    file-shape verdict applies: no errors anywhere (`_good_row`), so an errored rollout
    is dropped and re-run. Rewrites `traces.jsonl` to just the kept rows — verbatim,
    via a temp file + atomic rename, so an interrupted resume can't corrupt the prior
    good results — and the resumed rollouts then append. Pre-record files (one bare
    trace per line) load the same way, each trace as a single-trace record.
    `WireRecord`/`WireTrace` read any taskset's file without importing it."""
    path = resume_dir / TRACES_FILE
    selected = set(selected_idxs)

    def parse(row: dict) -> RolloutRecord:
        if sniff_record(row):
            return WireRecord.model_validate(row)
        return RolloutRecord.of(WireTrace.model_validate(row))

    good: dict[int, list[tuple[bytes, RolloutRecord | None]]] = defaultdict(list)
    if path.exists():
        with path.open("rb") as results:
            for line in results:
                if not line.strip():
                    continue
                try:
                    try:
                        row = from_json(line)
                    except ValueError:
                        row = json.loads(line)
                    idx = row["task"]["data"]["idx"]
                except (ValueError, KeyError, TypeError):
                    # A torn final line (the run died mid-write) or a foreign shape
                    # is not a keepable rollout — it's owed again, never a crash.
                    continue
                if idx not in selected or len(good[idx]) >= num_rollouts:
                    continue
                record: RolloutRecord | None = None
                if complete is None:
                    keepable = _good_row(row)
                else:
                    try:
                        record = parse(row)
                        keepable = complete(record)
                    except Exception:  # malformed row: redo it
                        keepable = False
                if keepable:
                    good[idx].append(
                        (line if line.endswith(b"\n") else line + b"\n", record)
                    )
    keep: list[bytes] = []
    records: list[RolloutRecord] = []
    owed: dict[int, int] = {}
    for idx in selected_idxs:
        rows = good.get(idx, [])
        keep.extend(line for line, _ in rows)
        records.extend(
            record if record is not None else parse(from_json(line))
            for line, record in rows
        )
        if missing := num_rollouts - len(rows):
            owed[idx] = missing
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_bytes(b"".join(keep))
    tmp.replace(path)
    return records, owed


def nothing_to_resume_msg(resume_dir: Path, num_tasks: int, num_rollouts: int) -> str:
    """The message shown (and then exit 0 - the run is already complete) when every selected
    rollout already completed without error."""
    return (
        f"nothing to resume in {resume_dir}: all {num_tasks}x{num_rollouts} rollouts "
        f"already completed without error"
    )
