"""The export-sft entrypoint: `uv run export-sft <run-dir> [options]`.

Offline sibling of `replay`, for training data instead of scores: loads a finished run's
saved traces (`traces.jsonl`) and reshapes them into an SFT dataset prime-rl's `uv run sft`
consumes directly — a `messages` column (OpenAI chat wire shape) plus a `tool_defs` column
(the tools advertised to the model, JSON-encoded; prime-rl converts them for the chat
template). One row per branch: a linear rollout contributes one sample, a compacted /
subagent rollout one per branch (`Trace.branches` — one training sample is built per
branch). Generation-errored traces are always dropped (scoring-only errors keep their
finished transcript); `--min-reward` / `--drop-truncated` select further. Writes
`<run-dir>/sft/train.parquet` (a `load_dataset`-readable dir) or pushes to the Hub with
`--push`.
"""

import json
import logging
import sys
from pathlib import Path

from pydantic_config import cli

from verifiers.v1.cli.output import TRACES_FILE, read_traces
from verifiers.v1.configs.export_sft import ExportSftConfig
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.trace import Trace, WireTrace
from verifiers.v1.utils.logging import setup_logging

logger = logging.getLogger(__name__)

USAGE = (
    "usage: uv run export-sft <run-dir> [options]\n"
    "       reshape a finished run's saved traces into an SFT dataset "
    "(messages + tool_defs columns)"
)


def sft_rows(trace: Trace) -> list[dict]:
    """A trace's SFT rows — one per branch: the branch's conversation as OpenAI chat wire
    dicts plus the trace's advertised tools, JSON-encoded (heterogeneous JSON-schema dicts
    don't fit a fixed Arrow schema; prime-rl accepts the encoded form)."""
    tool_defs = json.dumps(
        [t.model_dump(mode="json", exclude_none=True) for t in trace.tool_defs or []]
    )
    return [
        {
            "messages": [message_to_wire(m) for m in branch.messages],
            "tool_defs": tool_defs,
        }
        for branch in trace.branches
        if branch.messages
    ]


def select(traces: list[Trace], config: ExportSftConfig) -> list[Trace]:
    """The traces worth training on. Generation failures (`stop_condition == "error"`) always
    drop — a broken transcript is not a sample. A scoring-only error is different: `Trace.stop`
    keeps the first stop condition, so the generation outcome survives and the conversation is
    complete — those stay (mirroring `replay`, which re-scores exactly these), though their
    reward may be partial/zero, which `--min-reward` handles. Truncated and low-reward traces
    drop per config."""
    return [
        t
        for t in traces
        if t.stop_condition != "error"
        and not (config.drop_truncated and t.is_truncated)
        and (config.min_reward is None or t.reward >= config.min_reward)
    ]


def run_export(config: ExportSftConfig, source: Path) -> Path | str:
    """Export `source`'s traces per `config`; return the parquet dir or pushed repo id."""
    traces = read_traces(source, WireTrace)
    kept = select(traces, config)
    rows = [row for trace in kept for row in sft_rows(trace)]
    logger.info(
        "export-sft: %d trace(s) -> kept %d -> %d row(s)",
        len(traces),
        len(kept),
        len(rows),
    )
    if not rows:
        raise SystemExit("export-sft: no rows to export after selection")

    from datasets import Dataset  # deferred: heavy import, and only needed here

    dataset = Dataset.from_list(rows)
    if config.push:
        dataset.push_to_hub(config.push)
        logger.info("export-sft: pushed %d row(s) to %s", len(rows), config.push)
        return config.push
    out = config.output_dir or source / "sft"
    out.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(out / "train.parquet"))
    logger.info(
        "export-sft: wrote %s -> point prime-rl at data.name = %r",
        out / "train.parquet",
        str(out),
    )
    return out


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    if not argv or any(a in ("-h", "--help") for a in argv):
        print(USAGE)
        sys.argv = [sys.argv[0], "--help"]
        cli(ExportSftConfig)  # full, typed pydantic-config option help
        return
    source = Path(argv.pop(0))  # the finished run dir to export
    if not (source / TRACES_FILE).exists():
        raise SystemExit(f"{USAGE}\nno {TRACES_FILE} in {source}")
    sys.argv = [sys.argv[0], *argv]
    config = cli(ExportSftConfig)
    setup_logging("DEBUG" if config.verbose else "INFO")
    run_export(config, source)


if __name__ == "__main__":
    main()
