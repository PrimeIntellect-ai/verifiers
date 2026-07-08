"""Turn saved rollout records back into resume points.

A record is one ``Trace.to_record()`` JSONL line (prime-rl writes them to
``<output_dir>/rollouts/step_*/*_rollouts*.jsonl``). Everything here is a pure function of the
reloaded trace: pick a resume point, slice the conversation up to it, and hand back a ``Seed`` —
the new task's prompt plus enough metadata to filter, label, and restore it. The taskset
assembles seeds into tasks; nothing here touches config or I/O beyond record/index file reading.

Record files may ship with a derived sibling index (prime-rl writes ``index_<env>.jsonl`` next
to each ``train_rollouts_<env>.jsonl``): one ``index_row`` per record line with selection fields
(``task_name``, ``reward``, ``branches``, ...) and the line's byte span (``offset``, ``len``).
``iter_selected_records`` lets a consumer filter records index-side and parse only the
survivors; a missing index means full parse. ``index_path`` / ``index_row`` are the writer's
half of that contract.

Sandbox snapshots: a producer that snapshots the rollout's sandbox records the refs in
``trace.info["snapshots"]`` (``SNAPSHOTS_INFO_KEY``), keyed by the index (in ``trace.nodes``)
of the node the snapshot accompanies — the sandbox state as of that node entering the
conversation. A seed's anchor node is the node its resume point re-enters at: the restart's
fork node (compaction), the resumed tool result (tool-call), or the attempt's recorded final
node (recheck). Since resuming without the sandbox state is unfaithful once state *is*
captured, a record that carries any refs offers only snapshotted resume points — every
builder drops anchors without a ref.
"""

import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator
from glob import glob
from itertools import accumulate
from pathlib import Path
from random import Random
from typing import NamedTuple

from verifiers.v1.graph import MessageNode
from verifiers.v1.trace import Trace
from verifiers.v1.types import Messages, UserMessage

REPLAYED_PREFIXES = ("continue:", "recheck:")
"""Task-name prefixes marking a replayed task's own output — a record whose task name carries
one never seeds again, so replay doesn't compound on itself."""

RECHECK_PROMPT = (
    "Carefully check your work above. Re-verify the reasoning and the final answer against the "
    "task; if you find any mistakes, fix them. Then state your final answer."
)

SNAPSHOTS_INFO_KEY = "snapshots"
"""Where a record carries sandbox snapshot refs: ``trace.info["snapshots"]`` maps the index of a
node in ``trace.nodes`` (as a string — JSON keys) to an opaque ref restorable into a runtime,
capturing the sandbox as of that node entering the conversation."""


class Seed(NamedTuple):
    """One resume point: the replayed task's prompt, a display name, the estimated token length
    of the seeded context (for ``max_seed_tokens`` filtering), and the sandbox snapshot ref of
    its anchor node (None when the record carries no snapshots)."""

    prompt: str | Messages
    name: str
    tokens: int
    snapshot: str | None = None


def _natural(path: str) -> tuple[int | str, ...]:
    """Sort key ordering embedded numbers numerically, so ``step_9`` sorts before ``step_10``."""
    return tuple(
        int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path)
    )


def expand_records(records: str | list[str]) -> list[str]:
    """The sorted, deduplicated files a ``records`` value matches. A stable file order
    (numeric-aware, so step dirs sort in step order) is what keeps task indices identical
    across env-server pool workers, each re-expanding the globs as the source grows."""
    patterns = [records] if isinstance(records, str) else records
    matched = {path for pattern in patterns for path in glob(pattern, recursive=True)}
    return sorted(matched, key=_natural)


def iter_records(paths: Iterable[Path]) -> Iterator[dict]:
    """The record dicts of the given JSONL files, in file-then-line order."""
    for path in paths:
        with open(path) as lines:
            for number, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"{path}:{number} is not a JSON rollout record"
                    ) from e


def index_path(record_path: Path) -> Path | None:
    """The sibling index file a record file's name derives: ``index_<env>.jsonl`` next to
    ``train_rollouts_<env>.jsonl`` / ``eval_rollouts_<env>.jsonl``, or None when the filename
    doesn't follow that convention. Pure filename derivation — the writer's output path and the
    reader's lookup; existence is the reader's concern."""
    name = record_path.name
    for prefix in ("train_rollouts_", "eval_rollouts_"):
        if name.startswith(prefix):
            return record_path.with_name("index_" + name[len(prefix) :])
    return None


def index_row(trace: Trace, *, offset: int, length: int, **extra) -> dict:
    """One index line for a record: the fields ``iter_selected_records`` filters on —
    ``task_name``, ``reward``, ``branches`` — plus the record line's byte span (``offset``,
    ``len``); those five are required. The rest is informational provenance for pool tooling
    and lineage: ``trace``, ``task_idx``, ``turns``, ``stop``, and whatever the writer adds via
    ``extra`` (prime-rl records ``step`` and ``policy_version``)."""
    return {
        "trace": trace.id,
        "task_idx": trace.task.idx,
        "task_name": trace.task.name,
        "reward": trace.reward,
        "branches": trace.num_branches,
        "turns": trace.num_turns,
        "stop": trace.stop_condition,
        "offset": offset,
        "len": length,
        **extra,
    }


def iter_selected_records(
    path: Path,
    skipped: dict[str, int],
    *,
    min_reward: float | None = None,
    max_reward: float | None = None,
    min_branches: int = 1,
) -> Iterator[dict]:
    """The record dicts of one JSONL file, index-selected when a sibling index exists: rows
    whose record could never seed — already-replayed task names (``REPLAYED_PREFIXES``),
    rewards outside ``[min_reward, max_reward]``, branch counts below ``min_branches`` (2 when
    only compacted rollouts can seed) — are counted into ``skipped`` (the taskset's skip-reason
    keys, updated in place) and never parsed; only the surviving byte spans are read. Without
    an index every record line is yielded — the caller's parse-side checks filter then (they
    run on index survivors too, guarding stale indexes)."""
    index = index_path(path)
    if index is None or not index.exists():
        yield from iter_records([path])
        return
    with open(index) as rows, open(path, "rb") as records:
        for line in rows:
            if not line.strip():
                continue
            row = json.loads(line)
            name = row.get("task_name") or ""
            if isinstance(name, str) and name.startswith(REPLAYED_PREFIXES):
                skipped["replayed"] += 1
                continue
            reward = row["reward"]
            if (min_reward is not None and reward < min_reward) or (
                max_reward is not None and reward > max_reward
            ):
                skipped["source_reward"] += 1
                continue
            if row["branches"] < min_branches:
                skipped["no_seed"] += 1
                continue
            records.seek(row["offset"])
            yield json.loads(records.read(row["len"]))


def node_snapshots(trace: Trace) -> dict[int, str]:
    """The trace's snapshot refs, keyed by node object id — the working form of
    ``trace.info["snapshots"]`` (malformed indices fail loudly: a producer bug, not data)."""
    refs = trace.info.get(SNAPSHOTS_INFO_KEY) or {}
    return {id(trace.nodes[int(index)]): ref for index, ref in refs.items()}


def estimate_tokens(nodes: list[MessageNode]) -> int:
    """Token length of a node path: the recorded token count when the record carries token ids
    (train rollouts), else a chars/4 estimate over the message text (eval-relay rollouts record
    no tokens)."""
    recorded = sum(len(node.token_ids) for node in nodes)
    if recorded:
        return recorded
    chars = 0
    for node in nodes:
        message = node.message
        content = getattr(message, "content", None)
        if isinstance(content, str):
            chars += len(content)
        elif content:
            chars += sum(len(getattr(part, "text", None) or "") for part in content)
        chars += len(getattr(message, "reasoning_content", None) or "")
        for call in getattr(message, "tool_calls", None) or []:
            chars += len(call.name) + len(call.arguments)
    return chars // 4


def compaction_seeds(trace: Trace) -> list[Seed]:
    """One seed per context restart, detected structurally — no compaction-prompt matching, so
    any harness's rewrite shape (its own summary prompt, kept task message, extra user turns)
    qualifies, and mixed-harness record files need no per-harness rules.

    A restart is a branch that forks off shared history on *fabricated context*: its fork node
    is unsampled, sits before the branch's first sampled node, and the whole restart context —
    the branch's unsampled prefix, everything the restarted model was (re)launched with — is
    system/user messages only. (A fork carrying assistant/tool copies is a retokenization split,
    not a restart; a branch sharing nothing is a separate conversation, e.g. a subagent.)

    The canonical ``[system, user]`` restart lowers to a plain string prompt — a fresh launch of
    the same task then reproduces the post-compaction context under *any* harness, including
    agent CLIs like ``rlm``. Other shapes seed as ``Messages`` (sans the leading system message,
    which the harness re-emits) and need a message-seeding harness."""
    seeds: list[Seed] = []
    snapshots = node_snapshots(trace)
    branches = trace.branches
    seen: set[int] = {id(node) for node in branches[0].nodes} if branches else set()
    for branch in branches[1:]:
        nodes = branch.nodes
        fork = next((i for i, node in enumerate(nodes) if id(node) not in seen), None)
        seen.update(id(node) for node in nodes)
        first_sampled = next(
            (i for i, node in enumerate(nodes) if node.sampled), len(nodes)
        )
        if fork is None or not 1 <= fork < first_sampled:
            continue
        ref = snapshots.get(id(nodes[fork]))
        if snapshots and ref is None:
            continue  # snapshotted record: an unsnapshotted anchor would resume unfaithfully
        context = nodes[:first_sampled]
        if any(node.message.role not in ("system", "user") for node in context):
            continue
        messages = [node.message for node in context]
        prompt: str | Messages
        if (
            len(messages) == 2
            and messages[0].role == "system"
            and isinstance(messages[1].content, str)
        ):
            prompt = messages[1].content
        else:
            prompt = messages[1:] if messages[0].role == "system" else messages
        if prompt == trace.task.prompt or messages == trace.task.prompt:
            continue  # the task's own launch context (branch ordering put an auxiliary branch first)
        seeds.append(
            Seed(
                prompt=prompt,
                name=f"continue:{trace.id[:8]}:compaction{len(seeds)}",
                tokens=estimate_tokens(context),
                snapshot=ref,
            )
        )
    return seeds


def tool_call_seeds(
    trace: Trace, rng: Random, max_anchors: int | None = 1
) -> list[Seed]:
    """Mid-trajectory resume points: each candidate is the conversation from the start through
    a complete tool-result run (every tool_call id of the issuing assistant answered — resuming
    after a partial run would leave dangling calls in the context). When the record carries
    sandbox snapshots, only snapshotted tool nodes are candidates. ``max_anchors`` draws that
    many candidates uniformly (deterministic in ``rng``), kept in trajectory order; None keeps
    every candidate. Seeds are ``Messages`` prompts without the leading system message (the
    harness re-emits it), so they need a message-seeding harness (``default``/``null``). Empty
    when the trace has no valid point."""
    snapshots = node_snapshots(trace)
    candidates: list[tuple[list[MessageNode], int, int]] = []
    seen: set[int] = set()
    for branch in trace.branches:
        nodes = branch.nodes
        # Prefix sums of the recorded token counts, so each anchor's estimate is O(1) — with
        # `max_anchors = None` a long trace can anchor at every tool run.
        recorded = list(accumulate((len(node.token_ids) for node in nodes), initial=0))
        for end, node in enumerate(nodes):
            if node.message.role != "tool" or id(node) in seen:
                continue
            seen.add(id(node))
            if snapshots and id(node) not in snapshots:
                continue
            if end + 1 < len(nodes) and nodes[end + 1].message.role == "tool":
                continue  # resume only at the end of the tool-result run
            start = end
            while start > 0 and nodes[start - 1].message.role == "tool":
                start -= 1
            issuer = nodes[start - 1].message if start > 0 else None
            if issuer is None or issuer.role != "assistant" or not issuer.tool_calls:
                continue
            answered = {nodes[i].message.tool_call_id for i in range(start, end + 1)}
            if {call.id for call in issuer.tool_calls} != answered:
                continue
            candidates.append((nodes, end, recorded[end + 1]))
    if max_anchors is not None and max_anchors < len(candidates):
        keep = sorted(rng.sample(range(len(candidates)), max_anchors))
        candidates = [candidates[i] for i in keep]
    seeds: list[Seed] = []
    ordinals: Counter[str] = Counter()
    for nodes, end, tokens in candidates:
        prefix = nodes[: end + 1]
        messages = [node.message for node in prefix]
        if messages[0].role == "system":
            messages = messages[1:]
        base = f"continue:{trace.id[:8]}:tool-call{end}"
        # Two branches can anchor at the same in-branch index: an ordinal disambiguates.
        ordinal = ordinals[base]
        ordinals[base] += 1
        seeds.append(
            Seed(
                prompt=messages,
                name=f"{base}-{ordinal}" if ordinal else base,
                tokens=tokens or estimate_tokens(prefix),
                snapshot=snapshots.get(id(nodes[end])),
            )
        )
    return seeds


def recheck_seed(trace: Trace, recheck_prompt: str) -> Seed | None:
    """The source rollout's finished attempt — its final branch — with truncation artifacts
    stripped (everything from the first tool run whose calls never got all their results, a
    trailing user turn the model never answered) and the verification request appended as a new
    user turn. A ``Messages`` prompt without the leading system message. The snapshot anchor is
    the branch's recorded final node (the attempt's end state). None when no model-produced
    assistant turn survives cleanup (nothing to check)."""
    branches = trace.branches
    if not branches:
        return None
    nodes = list(branches[-1].nodes)
    snapshots = node_snapshots(trace)
    ref = snapshots.get(id(nodes[-1]))
    if snapshots and ref is None:
        return None  # snapshotted record: an unsnapshotted anchor would resume unfaithfully
    for i, node in enumerate(nodes):
        message = node.message
        if message.role != "assistant" or not message.tool_calls:
            continue
        run = i + 1
        while run < len(nodes) and nodes[run].message.role == "tool":
            run += 1
        answered = {nodes[j].message.tool_call_id for j in range(i + 1, run)}
        if {call.id for call in message.tool_calls} != answered:
            nodes = nodes[:i]  # dangling calls would make the replayed prompt malformed
            break
    while nodes and nodes[-1].message.role == "user":
        nodes.pop()
    if not any(node.sampled and node.message.role == "assistant" for node in nodes):
        return None
    messages = [node.message for node in nodes]
    if messages[0].role == "system":
        messages = messages[1:]
    return Seed(
        prompt=[*messages, UserMessage(content=recheck_prompt)],
        name=f"recheck:{trace.id[:8]}",
        tokens=estimate_tokens(nodes) + len(recheck_prompt) // 4,
        snapshot=ref,
    )
