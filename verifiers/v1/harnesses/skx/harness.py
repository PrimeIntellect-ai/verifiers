import json
import logging
from pathlib import Path

from pydantic import BaseModel, Field

from verifiers.v1.clients import ModelContext
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.harness import Harness
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.task import TaskData
from verifiers.v1.trace import Trace
from verifiers.v1.types import AssistantMessage, content_text

logger = logging.getLogger(__name__)

PROGRAM_SOURCE = (Path(__file__).resolve().parent / "program.py").read_text()


class SkxCompaction(BaseModel):
    """Context compaction for long episodes. When the last completion reports a
    context of `trigger_tokens` or more, the program summarizes everything between
    the task prompt and the recent tail and continues from the summary. The trace
    records the pre- and post-compaction histories as separate branches (each its
    own training sample), and the summary completion itself is masked from the
    loss when `mask_summaries` is set."""

    enabled: bool = False
    trigger_tokens: int = Field(22528, ge=4096)
    keep_recent_messages: int = Field(6, ge=2)
    #: Token ceiling on the retained tail. `keep_recent_messages` bounds the tail
    #: by COUNT only, and a message is not a bounded quantity: with 8192-token
    #: completions and multi-thousand-token eval observations, six messages
    #: measured 22,555 tokens on a real rollout -- 69% of a 32,768 window, from a
    #: 16,384 trigger. With a 3-turn compaction cooldown behind that, further
    #: growth can overrun `max_model_len`, which is a hard vLLM failure rather
    #: than a graceful truncation. `pi` already bounds its tail in tokens; this is
    #: the same guarantee for skx. 0 disables (count-only, prior behaviour).
    keep_recent_tokens: int = Field(8192, ge=0)
    mask_summaries: bool = True


class SkxEvalContract(BaseModel):
    """Refuse a finish that never ran the trusted evaluator. Such a rollout scores
    the `no_eval` tier, so it carries no evidence about the kernel at all; instead
    of accepting the exit the program answers it with a corrective turn and lets
    the model continue. `max_nudges` bounds the correction (0 disables it); the
    agent's own turn and token budgets still end the episode."""

    max_nudges: int = Field(2, ge=0, le=4)


class SkxHarnessConfig(HarnessConfig):
    compaction: SkxCompaction = SkxCompaction()
    eval_contract: SkxEvalContract = SkxEvalContract()


_SUMMARIZER_PREFIX = "You are a context summarization assistant."


def _branch_root(trace: Trace, node) -> int | None:
    """Index of the root node of the branch `node` sits on."""
    index = node.parent
    if index is None:
        return None
    for _ in range(len(trace.nodes)):
        parent = trace.nodes[index]
        if parent.parent is None:
            return index
        index = parent.parent
    return None


def _mask_summaries(trace: Trace) -> tuple[int, int, int]:
    """Mask every sampled completion on an auxiliary summarizer branch.

    Branch identity, rather than global text matching, prevents an ordinary
    policy completion that happens to equal summary text from being masked.
    Returns ``(masked_nodes, masked_tokens, summarizer_completions)``.
    """
    masked_nodes = 0
    masked_tokens = 0

    def mask(node) -> None:
        nonlocal masked_nodes, masked_tokens
        masked_nodes += 1
        masked_tokens += sum(node.mask)
        node.mask = [False] * len(node.mask)
        node.logprobs = []
        if getattr(node, "kept_tokens", None) is not None:
            node.kept_tokens = None  # counts scatter onto mask-True positions

    summarizer_roots = {
        index
        for index, node in enumerate(trace.nodes)
        if node.parent is None
        and getattr(node.message, "role", None) == "system"
        and isinstance(getattr(node.message, "content", None), str)
        and node.message.content.startswith(_SUMMARIZER_PREFIX)
    }
    summarizer_nodes = [
        node
        for node in trace.nodes
        if node.sampled
        and isinstance(node.message, AssistantMessage)
        and _branch_root(trace, node) in summarizer_roots
    ]
    for node in summarizer_nodes:
        mask(node)
    return masked_nodes, masked_tokens, len(summarizer_nodes)


def _has_trainable_tokens(trace: Trace) -> bool:
    """Whether this trace carries token-level training signal at all.

    Only a tokenized turn attaches ids to its node; the eval relay commits the same
    message graph with empty `token_ids`/`mask` (see `graph.commit_turn`, which builds
    the assistant mask from `tokens.completion_ids` and leaves it empty when the turn
    carried no tokens). On such a trace a perfectly correct masking pass necessarily
    masks zero tokens, so a zero token count proves nothing about masking.
    """
    return any(any(node.mask) for node in trace.nodes if node.sampled)


def _unmasked_summary_tokens(trace: Trace, summaries: list[str]) -> int:
    """Trainable tokens still sitting on a sampled node that carries a tracker summary.

    This is the only compaction accounting failure that can contaminate the loss: a
    summary completion the branch walk did not recognize, left mask-True. Text identity
    is the wrong rule for *deciding* what to mask (an ordinary policy completion may
    legitimately repeat the text, which is why `_mask_summaries` goes by branch), but it
    is the right detector for "a summary we know was generated is still trainable". It
    only runs once a tracker record is already unaccounted for.
    """
    texts = [text for text in (summary.strip() for summary in summaries) if text]
    if not texts:
        return 0
    leaked = 0
    for node in trace.nodes:
        if not node.sampled or not any(node.mask):
            continue
        content = content_text(getattr(node.message, "content", None))
        if content and any(text in content for text in texts):
            leaked += sum(node.mask)
    return leaked


def _account_compaction_masking(trace: Trace, records: list[dict]) -> None:
    """Mask the summaries, record the accounting, and fail only on a real leak.

    The program's tracker says how many summaries it generated; the trace says how many
    summarizer completions were actually found and masked. A disagreement is worth a
    metric, not a dead rollout: it costs a completed episode (and its evaluator calls)
    over bookkeeping, and neither a short count nor a zero token count can by itself put
    summary text into the loss — an unmasked *token* can, and that alone is fatal.
    """
    # Sampled before masking: masking clears the summarizer's own tokens, so asking
    # afterwards could mistake a successfully masked trace for an untokenized one.
    had_trainable_tokens = _has_trainable_tokens(trace)
    nodes, tokens, branches = _mask_summaries(trace)
    # A summary the program generated but the branch walk never found. Suspicious, but
    # harmless on its own: a summarizer completion that is absent from the graph trains
    # nothing. Only a *present and still trainable* one leaks.
    unmatched = max(0, len(records) - branches)
    # A trace with no tokens anywhere (the eval relay commits messages without token
    # ids) masks zero tokens even when masking is perfect, so `tokens == 0` is only
    # evidence of a miss when the trace is tokenized at all.
    tokenless = bool(branches) and not had_trainable_tokens
    leaked = (
        _unmasked_summary_tokens(trace, [record["summary"] for record in records])
        if unmatched
        else 0
    )
    anomalies = (
        bool(unmatched)
        + (nodes != branches)
        + bool(branches and not tokens and not tokenless)
    )
    trace.record_metric("skx_compactions", len(records))
    trace.record_metric("skx_compaction_branches", branches)
    trace.record_metric("skx_compaction_nodes_masked", nodes)
    trace.record_metric("skx_compaction_tokens_masked", tokens)
    trace.record_metric(
        "skx_compaction_fallbacks",
        sum(record.get("fallback") is True for record in records),
    )
    trace.record_metric(
        "skx_compaction_truncations",
        sum(record.get("finish_reason") == "length" for record in records),
    )
    trace.record_metric("skx_compaction_summaries_unmatched", unmatched)
    trace.record_metric("skx_compaction_tokenless", int(tokenless))
    trace.record_metric("skx_compaction_masking_anomalies", anomalies)
    trace.record_metric("skx_compaction_summary_tokens_leaked", leaked)
    if anomalies:
        logger.warning(
            "skx: compaction masking accounting mismatch (summaries=%d branches=%d "
            "masked_nodes=%d masked_tokens=%d tokenless=%s); rollout kept",
            len(records),
            branches,
            nodes,
            tokens,
            tokenless,
        )
    if leaked:
        # Unmasked summary tokens would train the policy on summarizer output. No score
        # is worth that. The text match is deliberately not reused to mask the node —
        # identity is not provenance — so the rollout dies instead of being guessed at.
        raise RuntimeError(
            "SKX compaction masking integrity failure: "
            f"summaries={len(records)} branches={branches} masked_nodes={nodes} "
            f"unmasked_summary_tokens={leaked}"
        )


class SkxHarness(Harness[SkxHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    SUPPORTS_MCP = True
    SUPPORTS_RESUME = True
    EXECUTES_CODE = False
    NEEDS_CONTAINER = False

    async def setup(self, runtime: Runtime) -> None:
        await runtime.prepare_uv_script(PROGRAM_SOURCE, self.config.resolved_env)

    async def launch(
        self,
        ctx: ModelContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
        data: TaskData,
    ) -> ProgramResult:
        system_prompt, prompt = self.resolve_prompt(data)
        env = {**self.config.resolved_env}
        args = [
            f"--base-url={endpoint}",
            f"--api-key={secret}",
            f"--model={ctx.model}",
        ]
        if system_prompt:
            args.append(f"--system-prompt={system_prompt}")
        if mcp_urls:
            # The program connects to the tool servers over HTTP; hand it a standard
            # `mcpServers` URL config (the `mcp` client itself comes from the uv deps).
            args.append(
                "--mcp-config="
                + json.dumps(
                    {
                        "mcpServers": {
                            name: {"url": url} for name, url in mcp_urls.items()
                        }
                    }
                )
            )
        if isinstance(prompt, str):
            args.append(f"--prompt={prompt}")
        elif prompt is not None:
            # Base64 images can exceed exec limits, so hand Messages off through a file.
            path = f".vf-initial-messages-{trace.id}.json"
            await runtime.write(
                path,
                json.dumps([message_to_wire(m) for m in prompt]).encode(),
            )
            args.append(f"--initial-messages-file={path}")
        stats_path = f".vf-agent-stats-{trace.id}.json"
        args.append(f"--stats-file={stats_path}")
        args.append(f"--eval-nudges={self.config.eval_contract.max_nudges}")
        tracker_path = ""
        if self.config.compaction.enabled:
            tracker_path = f".vf-compaction-{trace.id}.jsonl"
            args += [
                f"--compact-trigger-tokens={self.config.compaction.trigger_tokens}",
                f"--compact-keep-recent={self.config.compaction.keep_recent_messages}",
                f"--compact-keep-recent-tokens={self.config.compaction.keep_recent_tokens}",
                f"--compact-tracker-file={tracker_path}",
            ]
        program = await runtime.prepare_uv_script(
            PROGRAM_SOURCE, self.config.resolved_env
        )
        result = await runtime.run_program([*program, *args], env)
        try:
            stats = json.loads((await runtime.read(stats_path)).decode(errors="replace"))
        except (FileNotFoundError, json.JSONDecodeError):
            stats = {}
        for key in (
            "repeat_tool_calls",
            "deduped_observations",
            "model_call_attempts",
            "model_call_retries",
            "trusted_evals",
            "eval_nudges",
        ):
            value = stats.get(key)
            trace.record_metric(f"skx_{key}", value if isinstance(value, int) else 0)
        if tracker_path and self.config.compaction.mask_summaries:
            try:
                raw = (await runtime.read(tracker_path)).decode(errors="replace")
            except FileNotFoundError:
                raw = ""
            records = []
            for line in raw.splitlines():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("skx: ignored malformed compaction record")
                    continue
                if isinstance(value, dict) and isinstance(value.get("summary"), str):
                    records.append(value)
            _account_compaction_masking(trace, records)
        return result
