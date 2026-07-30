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
from verifiers.v1.types import AssistantMessage

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
    mask_summaries: bool = True


class SkxHarnessConfig(HarnessConfig):
    compaction: SkxCompaction = SkxCompaction()


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
        tracker_path = ""
        if self.config.compaction.enabled:
            tracker_path = f".vf-compaction-{trace.id}.jsonl"
            args += [
                f"--compact-trigger-tokens={self.config.compaction.trigger_tokens}",
                f"--compact-keep-recent={self.config.compaction.keep_recent_messages}",
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
            nodes, tokens, branches = _mask_summaries(trace)
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
            if len(records) > branches or nodes != branches or (branches and not tokens):
                raise RuntimeError(
                    "SKX compaction masking integrity failure: "
                    f"summaries={len(records)} branches={branches} masked_nodes={nodes}"
                )
        return result
