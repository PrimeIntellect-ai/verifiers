"""Test-time training (TTT): online LoRA updates at compaction boundaries.

A rollout under TTT trains a per-rollout LoRA adapter as it goes: every time the harness
rewrites its context (a compaction — the only rewrite the compacting harness performs), the
just-abandoned branch is sent to an external TTT service, which takes gradient step(s) on a
per-rollout adapter, saves a versioned checkpoint, and (re)loads the adapter into the
inference engine. From then on the rollout's model calls use the adapter, so what fell out
of the attention window lives on in the weights.

The invariant everything downstream builds on: **every branch is sampled under exactly one
adapter version**. Updates fire only at branch forks (detected here, passively, from the
trace graph — the harness knows nothing about TTT), each committed node is stamped with the
version it was sampled under (`MessageNode.ttt_version`), and each update is checkpointed —
so an RL trainer can later replay any branch with the exact adapter it was sampled under.

The hook lives on the `RolloutSession` and is driven by the interception server:

- `on_turn_prepared(turn)` — before each model call: if the resolved prefix does not extend
  the previous leaf, the previous branch was abandoned → run one update on it (blocking;
  correctness over latency), then switch the session's model to the adapter and salt the
  prefix cache for the new version.
- `after_commit(trace, num_nodes_before)` — after each committed turn: stamp the new nodes
  with the current version and advance the leaf pointer.
- `finalize_rollout()` — after the harness returns: optionally train the final branch.
- `aclose()` — always: release the adapter (service + engine); checkpoints stay on disk.

TTT requires the renderer (train) client: updates consume the exact token ids the engine
saw, so a relay (eval) client — which carries no token ids — fails loudly.
"""

import asyncio
import logging
import re
import time
from typing import TYPE_CHECKING, Literal

import httpx
from pydantic_config import BaseConfig

from verifiers.v1.errors import TTTError
from verifiers.v1.types import SamplingConfig

if TYPE_CHECKING:
    from verifiers.v1.clients import RolloutContext
    from verifiers.v1.graph import PendingTurn
    from verifiers.v1.trace import Trace

logger = logging.getLogger(__name__)


class TTTConfig(BaseConfig):
    """Test-time training for this env's rollouts (None = off). Triggering lives here;
    training hyperparameters (LR, optimizer, rank, steps) live on the TTT service."""

    base_url: str
    """The TTT service (e.g. `http://localhost:8092`). It owns the base model copy, the
    per-rollout adapters + optimizers, checkpointing, and the engine adapter loads."""
    enabled: bool = True
    """Master switch: False leaves rollouts untouched (no hook, no updates) so an arm can
    be toggled in config without deleting the block."""
    loss_scope: Literal["all", "sampled"] = "all"
    """Which tokens of an abandoned branch get loss. `all` (default): every not-yet-trained
    token, including tool outputs and prompts — this is memory formation for context
    extension, not policy learning. `sampled`: only model-sampled tokens (ablation)."""
    train_final_branch: bool = False
    """Also run an update on the final branch when the rollout ends. Off by default: no
    token is ever sampled under the resulting version, so it only matters for post-hoc
    analysis of the final adapter."""
    update_timeout: float = 600.0
    """Max seconds to wait for one `/update` call (a blocking gradient step + checkpoint +
    engine load)."""
    adapter_prefix: str = "ttt"
    """Adapter names are `{prefix}-{trace.id}` — unique per rollout."""
    qa: "QAConfig | None" = None
    """Cartridges-style Q&A at compaction (None = off): before each update, generate
    question–answer pairs about the abandoned branch (with the full branch in context) and
    train the adapter on those pairs instead of (or in addition to) the raw branch."""


DEFAULT_QA_SEEDS = [
    "the concrete facts, data, names, numbers, paths, and exact values established so far",
    "the approaches, commands, or strategies that worked, and why",
    "the approaches that failed or turned out to be dead ends, and why",
    "theories, hypotheses, or open questions about the problem, and the evidence for or against each",
    "the setup of the task: the goal, the constraints, the tools available, and the current state of progress",
    "lessons about how the available tools behave: quirks, failure modes, and how to use them effectively",
]

# The generation instruction: one seed focuses one call; the model writes BOTH the questions
# and the answers, several per call, in a tagged structure we can extract robustly (JSON
# breaks on long free text with quotes/newlines; tag scanning salvages well-formed items
# from a partially malformed generation). The self-containment contract is the key quality
# lever: pairs are later trained WITHOUT this conversation in context, so a question that
# says "the conversation above" has no retrieval key. Lessons are folded into the same
# question/answer shape with the *trigger condition* as the question — the future context
# matching the trigger is what cues the memory.
QA_GENERATION_PROMPT = """\
You are creating a study set about the conversation above, focused on: {seed}.

Write {num_items} question-answer items. Each item must teach something from the
conversation that is worth remembering.

Rules:
- SELF-CONTAINED: the question must be fully understandable by someone who never saw this
  conversation. Name the task, entities, files, tools, and values explicitly. Never write
  "the conversation above", "this task", "the error", or similar references.
- For factual items, ask for the specific fact and answer it precisely.
- For lessons (what worked, what failed, tool behavior), phrase the question as the
  situation in which the lesson applies ("When doing X and Y happens, what should you
  do?") and the answer as the lesson, including when it is valid and when it is not.
- Answers must be correct and supported by the conversation.

Format each item exactly like this:

<item>
<type>qa or lesson</type>
<question>...</question>
<answer>...</answer>
</item>"""


def parse_qa_items(text: str) -> list[dict]:
    """Extract `<item>` blocks from a generation: `[{type, question, answer}]`. Malformed
    blocks (missing question/answer) are dropped; a missing/unknown type defaults to "qa"."""
    items: list[dict] = []
    for block in re.findall(r"<item>(.*?)</item>", text, re.DOTALL):

        def field(tag: str) -> str:
            m = re.search(rf"<{tag}>(.*?)</{tag}>", block, re.DOTALL)
            return m.group(1).strip() if m else ""

        question, answer = field("question"), field("answer")
        if not question or not answer:
            continue
        kind = field("type").lower()
        items.append(
            {
                "type": kind if kind in ("qa", "lesson") else "qa",
                "question": question,
                "answer": answer,
            }
        )
    return items


def dedup_items(items: list[dict], threshold: float = 0.85) -> list[dict]:
    """Drop near-duplicate questions (normalized token-overlap Jaccard >= threshold),
    keeping the first occurrence — repeated seeds and overlapping foci produce them."""
    kept: list[dict] = []
    kept_tokens: list[set[str]] = []
    for item in items:
        tokens = set(re.findall(r"[a-z0-9]+", item["question"].lower()))
        duplicate = any(
            tokens and other and len(tokens & other) / len(tokens | other) >= threshold for other in kept_tokens
        )
        if not duplicate:
            kept.append(item)
            kept_tokens.append(tokens)
    return kept


class QAConfig(BaseConfig):
    """Q&A generation + training at compaction. Each compaction runs `num_generations`
    parallel generations (one seed focus each) against the full abandoned branch; the model
    writes both the questions and the answers (plus trigger-phrased lessons), several per
    call, extracted structurally. The exchanges are committed to the trace as `ttt_qa`
    branches — real sampled tokens, so RL trains the generation behavior itself (good
    lessons get reinforced through the rollout's advantage) — but stay outside
    `RolloutLimits` and the trace's turn/token metrics; `max_tokens` below is their own
    budget. The extracted pairs are what the adapter trains on."""

    num_generations: int = 4
    """Parallel QA generations per compaction (one seed each, cycled from `seeds`)."""
    items_per_generation: int = 4
    """Q&A items requested per generation (`{num_items}` in the prompt)."""
    seeds: list[str] = DEFAULT_QA_SEEDS
    """Focus seeds — `{seed}` in `QA_GENERATION_PROMPT`. Diversity comes from the seed
    (focus) times the model's own question choices within each focus."""
    generation_prompt: str = QA_GENERATION_PROMPT
    """The instruction template (must keep the `{seed}` / `{num_items}` placeholders and
    the `<item>` output format)."""
    max_tokens: int | None = 2048
    """Completion budget per QA generation (None = the rollout sampling's own value)."""
    temperature: float | None = None
    """Sampling temperature for QA generations (None = the rollout sampling's own)."""
    dedup_threshold: float | None = 0.85
    """Near-duplicate question filter (token-overlap Jaccard); None disables."""
    also_train_rollout: bool = False
    """Also train on the raw abandoned branch, in the same update. Default off: the plan's
    Q&A arm trains on the Q&A dataset *instead of* the rollout itself."""
    recycle_to_policy: bool = False
    """Recycle the Q&A pairs into the *policy's* main weights during RL: the trainer side
    (prime-rl) renders each pair (system-prompt-conditioned) and routes it to the ce loss
    component — the "one SFT step after the RL update" from the plan, riding the same
    training batch. Consumed by the prime-rl orchestrator."""


class TTTRolloutHook:
    """One rollout's TTT state: fork detection, update calls, version bookkeeping, and the
    session's model/sampling overrides. Created by the `Rollout` when `TTTConfig` is present
    and attached to the `RolloutSession`; driven by the interception server."""

    def __init__(self, config: TTTConfig, trace: "Trace", ctx: "RolloutContext | None" = None) -> None:
        self.config = config
        self.trace = trace
        self.ctx = ctx
        """The rollout's client/model/sampling — needed only for Q&A generation (`qa`
        set); the plain update path never touches the model."""
        if config.qa is not None and ctx is None:
            raise ValueError("ttt: qa generation needs the rollout context (ctx).")
        self.adapter_name = f"{config.adapter_prefix}-{trace.id}"
        self.version = 0
        """The adapter version the *next* sampled token runs under: 0 = base model (no
        adapter yet); k = after the k-th update."""
        self.last_leaf: int | None = None
        """Node id of the last committed assistant node — the leaf the next turn's prompt
        must extend, else its branch was abandoned."""
        self.seen: set[int] = set()
        """Node ids already carried in an update payload (as loss or context). A node is
        trained at most once; shared prefixes (the system prompt) stay context forever."""
        self.model_override: str | None = None
        self.sampling_override: "SamplingConfig | None" = None
        self.tools_wire: list[dict] | None = None
        """The tool schemas from the harness's most recent request (captured by the
        interception server via `capture_request`). Used twice: QA generations advertise
        the same tools (identical rendered system block → clean branching), and the
        `/update` payload ships them so the service renders standalone QA pairs with the
        same system+tools conditioning the rollout saw."""
        self.updates: list[dict] = []
        """Per-update records (version, token counts, loss, seconds) — mirrored into
        `trace.info["ttt"]` so results/W&B surface TTT behavior per rollout."""
        self._http: httpx.AsyncClient | None = None

    # -- driven by the interception server ------------------------------------------------

    def capture_request(self, body: dict) -> None:
        """Observe a harness request body: remember the advertised tool schemas (see
        `tools_wire`). Tool sets are stable within a rollout, so last-write-wins is fine."""
        tools = body.get("tools")
        if tools:
            self.tools_wire = tools

    async def on_turn_prepared(self, turn: "PendingTurn") -> None:
        """Before a model call: detect a branch fork and run one update on the abandoned
        branch. A turn that extends the previous leaf (append-only continuation, an SDK
        retry, a user-sim exchange) reuses it as its prefix tail; anything else means the
        harness rewrote its context — under the compacting harness, a compaction."""
        if self.last_leaf is None:
            return  # first turn — nothing committed yet
        if turn.prefix_node_ids and turn.prefix_node_ids[-1] == self.last_leaf:
            return  # extends the leaf — same branch
        if not turn.tail:
            # The whole prompt matched existing nodes: a replay of an already-committed
            # path (an SDK retry re-sending a committed turn), not a context rewrite — a
            # rewrite always carries at least one new message (the summary). Don't update;
            # the re-sampled turn commits as a dead-end branch under the current version.
            return
        abandoned = self._path(self.last_leaf)
        shared = set(turn.prefix_node_ids)
        await self._update(abandoned, shared)

    def after_commit(self, trace: "Trace", num_nodes_before: int) -> None:
        """After a committed turn: stamp the newly created nodes with the current version
        (the version their tokens were prefilled/sampled under) and advance the leaf."""
        for node in trace.nodes[num_nodes_before:]:
            node.ttt_version = self.version
        self.last_leaf = len(trace.nodes) - 1

    # -- driven by the Rollout -------------------------------------------------------------

    async def finalize_rollout(self) -> None:
        """After the harness returns: optionally train the final branch (no context rewrite
        follows it, so nothing is ever sampled under the resulting version)."""
        if not self.config.train_final_branch or self.last_leaf is None:
            return
        await self._update(self._path(self.last_leaf), shared=set())

    async def aclose(self) -> None:
        """Release the rollout's adapter (service optimizer state + engine slot); its
        checkpoints stay on disk for RL replay. Always called (rollout `finally`)."""
        if self.version == 0 and self._http is None:
            return  # never updated, nothing to release
        try:
            response = await self._client().post(
                f"{self.config.base_url}/release",
                json={"rollout_id": self.trace.id, "adapter_name": self.adapter_name},
                timeout=60.0,
            )
            response.raise_for_status()
        except Exception:
            logger.warning(
                "ttt: release failed for %s (adapter %s)",
                self.trace.id,
                self.adapter_name,
                exc_info=True,
            )
        finally:
            if self._http is not None:
                await self._http.aclose()
                self._http = None

    # -- internals ---------------------------------------------------------------------------

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient()
        return self._http

    def _system_prompt(self) -> str | None:
        """The rollout's system prompt text (the trace's root system node), for the
        service's standalone QA rendering."""
        for node in self.trace.nodes:
            if node.parent is None and node.message.role == "system":
                content = node.message.content
                return content if isinstance(content, str) else None
        return None

    def _path(self, leaf: int) -> list[int]:
        """Node ids from root to `leaf` — the branch that ends at `leaf`."""
        path: list[int] = []
        nid: int | None = leaf
        while nid is not None:
            path.append(nid)
            nid = self.trace.nodes[nid].parent
        path.reverse()
        return path

    async def _generate_qa(self, path: list[int]) -> list[dict]:
        """Generate the Q&A items for an abandoned branch: `num_generations` parallel
        calls, each with the FULL branch in context plus one seeded generation instruction
        (the model writes both the questions and the answers, several per call), through
        the same client the rollout uses, under the current pre-update adapter.

        Each exchange is COMMITTED to the trace as a `ttt_qa`-tagged branch: real sampled
        tokens under a known adapter version, so RL trains the generation behavior itself
        (the rollout's advantage reinforces lessons that helped). The tag keeps them out of
        `RolloutLimits` and the trace's turn/token metrics — they run on `qa.max_tokens`.

        Returns the structurally extracted, deduplicated `[{type, question, answer}]`
        items; the update trains the adapter on them rendered standalone (branch context
        absent, system prompt + tools present), Cartridges-style, so the knowledge must
        come from the weights rather than a context-conditioned mapping."""
        from verifiers.v1 import graph
        from verifiers.v1.dialects import ChatDialect
        from verifiers.v1.dialects.chat import message_to_wire

        assert self.config.qa is not None and self.ctx is not None
        qa = self.config.qa
        dialect = ChatDialect()
        branch_messages = [self.trace.nodes[nid].message for nid in path]
        branch_wire = [message_to_wire(m) for m in branch_messages]
        sampling_data = self.ctx.sampling.model_dump(exclude_none=True)
        if qa.max_tokens is not None:
            sampling_data["max_tokens"] = qa.max_tokens
        if qa.temperature is not None:
            sampling_data["temperature"] = qa.temperature
        sampling = self.salted_sampling(SamplingConfig(**sampling_data))
        model = self.model_override or self.ctx.model

        from verifiers.v1.types import UserMessage

        async def generate(prompt_text: str):
            # A prepared turn (same prefix walk the interception server does) lets the
            # renderer client bridge to the branch's exact stored tokens instead of
            # re-rendering — the QA branch shares the abandoned branch's prefix verbatim.
            prompt = [*branch_messages, UserMessage(content=prompt_text)]
            turn = graph.prepare_turn(self.trace, prompt)
            body = {
                "messages": [*branch_wire, {"role": "user", "content": prompt_text}],
                **({"tools": self.tools_wire} if self.tools_wire else {}),
            }
            response = await self.ctx.client.get_response(
                dialect, body, model, sampling, session_id=self.trace.id, turn=turn
            )
            return turn, response

        prompts = [
            qa.generation_prompt.format(seed=qa.seeds[i % len(qa.seeds)], num_items=qa.items_per_generation)
            for i in range(qa.num_generations)
        ]
        generations = await asyncio.gather(*(generate(p) for p in prompts))

        # Commit each exchange as a tagged branch (sequentially — graph mutation), then
        # extract + dedup the items across all generations.
        items: list[dict] = []
        for turn, response in generations:
            before = len(self.trace.nodes)
            turn.commit(response)
            for node in self.trace.nodes[before:]:
                node.ttt_qa = True
                node.ttt_version = self.version
            items.extend(parse_qa_items(response.message.content or ""))
        if qa.dedup_threshold is not None:
            items = dedup_items(items, qa.dedup_threshold)
        return items

    def payload(self, path: list[int], shared: set[int]) -> tuple[list[int], list[bool]]:
        """The update payload for an abandoned branch: its flat token sequence and a
        per-token loss mask. Loss goes to not-yet-trained nodes (`seen`) that are not shared
        with the new branch (`shared` — the fork turn's reused prefix, e.g. the system
        prompt, which is context in every branch and never trained); within a node,
        `loss_scope` picks all tokens or only sampled ones. Everything else is context."""
        token_ids: list[int] = []
        loss_mask: list[bool] = []
        for nid in path:
            node = self.trace.nodes[nid]
            in_loss = nid not in shared and nid not in self.seen
            token_ids.extend(node.token_ids)
            if not in_loss:
                loss_mask.extend([False] * len(node.token_ids))
            elif self.config.loss_scope == "sampled":
                loss_mask.extend(node.mask)
            else:
                loss_mask.extend([True] * len(node.token_ids))
        return token_ids, loss_mask

    async def _update(self, path: list[int], shared: set[int]) -> None:
        """One blocking update on an abandoned branch: build the payload (plus, when
        configured, the Q&A pairs generated with the branch still in context), call the
        service (gradient step(s) + checkpoint + engine adapter load), then switch this
        rollout's model to the adapter and salt the prefix cache for the new version."""
        token_ids, loss_mask = self.payload(path, shared)
        if not token_ids:
            raise TTTError(
                "ttt: the abandoned branch carries no token ids — TTT requires the "
                "renderer (train) client (`--client.type train`), not the eval relay."
            )
        if not any(loss_mask):
            return  # nothing new to train (e.g. a fork right after a fork)
        qa_pairs: list[dict] | None = None
        train_rollout = True
        if self.config.qa is not None:
            try:
                qa_pairs = await self._generate_qa(path)
            except TTTError:
                raise
            except Exception as e:
                raise TTTError(f"ttt: qa generation failed: {type(e).__name__}: {e}") from e
            train_rollout = self.config.qa.also_train_rollout
            if not qa_pairs and not train_rollout:
                raise TTTError(
                    "ttt: qa generation produced no usable pairs and `qa.also_train_rollout` is off — nothing to train."
                )
        seq_no = self.version + 1
        start = time.perf_counter()
        try:
            response = await self._client().post(
                f"{self.config.base_url}/update",
                json={
                    "rollout_id": self.trace.id,
                    "adapter_name": self.adapter_name,
                    "token_ids": token_ids,
                    "loss_mask": loss_mask,
                    "seq_no": seq_no,
                    "qa_pairs": qa_pairs,
                    "train_rollout": train_rollout,
                    # QA conditioning: the service renders each standalone pair as
                    # [system, question, answer] with these tools, loss on the answer only
                    # — the same system+tools context the rollout itself ran under, so
                    # tool lessons are learned next to the tool descriptions.
                    "system_prompt": self._system_prompt() if qa_pairs else None,
                    "tools": self.tools_wire if qa_pairs else None,
                },
                timeout=self.config.update_timeout,
            )
            response.raise_for_status()
            result = response.json()
        except TTTError:
            raise
        except Exception as e:
            raise TTTError(f"ttt: update {seq_no} failed: {type(e).__name__}: {e}") from e
        if result.get("version") != seq_no:
            raise TTTError(
                f"ttt: service applied version {result.get('version')} for update "
                f"{seq_no} — adapter state is out of sync, aborting the rollout."
            )
        self.seen.update(path)
        self.seen.update(shared)
        self.version = seq_no
        self.model_override = self.adapter_name
        self.sampling_override = None  # rebuilt lazily per ctx sampling (see salted_sampling)
        record = {
            "version": seq_no,
            "num_tokens": len(token_ids),
            "num_loss_tokens": sum(loss_mask),
            "loss": result.get("loss"),
            "seconds": round(time.perf_counter() - start, 3),
        }
        if result.get("ckpt_path"):
            record["ckpt_path"] = result["ckpt_path"]
        if qa_pairs is not None:
            record["num_qa_pairs"] = len(qa_pairs)
            record["qa_pairs"] = qa_pairs
            record["trained_rollout"] = train_rollout
        self.updates.append(record)
        info = self.trace.info.setdefault("ttt", {"adapter": self.adapter_name})
        info["updates"] = self.updates
        if qa_pairs is not None:
            # QA conditioning for downstream consumers (policy recycling renders pairs
            # with the same system+tools frame the adapter training used).
            info["system_prompt"] = self._system_prompt()
            info["tools"] = self.tools_wire
        logger.info(
            "ttt: update %d applied for %s (%d/%d loss tokens, loss=%s, %.2fs)",
            seq_no,
            self.trace.id,
            record["num_loss_tokens"],
            record["num_tokens"],
            record["loss"],
            record["seconds"],
        )

    def salted_sampling(self, sampling: "SamplingConfig") -> "SamplingConfig":
        """`sampling` with a per-version cache salt. Mandatory once an adapter is live: the
        engine reloads the adapter in place under the same name (same lora id), so without a
        fresh salt the prefix cache would serve KV computed under the old weights."""
        if self.version == 0:
            return sampling
        data = sampling.model_dump(exclude_none=True)
        extra = dict(data.get("extra_body") or {})
        base_salt = extra.get("cache_salt")
        salt = f"{self.adapter_name}-v{self.version}"
        extra["cache_salt"] = f"{base_salt}:{salt}" if base_salt else salt
        data["extra_body"] = extra
        return type(sampling)(**data)
