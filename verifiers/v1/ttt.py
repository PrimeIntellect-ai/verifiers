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

import logging
import time
from typing import TYPE_CHECKING, Literal

import httpx
from pydantic_config import BaseConfig

from verifiers.v1.errors import TTTError

if TYPE_CHECKING:
    from verifiers.v1.graph import PendingTurn
    from verifiers.v1.trace import Trace
    from verifiers.v1.types import SamplingConfig

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


class TTTRolloutHook:
    """One rollout's TTT state: fork detection, update calls, version bookkeeping, and the
    session's model/sampling overrides. Created by the `Rollout` when `TTTConfig` is present
    and attached to the `RolloutSession`; driven by the interception server."""

    def __init__(self, config: TTTConfig, trace: "Trace") -> None:
        self.config = config
        self.trace = trace
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
        self.updates: list[dict] = []
        """Per-update records (version, token counts, loss, seconds) — mirrored into
        `trace.info["ttt"]` so results/W&B surface TTT behavior per rollout."""
        self._http: httpx.AsyncClient | None = None

    # -- driven by the interception server ------------------------------------------------

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

    def _path(self, leaf: int) -> list[int]:
        """Node ids from root to `leaf` — the branch that ends at `leaf`."""
        path: list[int] = []
        nid: int | None = leaf
        while nid is not None:
            path.append(nid)
            nid = self.trace.nodes[nid].parent
        path.reverse()
        return path

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
        """One blocking update on an abandoned branch: build the payload, call the service
        (gradient step(s) + checkpoint + engine adapter load), then switch this rollout's
        model to the adapter and salt the prefix cache for the new version."""
        token_ids, loss_mask = self.payload(path, shared)
        if not token_ids:
            raise TTTError(
                "ttt: the abandoned branch carries no token ids — TTT requires the "
                "renderer (train) client (`--client.type train`), not the eval relay."
            )
        if not any(loss_mask):
            return  # nothing new to train (e.g. a fork right after a fork)
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
        self.updates.append(record)
        info = self.trace.info.setdefault("ttt", {"adapter": self.adapter_name})
        info["updates"] = self.updates
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
