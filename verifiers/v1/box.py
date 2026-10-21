"""The owner of a box `Agent.provision` hands out.

A provisioned box is borrowed by the runs placed into it (`run(runtime=box)`), possibly
by several seats at once (a control agent and its reviewer). Its `Box` owner, attached
to the runtime, is what lets those borrowed runs be saved and recovered:

- **Checkpoints.** A borrowed run checkpoints the box through its owner, one checkpoint
  at a time across every seat on it: per call when the run's `save = "per_call"` or the
  provisioning agent's is, and on `Interaction.checkpoint()`. Each checkpoint is
  recorded on the run's own trace, matching that run's conversation.
- **Restore.** The owner restores the box, never a borrower: `replace()` starts a fresh
  box from the latest checkpoint any seat took and swaps it in under the same backend
  object, so every seat sharing the box (and every view of it) sees the restored box. An
  interaction retry on a borrowed box goes through it (`AgentConfig.interaction_retries`).
  After a crash, a durable `agent.provision(task)` restores the box from the latest
  checkpoint its runs recorded or started on (`verifiers.v1.durable`), and their
  durable runs resume against it.

- **Live continuation.** A run that continues a trace whose run ran on this same box
  (`run(resume=trace, runtime=box)`, not replaced since) finds the task as that run left
  it, processes included: it runs neither `Task.setup` nor `Task.restore` (`hosted`).
  This is how a conversation takes turns on one box (`verifiers.v1.conversation`).

Seats that take turns on a box resume exactly; seats that change it at the same time
resume each on its own latest checkpoint, with the box as of the latest of all.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

from verifiers.v1.durable import BOX_LABEL, Durable
from verifiers.v1.runtimes import (
    Runtime,
    RuntimeConfig,
    make_backend,
    provision_backend,
    restore_config,
)
from verifiers.v1.trace import Checkpoint

logger = logging.getLogger(__name__)

_ATTR = "_vf_box"


class Box:
    """The owner of one provisioned box (see the module docs)."""

    def __init__(
        self,
        runtime: Runtime,
        config: RuntimeConfig,
        *,
        save: bool = False,
        key: str | None = None,
        fingerprint: str | None = None,
        ref: str | None = None,
        durable: Durable | None = None,
        restored: Checkpoint | None = None,
    ) -> None:
        self.runtime = runtime
        self.config = config
        """The runtime config the box was provisioned from, without its checkpoint."""
        self.save = save
        """Borrowed runs checkpoint the box per call (the provisioning agent's
        `save = "per_call"`), whatever their own `save`."""
        self.key = key
        self.fingerprint = fingerprint
        self.ref = ref
        """The box's durable address (`durable.box_ref`), when it has a key."""
        self.durable = durable
        self.restored = restored
        """The checkpoint the current box started from, if any."""
        self.latest = restored
        """The latest checkpoint of the box: taken through this owner, or the one it
        started from."""
        self.generation = 0
        """How many times `replace` swapped the box."""
        self.hosted: dict[str, Any] = {}
        """The traces whose runs set up or continued on the current box (this
        generation), each with its task's config as that run left it. A continuation of
        one of them finds the task as that run left it, its config included (a tool
        server setup started in the box keeps its url): it runs neither `Task.setup`
        nor `Task.restore` (`Rollout.open`)."""
        self._lock = asyncio.Lock()
        # On the backend: every view of the box (`with_env`, `with_user`) shares it.
        runtime.backend.__dict__[_ATTR] = self

    @property
    def labels(self) -> dict[str, Any]:
        """The labels every run on a durable box carries (`durable.BOX_LABEL`), as of
        the run's start: the box's key, fingerprint and ref, and the checkpoint the box
        was at then (its `latest`), so the run that moved last names the box's latest
        checkpoint even before it takes one of its own (`durable.box_checkpoint`)."""
        if self.key is None:
            return {}
        label: dict[str, Any] = {
            "key": self.key,
            "fingerprint": self.fingerprint,
            "ref": self.ref,
        }
        if self.latest is not None:
            label["checkpoint"] = {"id": self.latest.id, "time": self.latest.time}
        return {BOX_LABEL: label}

    async def checkpoint(self) -> Checkpoint:
        """Checkpoint the box, one at a time across its seats. The caller records the
        returned id on its run's trace, with the turns its conversation is at."""
        async with self._lock:
            checkpoint_id = await self.runtime.backend.checkpoint()
            self.latest = Checkpoint(id=checkpoint_id, turns=0, time=time.time())
            return self.latest

    async def replace(self, generation: int, source: Checkpoint | None = None) -> bool:
        """Start a fresh box from the latest checkpoint (or `source`) and swap it in for
        the box of `generation` (what the failed run ran on); the old box is torn down.
        A box another seat already replaced is left as it is. False when there is no
        checkpoint to restore from."""
        async with self._lock:
            if self.generation != generation:
                return True
            source = source or self.latest
            if source is None:
                return False
            logger.warning(
                "box %s: restoring it from checkpoint %s on a fresh box",
                self.runtime.name,
                source.id,
            )
            backend = self.runtime.backend
            fresh = make_backend(restore_config(self.config, source.id))
            fresh.env = dict(backend.env)
            try:
                await fresh.start()
            except BaseException:
                await fresh.stop()
                raise
            old = backend.adopt(fresh)
            backend.__dict__[_ATTR] = self
            self.generation += 1
            self.hosted = {}
            self.restored = self.latest = Checkpoint(
                id=source.id, turns=0, restored=True, time=time.time()
            )
        with contextlib.suppress(Exception):
            await old.stop()
        return True


def box_of(runtime: Runtime | None) -> Box | None:
    """The owner of a box `Agent.provision` handed out (any view of it); None for a
    box made some other way."""
    backend = getattr(runtime, "backend", None)
    return getattr(backend, "__dict__", {}).get(_ATTR)


@asynccontextmanager
async def provision_box(
    config: RuntimeConfig,
    *,
    env: Mapping[str, str] | None = None,
    checkpoint: Checkpoint | None = None,
    save: bool = False,
    key: str | None = None,
    fingerprint: str | None = None,
    ref: str | None = None,
    durable: Durable | None = None,
) -> AsyncIterator[Runtime]:
    """Provision a box from `config` (started from `checkpoint` when given) with its
    `Box` owner, and tear it down on exit."""
    start = restore_config(config, checkpoint.id) if checkpoint is not None else config
    async with provision_backend(start, env=dict(env or {})) as backend:
        runtime = backend.bind()
        Box(
            runtime,
            config,
            save=save and backend.supports_checkpoint,
            key=key,
            fingerprint=fingerprint,
            ref=ref,
            durable=durable,
            restored=checkpoint,
        )
        yield runtime
