"""ReplayHarness — rollout-time sampling for an online (growing) replay buffer.

The offline path materializes tasks in ``ReplayTaskset.load_tasks``, which runs once at
env-server start and so can't see rollouts written later. This harness instead samples a stored
trace + resume point from the *live* buffer on each rollout, so this run's own rollouts are
replayed as the buffer fills. It restores the resume point's sandbox snapshot, seeds the default
chat loop with the replay prefix (``root->node``), and stashes provenance in ``trace.info`` so
``ReplayTaskset.score`` can reuse the original verifier.

It reuses the default harness's program (a growing-message-list chat loop) and seeds it via the
same ``INITIAL_MESSAGES`` channel the default harness uses for a Messages prompt.
"""

from __future__ import annotations

import glob
import json
import random
from pathlib import Path

from verifiers.v1.clients import RolloutContext
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.harnesses.default.harness import (
    PROGRAM_SOURCE,
    DefaultHarness,
    DefaultHarnessConfig,
)
from verifiers.v1.runtimes import ProgramResult, Runtime
from verifiers.v1.trace import Trace, WireTrace

from replay.selector import (
    DEFAULT_FOLLOWUP,
    DEFAULT_KINDS,
    build_seed,
    resume_points,
    snapshot_ref_of,
)


class ReplayHarnessConfig(DefaultHarnessConfig):
    buffer_glob: str = ""
    """Glob of stored-rollout JSONL files to sample from (the live, possibly growing, buffer)."""
    kinds: list[str] = DEFAULT_KINDS
    """Which replay kinds to sample (see ReplayTaskset; add ``"judge"`` to opt in)."""
    followup: str = DEFAULT_FOLLOWUP
    """The user turn appended for ``recheck`` points."""


class ReplayHarness(DefaultHarness):
    """Subclasses the default harness (its chat-loop program) but seeds from a sampled buffer
    rollout instead of ``task.prompt``."""

    SUPPORTS_MESSAGE_PROMPT = True

    async def launch(
        self,
        ctx: RolloutContext,
        trace: Trace,
        runtime: Runtime,
        endpoint: str,
        secret: str,
        mcp_urls: dict[str, str],
    ) -> ProgramResult:
        rng = random.Random(trace.id)  # deterministic per rollout, varies across rollouts
        sample = self._sample(rng)
        if sample is None:  # buffer empty (warmup) or no matching resume points yet
            trace.stop("replay_buffer_empty")
            return ProgramResult(exit_code=0, stdout="", stderr="")
        src, point = sample

        ref = snapshot_ref_of(src, point["node"])
        if ref is not None:  # exec/sandbox replay; skeleton refs are None -> skip
            await runtime.restore(ref)

        # Stash provenance so ReplayTaskset.score can reuse the original verifier.
        trace.info["replay"] = {
            "source_id": src.id,
            "resume_node": point["node"],
            "kind": point["kind"],
            "original_task": src.task.model_dump(),
            "original_reward": src.reward,
        }

        # Seed the default chat loop with the replay prefix (mirror DefaultHarness.launch).
        seed = build_seed(src, point, self.config.followup)
        env = {**self.config.env}
        env["INITIAL_MESSAGES"] = json.dumps([message_to_wire(m) for m in seed])
        args = [f"--base-url={endpoint}", f"--api-key={secret}", f"--model={ctx.model}"]
        if mcp_urls:
            args.append(
                "--mcp-config="
                + json.dumps(
                    {"mcpServers": {name: {"url": url} for name, url in mcp_urls.items()}}
                )
            )
        program = await runtime.prepare_uv_script(PROGRAM_SOURCE, self.config.env)
        return await runtime.run_program([*program, *args], env)

    def _sample(self, rng: random.Random) -> tuple[Trace, dict] | None:
        """Scan the live buffer in random order; return the first (trace, resume point) found.
        Re-globs every rollout, so files written after env-server start are included."""
        kinds = set(self.config.kinds)
        files = sorted(glob.glob(self.config.buffer_glob))
        rng.shuffle(files)
        for path in files:
            try:
                lines = Path(path).read_text().splitlines()
            except OSError:
                continue
            rng.shuffle(lines)
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                src = WireTrace.model_validate(json.loads(line))
                points = resume_points(src, kinds=kinds)
                if points:
                    return src, rng.choice(points)
        return None


__all__ = ["ReplayHarness", "ReplayHarnessConfig"]
