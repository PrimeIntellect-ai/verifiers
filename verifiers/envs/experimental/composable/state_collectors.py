"""State collectors for composable harness artifacts."""

from __future__ import annotations

import logging
import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING

from verifiers.types import State

if TYPE_CHECKING:
    from verifiers.envs.experimental.composable.composable_env import ComposableEnv

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GitPatchCollector:
    """Collect a git diff of agent edits relative to the post-setup tree."""

    state_key: str = "agent_patch"
    timeout: int = 120
    git_path: str = "/usr/bin/git"

    @property
    def _base_tree_key(self) -> str:
        return f"_{self.state_key}_base_tree"

    @property
    def _workdir_key(self) -> str:
        return f"_{self.state_key}_workdir"

    def snapshot_command(self) -> str:
        """Return a shell command that snapshots the current worktree as a tree."""
        return _git_tree_command(self.git_path)

    def diff_command(self, base_tree: str) -> str:
        """Return a shell command that diffs the current worktree against a tree."""
        return _git_diff_command(self.git_path, base_tree)

    async def post_sandbox_setup(self, env: ComposableEnv, state: State) -> None:
        sandbox_id = state.get("sandbox_id")
        if not sandbox_id:
            return
        workdir = env.taskset.get_workdir(state.get("info") or {})
        try:
            result = await env.sandbox_client.execute_command(
                sandbox_id,
                self.snapshot_command(),
                working_dir=workdir,
                timeout=self.timeout,
            )
        except Exception as exc:
            logger.warning("Failed to capture git patch baseline: %s", exc)
            return
        if result.exit_code != 0:
            output = ((result.stderr or "") + (result.stdout or ""))[:500]
            logger.warning(
                "Failed to capture git patch baseline (exit=%s): %s",
                result.exit_code,
                output,
            )
            return
        base_tree = (result.stdout or "").strip()
        if base_tree:
            state[self._base_tree_key] = base_tree
            state[self._workdir_key] = workdir

    async def post_rollout(self, env: ComposableEnv, state: State) -> None:
        state.setdefault(self.state_key, "")
        sandbox_id = state.get("sandbox_id")
        base_tree = state.get(self._base_tree_key)
        if not sandbox_id or not isinstance(base_tree, str) or not base_tree:
            return
        workdir = state.get(self._workdir_key) or env.taskset.get_workdir(
            state.get("info") or {}
        )
        try:
            result = await env.sandbox_client.execute_command(
                sandbox_id,
                self.diff_command(base_tree),
                working_dir=workdir,
                timeout=self.timeout,
            )
        except Exception as exc:
            logger.warning("Failed to collect git patch: %s", exc)
            return
        if result.exit_code != 0:
            output = ((result.stderr or "") + (result.stdout or ""))[:500]
            logger.warning(
                "Failed to collect git patch (exit=%s): %s",
                result.exit_code,
                output,
            )
            return
        state[self.state_key] = result.stdout or ""


def _git_tree_command(git_path: str) -> str:
    return _git_command(
        git_path,
        """
"$git" -c core.fileMode=false add -A -- .
"$git" -c core.fileMode=false write-tree
""",
    )


def _git_diff_command(git_path: str, base_tree: str) -> str:
    return _git_command(
        git_path,
        f"""
"$git" -c core.fileMode=false add -A -- .
current_tree="$("$git" -c core.fileMode=false write-tree)"
"$git" -c core.fileMode=false diff --binary --full-index --text {shlex.quote(base_tree)} "$current_tree"
""",
    )


def _git_command(git_path: str, body: str) -> str:
    script = f"""\
set -euo pipefail
git_path={shlex.quote(git_path)}
if [ -x "$git_path" ]; then
  git="$git_path"
else
  git="$(PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin command -v git || true)"
fi
[ -n "$git" ] || exit 0
"$git" rev-parse --is-inside-work-tree >/dev/null 2>&1 || exit 0
tmp_index="$(mktemp)"
trap 'rm -f "$tmp_index"' EXIT
export GIT_INDEX_FILE="$tmp_index"
if "$git" rev-parse --verify HEAD >/dev/null 2>&1; then
  "$git" -c core.fileMode=false read-tree HEAD
else
  "$git" -c core.fileMode=false read-tree --empty
fi
{body.strip()}
"""
    return f"bash -lc {shlex.quote(script)}"
