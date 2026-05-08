"""SWE-rebench-V2 TaskSet.

SWE-rebench-V2 (https://huggingface.co/datasets/nebius/SWE-rebench-V2) is a
32k-instance multilingual bug-fix dataset across 20 programming languages.
Each row carries a pre-built docker image, a gold ``patch`` that resolves
the issue, a ``test_patch`` that introduces the validating tests, and an
``install_config`` dict pinning the per-instance ``test_cmd`` and the name
of the log parser used to map framework output → {test_id: status}.

Upstream grading (``scripts/eval.py``):

1. ``git reset --hard HEAD`` inside the image's repo workdir.
2. Apply ``patch`` then ``test_patch`` with ``git apply -v --3way --recount
   --ignore-space-change --whitespace=nowarn``.
3. Run ``install_config.test_cmd`` — the image ships with all deps
   pre-installed, so ``install_config.install`` is *not* re-run here.
4. Parse output with the parser named by ``install_config.log_parser``
   (looked up in ``swe_rebench_v2_log_parsers.NAME_TO_PARSER``).
5. Reward is 1.0 iff every ``FAIL_TO_PASS`` and ``PASS_TO_PASS`` id maps to
   ``PASSED`` in the resulting status map (timing suffixes normalized to
   match upstream).

Workdir is ``/{repo-name}`` (second half of ``owner/repo``) — *not*
``/testbed`` — because upstream eval scripts cd to that path.
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Any

import verifiers as vf
from datasets import load_dataset
from verifiers.envs.experimental.composable import SandboxSpec, SandboxTaskSet

from verifiers.envs.experimental.composable.tasksets.swe import (
    swe_rebench_v2_log_parsers as _lp,
)
from verifiers.envs.experimental.composable.tasksets.swe._filters import (
    combine_filter_fns,
    deprecated_filter_repos_filter_fn,
    deprecated_swerebench_language_filter_fn,
)
from verifiers.envs.experimental.composable.tasksets.swe._test_patch import (
    revert_and_reapply_test_patch,
)

logger = logging.getLogger(__name__)


DATASET_NAME = "nebius/SWE-rebench-V2"


# Broad PATH covering all known SWE-rebench-V2 language toolchains.
# Individual images only need the subset that matches their base image.
ENV_VARS_SWE_REBENCH_V2 = {
    "PATH": (
        "/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:"
        "/opt/conda/envs/testbed/bin:/opt/conda/bin:"
        "/usr/local/cargo/bin:/usr/local/go/bin:"
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    ),
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
    "CI": "1",
    "_JAVA_OPTIONS": "-Djava.net.preferIPv6Addresses=false",
}


# Timing-suffix regexes from upstream scripts/eval.py::_normalize_test_name.
# Some parsers emit timing-decorated test names (e.g. "foo [1.34 ms]");
# the dataset's F2P/P2P were captured from earlier runs, so we strip the
# same suffixes on both sides before comparing.
_TIMING_NORMALIZE_RES = [
    re.compile(r"\s*\[\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\]\s*$", re.IGNORECASE),
    re.compile(r"\s+in\s+\d+(?:\.\d+)?\s+(?:msec|sec)\b", re.IGNORECASE),
    re.compile(r"\s*\(\s*\d+(?:\.\d+)?\s*(?:ms|s)\s*\)\s*$", re.IGNORECASE),
]


def _normalize_test_name(name: str) -> str:
    for pat in _TIMING_NORMALIZE_RES:
        name = pat.sub("", name)
    return name.strip()


def _process_example(x: dict) -> dict:
    return {
        "question": x["problem_statement"],
        "info": {**x},
        "answer": "",
    }


def _resolve_parser(parser_name: str):
    parser = _lp.NAME_TO_PARSER.get(parser_name)
    if parser is None:
        parser = getattr(_lp, parser_name, None)
    if parser is None:
        raise ValueError(f"Unknown SWE-rebench-V2 log parser: {parser_name!r}")
    return parser


def _repo_workdir(repo: str) -> str:
    """Mirror upstream eval.py: ``workdir = f'/{repo.split("/")[1]}'``."""
    if "/" not in repo:
        raise ValueError(f"Expected owner/repo, got {repo!r}")
    return f"/{repo.split('/', 1)[1]}"


def _extract_install_config(info: dict) -> dict:
    """Install config is JSON-serialized at dataset-build time; parse here."""
    cfg = info.get("install_config")
    if isinstance(cfg, str):
        return json.loads(cfg)
    if isinstance(cfg, dict):
        return cfg
    raise ValueError("install_config missing or wrong type")


def _normalize_test_cmds(test_cmd: Any) -> list[str]:
    if isinstance(test_cmd, str):
        cmds = [test_cmd]
    elif isinstance(test_cmd, list):
        cmds = [c for c in test_cmd if isinstance(c, str) and c.strip()]
    else:
        raise ValueError(f"install_config.test_cmd unsupported type: {type(test_cmd)}")
    if not cmds:
        raise ValueError("install_config.test_cmd is empty")
    return cmds


def _build_eval_script(test_cmds: list[str], workdir: str) -> str:
    """Run the upstream ``test_cmd`` sequence and emit a SENTINEL bookend.

    The sentinel delimits the test region so we can pass a bounded slice to
    the parser even when install/build steps are noisy. Individual commands
    are wrapped with ``|| FAIL=1`` so the script keeps emitting output after
    the first failing command (parsers need the full summary).
    """
    lines = [
        "#!/bin/bash",
        "set -uo pipefail",
        "",
        f"cd {workdir}",
        "",
        'echo "SWEREBENCH_V2_TEST_OUTPUT_START"',
        "FAIL=0",
    ]
    for cmd in test_cmds:
        lines.append(f"{cmd} || FAIL=1")
    lines += [
        'echo "SWEREBENCH_V2_TEST_OUTPUT_END"',
        "",
        'exit "$FAIL"',
        "",
    ]
    return "\n".join(lines)


def _slice_test_region(output: str) -> str:
    start = "SWEREBENCH_V2_TEST_OUTPUT_START"
    end = "SWEREBENCH_V2_TEST_OUTPUT_END"
    if start in output:
        output = output.split(start, 1)[1]
    if end in output:
        output = output.rsplit(end, 1)[0]
    return output


def _is_resolved(
    status_map: dict[str, str], fail_to_pass: list[str], pass_to_pass: list[str]
) -> bool:
    """All F2P and P2P must be PASSED after timing-suffix normalization."""
    normalized = {_normalize_test_name(k): v for k, v in status_map.items()}
    expected = [
        _normalize_test_name(n) for n in list(fail_to_pass) + list(pass_to_pass)
    ]
    if not expected:
        return False
    return all(normalized.get(t) == _lp.TestStatus.PASSED.value for t in expected)


class SWERebenchV2Rubric(vf.Rubric):
    """Scores SWE-rebench-V2 tasks via log-parser comparison to F2P/P2P."""

    def __init__(self, taskset: "SWERebenchV2TaskSet", **kwargs):
        super().__init__(**kwargs)
        self.taskset = taskset
        self.add_reward_func(self.solved)

    async def solved(self, state, info, **kwargs) -> float:
        if isinstance(state.get("error"), vf.InfraError):
            return 0.0
        sandbox_client = state.get("sandbox_client")
        sandbox_id = state.get("sandbox_id")
        if not sandbox_client or not sandbox_id:
            return 0.0
        try:
            test_output = await self.taskset._run_tests(
                sandbox_client, sandbox_id, state, state.get("test_timeout", 900)
            )
            state["test_output"] = test_output
        except Exception as e:
            logger.warning(f"Test execution failed: {e}")
            state["test_output"] = f"ERROR: {e}"
            return 0.0
        return float(self.taskset._calculate_reward(test_output, info))

    @vf.cleanup
    async def cleanup_sandbox(self, state: vf.State) -> None:
        sandbox_client = state.get("sandbox_client")
        sandbox_id = state.get("sandbox_id")
        if sandbox_client and sandbox_id:
            try:
                await sandbox_client.delete(sandbox_id)
            except Exception:
                pass


class SWERebenchV2TaskSet(SandboxTaskSet):
    """TaskSet for nebius/SWE-rebench-V2 (32k multilingual real bug fixes)."""

    def __init__(
        self,
        language: str | None = None,
        dataset_name: str = DATASET_NAME,
        split: str = "train",
        filter_repos: list[str] | None = None,
        filter_fn: str | None = None,
        ds_num_proc: int | None = None,
        ds_keep_in_memory: bool = True,
        timeout_minutes: int | None = None,
    ):
        self.language = language
        self.dataset_name = dataset_name
        self.split = split
        self.ds_num_proc = ds_num_proc
        self.ds_keep_in_memory = ds_keep_in_memory
        self.timeout_minutes = timeout_minutes
        suffix = f"-{language}" if language else ""
        filter_fn = combine_filter_fns(
            deprecated_swerebench_language_filter_fn(language),
            deprecated_filter_repos_filter_fn(filter_repos),
            filter_fn,
        )
        super().__init__(
            dataset=self._build_dataset,
            name=f"swe/swerebench-v2{suffix}",
            filter_fn=filter_fn,
        )

    def _build_dataset(self) -> Any:
        _kw = dict(
            num_proc=self.ds_num_proc,
            keep_in_memory=self.ds_keep_in_memory,
            load_from_cache_file=False,
        )
        dataset = load_dataset(
            self.dataset_name,
            split=self.split,
            keep_in_memory=self.ds_keep_in_memory,
            num_proc=self.ds_num_proc,
        )
        # Row-level dicts (``install_config``, ``meta``) have variable
        # sub-schemas; Arrow can't infer a consistent struct type across
        # batches, so pre-serialize them to JSON strings. Same trick as
        # swe_lego.py.
        struct_cols = [
            col
            for col, feat in dataset.features.items()
            if hasattr(feat, "__class__")
            and feat.__class__.__name__
            not in (
                "Value",
                "Sequence",
                "ClassLabel",
                "Translation",
                "List",
            )
            and not isinstance(feat, list)
        ]
        if struct_cols:
            dataset = dataset.map(
                lambda x: {
                    col: json.dumps(x[col]) if x[col] is not None else None
                    for col in struct_cols
                },
                num_proc=None,
                load_from_cache_file=False,
            )
        return dataset.map(_process_example, remove_columns=dataset.column_names, **_kw)

    def get_instruction(self, info: dict) -> str:
        return info["problem_statement"]

    def get_sandbox_spec(self, info: dict) -> SandboxSpec:
        return SandboxSpec(
            image=info["image_name"],
            timeout_minutes=self.timeout_minutes,
        )

    def get_workdir(self, info: dict) -> str:
        return _repo_workdir(info["repo"])

    def get_env_vars(self) -> dict[str, str]:
        return dict(ENV_VARS_SWE_REBENCH_V2)

    async def setup(self, state) -> None:
        """Prep the sandbox for the agent.

        Upstream images ship with the repo already checked out at
        ``/{repo-name}`` on the base commit, plus all build deps. We
        apply the dataset's ``test_patch`` here so the F2P tests are
        present both for gold-patch validation and for live rollouts.
        Ripgrep is pre-installed to short-circuit rlm's apt fallback.
        """
        sandbox_client = state["sandbox_client"]
        sandbox_id = state["sandbox_id"]
        info = state["info"]
        workdir = _repo_workdir(info["repo"])

        rg_install = (
            "command -v rg >/dev/null 2>&1 || { "
            "V=14.1.1; "
            "curl -sSL "
            "https://github.com/BurntSushi/ripgrep/releases/download/"
            "${V}/ripgrep-${V}-x86_64-unknown-linux-musl.tar.gz "
            "| tar xz -C /tmp "
            "&& install -m 755 /tmp/ripgrep-${V}-x86_64-unknown-linux-musl/rg /usr/local/bin/rg; "
            "}"
        )
        result = await sandbox_client.execute_command(
            sandbox_id, rg_install, timeout=120
        )
        if result.exit_code != 0:
            logger.warning(
                f"[{sandbox_id}] ripgrep install failed (exit={result.exit_code}); "
                f"rlm install will fall back to apt-get"
            )

        test_patch = (info or {}).get("test_patch") or ""
        if test_patch.strip():
            await self._apply_patch_file(
                sandbox_client, sandbox_id, workdir, test_patch, "test_patch"
            )

    async def _run_tests(
        self,
        sandbox_client: Any,
        sandbox_id: str,
        state: dict,
        test_timeout: int,
    ) -> str:
        info = state["info"]
        cfg = _extract_install_config(info)
        test_cmds = _normalize_test_cmds(cfg.get("test_cmd"))
        workdir = _repo_workdir(info["repo"])

        # Canonical SWE-bench grading dance: revert any agent edits to
        # files touched by ``test_patch`` (``git checkout <base_commit>
        # -- <path>`` for pre-existing files, ``rm -f <path>`` for
        # newly-added ones), then re-apply ``test_patch`` cleanly.
        # Closes the reward-hack where an agent weakens F2P assertions
        # mid-rollout. Agent source edits are untouched — only test-file
        # bits get canonicalized. Uses ``base_commit`` (not ``HEAD``) so
        # a ``git commit`` by the agent can't shift the checkout target
        # onto their tampered snapshot.
        test_patch: str = info.get("test_patch") or ""
        base_commit: str = info.get("base_commit") or ""
        if test_patch.strip() and base_commit:
            await revert_and_reapply_test_patch(
                sandbox_client,
                sandbox_id,
                workdir,
                test_patch,
                base_commit,
                apply_patch=self._apply_patch_file,
            )

        eval_script = _build_eval_script(test_cmds, workdir)

        with tempfile.NamedTemporaryFile(suffix=".sh", mode="w", delete=False) as f:
            f.write(eval_script)
            f.flush()
            local_path = f.name

        try:
            await sandbox_client.upload_file(sandbox_id, "/eval.sh", local_path)
        finally:
            Path(local_path).unlink(missing_ok=True)

        await sandbox_client.execute_command(sandbox_id, "chmod +x /eval.sh")
        env_str = " ".join(f"{k}={v}" for k, v in self.get_env_vars().items())
        command = f"export {env_str}; bash /eval.sh > /test_output.txt 2>&1"
        results = await sandbox_client.run_background_job(
            sandbox_id, command, timeout=test_timeout
        )
        if results.exit_code > 1:
            raise RuntimeError(f"Error running tests: exit_code={results.exit_code}")
        results = await sandbox_client.execute_command(
            sandbox_id, "cat /test_output.txt"
        )
        return results.stdout or ""

    def _calculate_reward(self, test_output: str, info: dict) -> float:
        if not test_output:
            return 0.0
        try:
            cfg = _extract_install_config(info)
            parser = _resolve_parser(cfg["log_parser"])
        except (KeyError, ValueError) as e:
            logger.warning(f"Could not resolve parser: {e}")
            return 0.0
        region = _slice_test_region(test_output)
        try:
            status_map = parser(region) or {}
        except Exception as e:
            logger.warning(f"Log parser raised: {e}")
            return 0.0
        if not status_map:
            return 0.0
        f2p = info.get("FAIL_TO_PASS") or []
        p2p = info.get("PASS_TO_PASS") or []
        return 1.0 if _is_resolved(status_map, f2p, p2p) else 0.0

    async def _apply_patch_file(
        self,
        sandbox_client: Any,
        sandbox_id: str,
        workdir: str,
        patch: str,
        label: str,
    ) -> None:
        with tempfile.NamedTemporaryFile(suffix=".patch", mode="w", delete=False) as f:
            f.write(patch)
            f.flush()
            local_path = f.name

        remote_path = f"/tmp/{label}.patch"
        try:
            await sandbox_client.upload_file(sandbox_id, remote_path, local_path)
        finally:
            Path(local_path).unlink(missing_ok=True)

        # Upstream eval.py uses ``git apply -v --3way --recount
        # --ignore-space-change --whitespace=nowarn``. We follow the same
        # flags; fall back to ``patch --fuzz=5`` if git apply rejects.
        git_cmd = (
            "git apply -v --3way --recount --ignore-space-change "
            f"--whitespace=nowarn {remote_path}"
        )
        result = await sandbox_client.execute_command(
            sandbox_id, git_cmd, working_dir=workdir, timeout=60
        )
        if result.exit_code != 0:
            result = await sandbox_client.execute_command(
                sandbox_id,
                f"patch --fuzz=5 -p1 -i {remote_path}",
                working_dir=workdir,
                timeout=60,
            )
            if result.exit_code != 0:
                stderr = (result.stderr or "")[:500]
                raise RuntimeError(
                    f"{label} apply failed: exit_code={result.exit_code} stderr={stderr}"
                )

    async def _apply_gold_patch(
        self, sandbox_client: Any, sandbox_id: str, state: dict
    ) -> None:
        info = state["info"]
        patch = info.get("patch", "")
        if not patch or not patch.strip():
            raise RuntimeError("No gold patch in info['patch']")
        workdir = _repo_workdir(info["repo"])
        await self._apply_patch_file(sandbox_client, sandbox_id, workdir, patch, "gold")

    def get_rubric(self):
        return SWERebenchV2Rubric(self)

    async def validate_instance(self, state) -> bool:
        """Apply gold patch, run tests, check reward > 0."""
        sandbox_client = state["sandbox_client"]
        sandbox_id = state["sandbox_id"]
        await self._apply_gold_patch(sandbox_client, sandbox_id, state)
        test_output = await self._run_tests(
            sandbox_client, sandbox_id, state, state.get("test_timeout", 900)
        )
        state["test_output"] = test_output
        info = state.get("info") or {}
        return self._calculate_reward(test_output, info) > 0
