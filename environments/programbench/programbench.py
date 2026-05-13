"""ProgramBench environment — mini-SWE-agent reconstructs programs from compiled binaries.

Each task provides a compiled binary (execute-only, chmod 111) + documentation.
The agent probes behavior by running the binary, then writes source code and a
compile.sh that produces a behaviorally equivalent executable, scored by pytest tests.

Usage::

    uv run vf-eval programbench -a '{"filter_language":"go","max_tasks":5}' -n 2 -r 1 -d -v
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from datasets import load_dataset

import verifiers.v1 as vf

logger = logging.getLogger(__name__)

# Unified toolchain image with all languages + pytest + tmux pre-installed.
# Built from docker/Dockerfile and pushed to DockerHub. All languages use this image
# so runtime apt-get installs are skipped and network_access=False is safe.
# Override via PRIME_TOOLCHAIN_IMAGE env var to use a team-scoped image from
# Prime's internal registry. Falls back to the public DockerHub image if unset.
TOOLCHAIN_IMAGE = os.environ.get(
    "PRIME_TOOLCHAIN_IMAGE", "primeintellect/programbench-toolchain:latest"
)

_LANGUAGE_IMAGES: dict[str, str] = {
    "go": TOOLCHAIN_IMAGE,
    "c": TOOLCHAIN_IMAGE,
    "cpp": TOOLCHAIN_IMAGE,
    "rust": TOOLCHAIN_IMAGE,
    "haskell": TOOLCHAIN_IMAGE,
    "java": TOOLCHAIN_IMAGE,
}
DEFAULT_IMAGE = TOOLCHAIN_IMAGE

# Processed dataset with README, analysis artifacts, and test branch lists.
# Built by scripts/preprocess_go_subset.py (and full pipeline for all 200 tasks).
DEFAULT_DATASET = "PrimeIntellect/programbench-processed"

WORKSPACE = "/workspace"
SRC_DIR = "/workspace/src"
BINARY_PATH = "/workspace/binary"
EXECUTABLE_PATH = "/workspace/executable"
TEST_DIR = "/workspace/tests"

# Per-language disk (GB): Rust needs cargo cache headroom even with warmed image.
_DISK_GB: dict[str, int] = {
    "rust": 12,
    "go": 6,
    "c": 4,
    "cpp": 6,
    "haskell": 12,
    "java": 8,
}
# Per-language compile timeout (seconds).
_COMPILE_TIMEOUT: dict[str, int] = {
    "rust": 480,
    "go": 60,
    "c": 30,
    "cpp": 60,
    "haskell": 600,
    "java": 120,
}
# Pytest timeout matching paper's eval/run.sh limit (container.py: 3600s).
_TEST_TIMEOUT: dict[str, int] = {
    "rust": 3600,
    "go": 3600,
    "c": 3600,
    "cpp": 3600,
    "haskell": 3600,
    "java": 3600,
}
# Per-language sandbox lifetime (minutes). Paper specifies 6-hour wall clock (§4).
_SANDBOX_TIMEOUT_MIN: dict[str, int] = {
    "rust": 360,
    "go": 360,
    "c": 360,
    "cpp": 360,
    "haskell": 360,
    "java": 360,
}

SYSTEM_PROMPT = f"""\
You are a software reverse-engineering expert. Your task is to reconstruct complete, \
compilable source code from a compiled binary and its documentation.

You have access to:
- The compiled binary at {BINARY_PATH} — run it to probe I/O behavior
- Documentation from the original repository (in the task description)

The binary is execute-only: you cannot read or decompile it. Tools like strings, \
objdump, nm, hexdump, xxd, strace, and ltrace will fail or are prohibited. You must \
infer the program's behavior by running it with different inputs.

The following are strictly prohibited and will result in disqualification:
- Internet access: no git clone, wget, curl to external hosts, or similar
- Source code lookup via package managers: no `cargo install`, `go get`, `apt-get source`, \
  `pip install` of the original project, or reading package manager caches such as \
  ~/.cargo/registry/src/ or Go module cache
- Wrapping or delegating to the original binary — your solution must be an independent \
  reimplementation

Note: this environment has no internet access. All toolchain dependencies are pre-installed.

Your deliverables:
1. Source code written to {SRC_DIR}/
2. A shell script at {SRC_DIR}/compile.sh that compiles your code and writes the \
executable to {EXECUTABLE_PATH}

The executable must reproduce the original binary's observable behavior (stdout, stderr, \
exit codes) as closely as possible. Focus on the public interface first.
"""


class ProgramBenchTasksetConfig(vf.TasksetConfig):
    dataset_name: str = DEFAULT_DATASET
    dataset_split: str = "train"
    filter_language: str | None = None
    filter_difficulty: str | None = None
    filter_task_ids: list[str] | None = None
    max_tasks: int | None = None
    hide_tests_from_agent: bool = True
    ds_num_proc: int | None = None
    ds_keep_in_memory: bool = True
    # Sandbox resource overrides (per-task, all languages)
    cpu_cores: int | None = None
    memory_gb: int | None = None
    network_access: bool = False
    # Per-language timeout overrides (seconds). None = use built-in defaults.
    compile_timeout: int | None = None
    test_timeout: int | None = None
    sandbox_timeout_minutes: int | None = None
    # Retry pytest up to N times per task, keep best result (paper §4 retry logic).
    test_retries: int = 3


class ProgramBenchTaskset(vf.Taskset):
    config_type = ProgramBenchTasksetConfig

    def __init__(
        self,
        dataset_name: str | None = None,
        dataset_split: str | None = None,
        filter_language: str | None = None,
        filter_difficulty: str | None = None,
        filter_task_ids: list[str] | None = None,
        max_tasks: int | None = None,
        hide_tests_from_agent: bool | None = None,
        ds_num_proc: int | None = None,
        ds_keep_in_memory: bool | None = None,
        cpu_cores: int | None = None,
        memory_gb: int | None = None,
        network_access: bool | None = None,
        compile_timeout: int | None = None,
        test_timeout: int | None = None,
        sandbox_timeout_minutes: int | None = None,
        test_retries: int | None = None,
        config: ProgramBenchTasksetConfig | Mapping[str, object] | None = None,
        **kwargs: Any,
    ):
        config = ProgramBenchTasksetConfig(config)
        self.dataset_name = (
            dataset_name if dataset_name is not None else config.dataset_name
        )
        self.dataset_split = (
            dataset_split if dataset_split is not None else config.dataset_split
        )
        self.filter_language = (
            filter_language if filter_language is not None else config.filter_language
        )
        self.filter_difficulty = (
            filter_difficulty
            if filter_difficulty is not None
            else config.filter_difficulty
        )
        self.filter_task_ids = (
            filter_task_ids if filter_task_ids is not None else config.filter_task_ids
        )
        self.max_tasks = max_tasks if max_tasks is not None else config.max_tasks
        self.hide_tests_from_agent = (
            hide_tests_from_agent
            if hide_tests_from_agent is not None
            else config.hide_tests_from_agent
        )
        self.ds_num_proc = (
            ds_num_proc if ds_num_proc is not None else config.ds_num_proc
        )
        self.ds_keep_in_memory = (
            ds_keep_in_memory
            if ds_keep_in_memory is not None
            else config.ds_keep_in_memory
        )
        self.cpu_cores = cpu_cores if cpu_cores is not None else config.cpu_cores
        self.memory_gb = memory_gb if memory_gb is not None else config.memory_gb
        self.network_access = (
            network_access if network_access is not None else config.network_access
        )
        self.compile_timeout = (
            compile_timeout if compile_timeout is not None else config.compile_timeout
        )
        self.test_timeout = (
            test_timeout if test_timeout is not None else config.test_timeout
        )
        self.sandbox_timeout_minutes = (
            sandbox_timeout_minutes
            if sandbox_timeout_minutes is not None
            else config.sandbox_timeout_minutes
        )
        self.test_retries = (
            test_retries if test_retries is not None else config.test_retries
        )
        super().__init__(
            source=self.load_rows,
            taskset_id="programbench",
            config=config,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def load_rows(self) -> list[dict[str, Any]]:
        dataset = load_dataset(
            self.dataset_name,
            split=self.dataset_split,
            keep_in_memory=self.ds_keep_in_memory,
            num_proc=self.ds_num_proc,
        )
        if self.filter_language:
            dataset = dataset.filter(
                lambda x: x.get("language") == self.filter_language
            )
        if self.filter_difficulty:
            dataset = dataset.filter(
                lambda x: x.get("difficulty") == self.filter_difficulty
            )
        if self.filter_task_ids:
            task_id_set = set(self.filter_task_ids)
            dataset = dataset.filter(lambda x: x.get("task_id") in task_id_set)
        if self.max_tasks:
            dataset = dataset.select(range(min(self.max_tasks, len(dataset))))

        rows = []
        for index, row in enumerate(dataset):
            info = {
                "task_id": row["task_id"],
                "language": row["language"],
                "difficulty": row.get("difficulty"),
                "readme": row.get("readme", ""),
                "docs": row.get("docs", ""),
                "file_type": row.get("file_type", ""),
                "binary_size": row.get("binary_size", 0),
                "binary_hf_repo": row.get("binary_hf_repo", ""),
                "binary_hf_filename": row.get("binary_hf_filename", ""),
                "test_branches": row.get("test_branches", []),
                "test_hf_repo": row.get(
                    "test_hf_repo", "programbench/ProgramBench-Tests"
                ),
                # example_io holds expected I/O pairs — kept for internal reference only,
                # never passed to the agent (ground truth).
                "example_io": row.get("example_io", []),
            }
            instruction = _build_instruction(info)
            sandbox = self.sandbox_config(info)
            command_timeout = int(sandbox.get("command_timeout", 900))
            rows.append(
                {
                    "example_id": index,
                    "task_id": info["task_id"],
                    "question": instruction,
                    "instruction": instruction,
                    "prompt": [{"role": "user", "content": instruction}],
                    "answer": "",
                    "info": info,
                    "sandbox": sandbox,
                    "program": {
                        "env": {
                            "AGENT_WORKDIR": SRC_DIR,
                            # Inject toolchain paths so mini-swe-agent inherits them.
                            "PATH": (
                                "/usr/local/go/bin:/usr/local/cargo/bin:/root/.cargo/bin"
                                ":/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
                            ),
                            "CARGO_HOME": "/usr/local/cargo",
                            "GOPATH": "/root/go",
                            "PAGER": "cat",
                            "MANPAGER": "cat",
                            # Let mini-swe-agent self-exit 60s before the sandbox timeout.
                            "AGENT_TIMEOUT_SECONDS": str(max(60, command_timeout - 60)),
                        }
                    },
                }
            )
        return rows

    def sandbox_config(self, info: dict) -> dict[str, object]:
        lang = info.get("language", "c")
        timeout_min = self.sandbox_timeout_minutes or _SANDBOX_TIMEOUT_MIN.get(lang, 20)
        # command_timeout controls the mini-swe-agent execution budget (run_background_job).
        # compile/test use separate per-call timeouts in _compile()/_run_tests().
        command_timeout = timeout_min * 60
        return {
            "image": _LANGUAGE_IMAGES.get(lang, DEFAULT_IMAGE),
            "cpu_cores": self.cpu_cores or 2,
            "memory_gb": self.memory_gb or 6,
            "disk_size_gb": _DISK_GB.get(lang, 8),
            "gpu_count": 0,
            "workdir": SRC_DIR,
            "scope": "rollout",
            "timeout_minutes": timeout_min,
            "command_timeout": command_timeout,
            # Paper §8.2: internet blocked at infra level to prevent source-code lookup.
            # Requires primeintellect/programbench-toolchain image with tools pre-installed.
            "network_access": self.network_access,
        }

    # ------------------------------------------------------------------
    # Rollout lifecycle
    # ------------------------------------------------------------------

    @vf.setup(priority=250)
    async def setup_programbench_sandbox(self, task, state, sandbox=None) -> None:
        import time

        t0 = time.monotonic()
        if sandbox is None:
            raise RuntimeError("ProgramBench setup requires an active sandbox.")
        state["_pb_sandbox"] = sandbox
        state["sandbox_id"] = getattr(sandbox, "id", state.get("sandbox_id"))

        info = task["info"]
        task_id = info["task_id"]
        lang = info.get("language", "c")
        image = _LANGUAGE_IMAGES.get(lang, DEFAULT_IMAGE)
        logger.info(
            "[%s] setup start lang=%s image=%s sandbox=%s",
            task_id,
            lang,
            image,
            state["sandbox_id"],
        )

        mkdir_result = await sandbox.execute(
            f"mkdir -p {SRC_DIR} {TEST_DIR}", timeout=10
        )
        if mkdir_result.exit_code != 0:
            logger.warning("[%s] mkdir failed exit=%s", task_id, mkdir_result.exit_code)

        # Write a profile.d snippet so bash -l (used by mini-swe-agent) sees toolchain paths.
        # Docker ENV sets PATH in the container process env, but login shells re-source
        # /etc/profile which may not include language-specific toolchain directories.
        profile_result = await sandbox.execute(
            "printf '%s\\n' "
            "'export PATH=/usr/local/go/bin:/usr/local/cargo/bin:/root/.cargo/bin:\"$PATH\"' "
            "'export CARGO_HOME=/usr/local/cargo' "
            "'export GOPATH=/root/go' "
            "'export PAGER=cat' "
            "'export MANPAGER=cat' "
            "> /etc/profile.d/pb_toolchain.sh",
            timeout=10,
        )
        if profile_result.exit_code != 0:
            logger.warning(
                "[%s] profile.d write failed exit=%s", task_id, profile_result.exit_code
            )

        # Check if pytest and tmux are already installed (pre-built in TOOLCHAIN_IMAGE).
        # If present, skip apt-get entirely so network_access=False sandboxes work cleanly.
        check = await sandbox.execute(
            "command -v pytest && command -v tmux && python3 -c 'import pytest; print(pytest.__version__)'",
            timeout=10,
        )
        if check.exit_code == 0:
            logger.info(
                "[%s] tooling pre-installed (pytest %s) — skipping install",
                task_id,
                (check.stdout or "").strip().split("\n")[-1],
            )
        else:
            # Fallback: install at runtime (requires network_access=True or accessible mirrors).
            logger.warning(
                "[%s] tooling not pre-installed, attempting runtime install", task_id
            )
            result = await sandbox.execute(
                "apt-get update -qq 2>&1 | tail -1 && "
                "apt-get install -y --no-install-recommends tmux python3-pip 2>&1 | tail -2 && "
                "python3 -m pip install --quiet --break-system-packages --no-cache-dir pytest pytest-xdist 2>&1 | tail -2 && "
                "python3 -c \"import pytest; print('pytest_ok', pytest.__version__)\"",
                timeout=180,
            )
            if result.exit_code != 0:
                logger.warning(
                    "[%s] runtime tooling install FAILED exit=%s — tests will likely fail. "
                    "Use primeintellect/programbench-toolchain image to avoid this.",
                    task_id,
                    result.exit_code,
                )
            else:
                logger.info("[%s] runtime tooling install ok", task_id)

        # Upload binary from HuggingFace (best-effort; agent can still work from docs).
        await self._upload_binary(sandbox, info)

        # Download test archive and hide it from the agent until scoring.
        await self._setup_tests(sandbox, task, state, info)

        elapsed = time.monotonic() - t0
        logger.info("[%s] setup complete in %.1fs", task_id, elapsed)

    async def _hf_download(self, repo_id: str, filename: str) -> str:
        from huggingface_hub import hf_hub_download

        # Pass token=None so huggingface_hub falls back to its own auth chain:
        # HF_TOKEN env var → ~/.cache/huggingface/token → huggingface-cli login cache.
        return await asyncio.to_thread(
            hf_hub_download,
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            token=None,
        )

    async def _upload_binary(self, sandbox, info: dict) -> None:
        hf_repo = info.get("binary_hf_repo", "")
        hf_filename = info.get("binary_hf_filename", "")
        if not hf_repo or not hf_filename:
            logger.debug(
                "No binary configured for task %s — skipping.", info["task_id"]
            )
            return
        try:
            local_path = await self._hf_download(hf_repo, hf_filename)
            await sandbox.upload_file(BINARY_PATH, local_path, timeout=60)
            # Execute-only: agent can run the binary but not read/decompile it.
            await sandbox.execute(f"chmod 111 {BINARY_PATH}", timeout=5)
        except Exception as exc:
            logger.warning(
                "Binary upload failed for %s: %r — agent will work from docs only.",
                info["task_id"],
                exc,
            )

    async def _setup_tests(self, sandbox, task, state, info: dict) -> None:
        test_branches = info.get("test_branches", [])
        if not test_branches:
            logger.warning("No test branches for task %s.", info["task_id"])
            state["_pb_test_archives"] = []
            return

        task_id = info["task_id"]
        test_hf_repo = info.get("test_hf_repo", "programbench/ProgramBench-Tests")

        # Download all branches concurrently — each is a separate oracle test suite.
        # Paper §4 scores against all branches; we union them for maximum coverage.
        async def _fetch(branch: str) -> tuple[str, str] | None:
            stem = branch.removesuffix(".tar.gz").removesuffix(".tar")
            try:
                path = await self._hf_download(
                    test_hf_repo,
                    f"{task_id}/tests/{stem}.tar.gz",
                )
                return (stem, path)
            except Exception as exc:
                logger.warning(
                    "Test archive download failed for %s/%s: %r", task_id, branch, exc
                )
                return None

        results = await asyncio.gather(*[_fetch(b) for b in test_branches])
        archives: list[tuple[str, str]] = [r for r in results if r is not None]

        if not archives:
            logger.warning("All test archive downloads failed for %s.", task_id)
            state["_pb_test_archives"] = []
            return

        logger.debug(
            "[%s] downloaded %d/%d test branches",
            task_id,
            len(archives),
            len(test_branches),
        )

        if self.hide_tests_from_agent:
            # Keep archives local; upload and extract at scoring time.
            state["_pb_test_archives"] = archives
        else:
            # WARNING: exposes test files to the agent during the rollout.
            # Paper §3 prohibits this — use only for local debugging, never for eval.
            logger.warning(
                "[%s] hide_tests_from_agent=False: test files will be visible to the "
                "agent. This violates paper §3 and must not be used for evaluation.",
                task_id,
            )
            await self._extract_archives(sandbox, archives, task_id)
            state["_pb_test_archives"] = archives

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @vf.metric(priority=-1)
    async def resolved_binary(self, task, state) -> float:
        """Binary 0/1 — primary paper metric (% Resolved). Logged but not used as RL signal."""
        return 1.0 if state.get("resolved") else 0.0

    @vf.reward(weight=1.0)
    async def solved(self, task, state) -> float:
        task_id = task.get("info", {}).get("task_id", "?")
        has_error = state.get("error") is not None
        sandbox = state.get("_pb_sandbox")
        logger.info(
            "[%s] solved() called: has_error=%s, sandbox=%s",
            task_id,
            has_error,
            type(sandbox).__name__,
        )
        if has_error:
            return 0.0
        if sandbox is None:
            return 0.0

        info = task["info"]
        lang = info.get("language", "c")

        try:
            return await self._evaluate(sandbox, state, info, lang)
        except Exception as exc:
            logger.warning("Evaluation failed for %s: %r", info["task_id"], exc)
            state["eval_error"] = str(exc)
            return 0.0

    async def _evaluate(self, sandbox, state, info: dict, lang: str) -> float:
        task_id = info["task_id"]
        await self._restore_tests(sandbox, state)
        if not state.get("_pb_test_archives"):
            state["eval_error"] = "no_test_archives"
            return 0.0
        if not await self._compile(sandbox, state, lang, task_id):
            return 0.0
        if await self._is_binary_wrap(sandbox, state, task_id):
            return 0.0
        passed, total = await self._run_tests(sandbox, state, task_id, lang)
        state["n_tests_passed"] = passed
        state["n_tests_total"] = total
        reward = passed / total if total > 0 else 0.0
        state["almost"] = reward >= 0.95
        state["resolved"] = passed == total and total > 0
        logger.info(
            "[%s] scored %d/%d tests passed (reward=%.3f almost=%s resolved=%s)",
            task_id,
            passed,
            total,
            reward,
            state["almost"],
            state["resolved"],
        )
        return reward

    async def _is_binary_wrap(self, sandbox, state: dict, task_id: str) -> bool:
        """Return True (and flag state) if the submitted executable is a copy of the reference binary."""
        result = await sandbox.execute(
            f"sha256sum {BINARY_PATH} {EXECUTABLE_PATH} 2>/dev/null", timeout=30
        )
        lines = (result.stdout or "").strip().splitlines()
        if len(lines) != 2:
            return False
        ref_hash = lines[0].split()[0]
        sub_hash = lines[1].split()[0]
        if ref_hash and ref_hash == sub_hash:
            state["compile_success"] = False
            state["eval_error"] = "binary_wrap_detected"
            logger.warning(
                "[%s] binary wrap detected — executable hash matches reference", task_id
            )
            return True
        return False

    async def _compile(self, sandbox, state: dict, lang: str, task_id: str) -> bool:
        compile_timeout = self.compile_timeout or _COMPILE_TIMEOUT.get(lang, 120)
        compile_result = await sandbox.run_background_job(
            f"cd {SRC_DIR} && chmod +x compile.sh && "
            "export PATH=/usr/local/cargo/bin:/root/.cargo/bin:/usr/local/go/bin:/usr/local/bin:/usr/bin:/bin && "
            "bash compile.sh 2>&1",
            timeout=compile_timeout,
        )
        state["compile_exit_code"] = compile_result.exit_code
        state["compile_log"] = (compile_result.stdout or "")[:3000]
        if compile_result.exit_code != 0:
            state["compile_success"] = False
            logger.debug(
                "[%s] compile failed exit=%s log=%s",
                task_id,
                compile_result.exit_code,
                (compile_result.stdout or "")[-500:],
            )
            return False
        check = await sandbox.execute(
            f"test -f {EXECUTABLE_PATH} && echo ok", timeout=10
        )
        if "ok" not in (check.stdout or ""):
            state["compile_success"] = False
            state["eval_error"] = "executable_not_produced"
            return False
        state["compile_success"] = True
        await sandbox.execute(
            f"cp {EXECUTABLE_PATH} {TEST_DIR}/executable && chmod +x {TEST_DIR}/executable",
            timeout=10,
        )
        return True

    async def _run_tests(
        self, sandbox, state: dict, task_id: str, lang: str = "c"
    ) -> tuple[int, int]:
        find_result = await sandbox.execute(
            f"find {TEST_DIR} -name 'test_*.py' -o -name '*_test.py' | head -20",
            timeout=10,
        )
        test_files = (find_result.stdout or "").strip()
        state["test_files_found"] = test_files
        if not test_files:
            state["eval_error"] = "no_test_files_in_archive"
            logger.warning(
                "[%s] no test files in archive; TEST_DIR=%s",
                task_id,
                (
                    await sandbox.execute(f"find {TEST_DIR} | head -20", timeout=10)
                ).stdout,
            )
            return 0, 0
        test_timeout = self.test_timeout or _TEST_TIMEOUT.get(lang, 3600)
        max_retries = max(1, self.test_retries)
        best_passed, best_total = 0, 0
        for attempt in range(max_retries):
            # First retry: enable worker restart recovery (paper §4); subsequent: serial.
            extra_flags = "--max-worker-restart=4" if attempt == 1 else ""
            flags = f"{extra_flags} " if extra_flags else ""
            pytest_result = await sandbox.run_background_job(
                f"cd {TEST_DIR} && python3 -m pytest {TEST_DIR} --tb=short -q "
                f"{flags}--junit-xml={TEST_DIR}/results.xml 2>&1",
                timeout=test_timeout,
            )
            state[f"pytest_log_attempt{attempt}"] = (pytest_result.stdout or "")[:4000]
            xml_result = await sandbox.execute(
                f"cat {TEST_DIR}/results.xml 2>/dev/null || echo '<empty/>'", timeout=10
            )
            passed, total = _parse_junit_xml(xml_result.stdout or "")
            if passed > best_passed or (passed == best_passed and total > best_total):
                best_passed, best_total = passed, total
                state["pytest_log"] = state[f"pytest_log_attempt{attempt}"]
            if passed == total and total > 0:
                break
        return best_passed, best_total

    async def _extract_archives(
        self, sandbox, archives: list[tuple[str, str]], task_id: str
    ) -> None:
        await sandbox.execute(f"mkdir -p {TEST_DIR}", timeout=10)
        for stem, local_path in archives:
            # Each branch goes into its own subdirectory to avoid filename collisions.
            branch_dir = f"{TEST_DIR}/{stem}"
            remote = f"{branch_dir}.tar.gz"
            await sandbox.execute(f"mkdir -p {branch_dir}", timeout=10)
            await sandbox.upload_file(remote, str(local_path), timeout=120)
            result = await sandbox.execute(
                f"tar -xzf {remote} -C {branch_dir} && rm {remote}", timeout=60
            )
            if result.exit_code != 0:
                logger.warning(
                    "[%s] failed to extract branch %s: exit=%s",
                    task_id,
                    stem,
                    result.exit_code,
                )

    async def _restore_tests(self, sandbox, state) -> None:
        archives: list[tuple[str, str]] = state.get("_pb_test_archives", [])
        if not archives:
            return
        task_id = state.get("sandbox_id", "?")
        await self._extract_archives(sandbox, archives, task_id)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    @vf.cleanup(priority=100)
    async def cleanup_pb_state(self, task, state) -> None:
        for _stem, local_path in state.pop("_pb_test_archives", []):
            Path(local_path).unlink(missing_ok=True)
        state.pop("_pb_sandbox", None)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _build_instruction(info: dict) -> str:
    task_id = info["task_id"]
    parts: list[str] = [
        f"# Program Reconstruction: `{task_id}`",
        "",
    ]

    if info.get("readme"):
        parts += ["## Documentation", info["readme"][:3000], ""]

    if info.get("docs"):
        parts += ["## Additional Docs", info["docs"][:2000], ""]

    parts.append("## Binary")
    if info.get("file_type"):
        parts.append(f"Type: {info['file_type']}")
    if info.get("binary_size"):
        parts.append(f"Size: {info['binary_size']:,} bytes")
    parts += [
        f"Located at: `{BINARY_PATH}` (execute-only — run it, do not decompile)",
        "",
    ]

    parts += [
        "## Your task",
        f"1. Run `{BINARY_PATH}` with various inputs to understand the program's behavior.",
        f"2. Write source code in `{SRC_DIR}/`.",
        f"3. Write `{SRC_DIR}/compile.sh` — it must compile your source and produce `{EXECUTABLE_PATH}`.",
        "4. Your executable will be scored against behavioral I/O tests.",
    ]
    return "\n".join(parts)


def _parse_junit_xml(xml: str) -> tuple[int, int]:
    """Return (passed, total) from JUnit XML string. Regex-based to avoid xml overhead."""
    if not xml or "<empty/>" in xml:
        return 0, 0
    try:
        total = len(re.findall(r"<testcase[\s>]", xml))
        failures = len(re.findall(r"<failure[\s>]", xml))
        errors = len(re.findall(r"<error[\s>]", xml))
        skipped = len(re.findall(r"<skipped[\s>]", xml))
        return max(0, total - failures - errors - skipped), max(0, total)
    except Exception:
        return 0, 0


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def load_taskset(
    config: ProgramBenchTasksetConfig | Mapping[str, object] | None = None,
    **kwargs: Any,
) -> ProgramBenchTaskset:
    return ProgramBenchTaskset(config=config, **kwargs)


def load_harness(
    config: vf.HarnessConfig | None = None,
    agent_workdir: str = SRC_DIR,
    max_turns: int | None = None,
    environment_timeout: int = 600,
    agent_step_limit: int = 1000,
    system_prompt: str | None = None,
    **kwargs: Any,
) -> vf.MiniSWEAgent:
    extra: list[str] = [f"agent.step_limit={agent_step_limit}"]
    return vf.MiniSWEAgent(
        agent_workdir=agent_workdir,
        max_turns=max_turns,
        environment_timeout=environment_timeout,
        extra_config_specs=extra,
        system_prompt=system_prompt,
        config=config,
        **kwargs,
    )


def load_environment(
    config: vf.EnvConfig | None = None,
    # Dataset — HF_TOKEN required (private PrimeIntellect dataset + test repo).
    dataset_name: str | None = None,
    dataset_split: str | None = None,
    filter_language: str | None = None,
    filter_difficulty: str | None = None,
    max_tasks: int | None = None,
    hide_tests_from_agent: bool | None = None,
    ds_num_proc: int | None = None,
    ds_keep_in_memory: bool | None = None,
    # Sandbox resources
    cpu_cores: int | None = None,
    memory_gb: int | None = None,
    network_access: bool = False,
    # Timeout overrides (seconds; None = per-language defaults)
    compile_timeout: int | None = None,
    test_timeout: int | None = None,
    sandbox_timeout_minutes: int | None = None,
    # mini-SWE-agent harness
    max_turns: int | None = None,
    environment_timeout: int = 600,
    agent_step_limit: int = 1000,
    system_prompt: str | None = SYSTEM_PROMPT,
) -> vf.Env:
    # Verify HuggingFace auth is available (private dataset + test archives).
    # Accepts HF_TOKEN env var or cached token from `huggingface-cli login`.
    try:
        from huggingface_hub import get_token

        if not get_token():
            raise RuntimeError(
                "No HuggingFace token found. Set HF_TOKEN or run `huggingface-cli login`."
            )
    except ImportError:
        pass  # huggingface_hub not installed yet; will fail at download time
    config = vf.EnvConfig(
        config,
        taskset=ProgramBenchTasksetConfig(
            dataset_name=dataset_name or DEFAULT_DATASET,
            dataset_split=dataset_split or "train",
            filter_language=filter_language,
            filter_difficulty=filter_difficulty,
            max_tasks=max_tasks,
            hide_tests_from_agent=(
                hide_tests_from_agent if hide_tests_from_agent is not None else True
            ),
            ds_num_proc=ds_num_proc,
            ds_keep_in_memory=(
                ds_keep_in_memory if ds_keep_in_memory is not None else True
            ),
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            network_access=network_access,
            compile_timeout=compile_timeout,
            test_timeout=test_timeout,
            sandbox_timeout_minutes=sandbox_timeout_minutes,
        ),
    )
    taskset = load_taskset(config=config.taskset)
    harness = load_harness(
        config=config.harness,
        agent_workdir=SRC_DIR,
        max_turns=max_turns,
        environment_timeout=environment_timeout,
        agent_step_limit=agent_step_limit,
        system_prompt=system_prompt,
    )
    return vf.Env(taskset=taskset, harness=harness)
