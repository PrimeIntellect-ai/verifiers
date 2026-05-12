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

import verifiers
import verifiers.v1 as vf

logger = logging.getLogger(__name__)

# Per-language public DockerHub images. Go/C/C++ use standard images + pytest installed
# in setup(). Rust uses a custom image with pre-warmed cargo registry to avoid
# 500MB+ crate downloads per rollout; build via docker/ when ready.
_LANGUAGE_IMAGES: dict[str, str] = {
    "go": "golang:1.22-bookworm",
    "c": "gcc:13-bookworm",
    "cpp": "gcc:13-bookworm",
    "rust": "rust:latest",  # TODO: switch to primeintellect/programbench-toolchain:latest once pushed
}
DEFAULT_IMAGE = "ubuntu:22.04"

# Processed dataset with README, analysis artifacts, and test branch lists.
# Built by scripts/preprocess_go_subset.py (and full pipeline for all 200 tasks).
DEFAULT_DATASET = "PrimeIntellect/programbench-processed"

WORKSPACE = "/workspace"
SRC_DIR = "/workspace/src"
BINARY_PATH = "/workspace/binary"
EXECUTABLE_PATH = "/workspace/executable"
TEST_DIR = "/workspace/tests"

# Per-language disk (GB): Rust needs cargo cache headroom even with warmed image.
_DISK_GB: dict[str, int] = {"rust": 12, "go": 6, "c": 4, "cpp": 6}
# Per-language compile timeout (seconds).
_COMPILE_TIMEOUT: dict[str, int] = {"rust": 480, "go": 60, "c": 30, "cpp": 60}
# Per-language sandbox lifetime (minutes).
_SANDBOX_TIMEOUT_MIN: dict[str, int] = {"rust": 45, "go": 20, "c": 10, "cpp": 20}

SYSTEM_PROMPT = f"""\
You are a software reverse-engineering expert. Your task is to reconstruct complete, \
compilable source code from a compiled binary and its documentation.

You have access to:
- The compiled binary at {BINARY_PATH} — run it to probe I/O behavior
- Documentation from the original repository (in the task description)

The binary is execute-only: you cannot read or decompile it. Tools like strings, \
objdump, nm, hexdump, and xxd will fail with permission denied. You must infer the \
program's behavior by running it with different inputs.

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
    max_tasks: int | None = None
    hide_tests_from_agent: bool = True
    ds_num_proc: int | None = None
    ds_keep_in_memory: bool = True


class ProgramBenchTaskset(vf.Taskset):
    config_type = ProgramBenchTasksetConfig

    def __init__(
        self,
        dataset_name: str | None = None,
        dataset_split: str | None = None,
        filter_language: str | None = None,
        filter_difficulty: str | None = None,
        max_tasks: int | None = None,
        hide_tests_from_agent: bool | None = None,
        ds_num_proc: int | None = None,
        ds_keep_in_memory: bool | None = None,
        config: ProgramBenchTasksetConfig | Mapping[str, object] | None = None,
        **kwargs: Any,
    ):
        config = ProgramBenchTasksetConfig(config)
        self.dataset_name = dataset_name or config.dataset_name
        self.dataset_split = dataset_split or config.dataset_split
        self.filter_language = (
            filter_language if filter_language is not None else config.filter_language
        )
        self.filter_difficulty = (
            filter_difficulty
            if filter_difficulty is not None
            else config.filter_difficulty
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
                "example_io": row.get("example_io", []),
                # Retained for internal logging only — NOT included in the agent prompt.
                "_nm_output": row.get("nm_output", ""),
                "_strings_output": row.get("strings_output", ""),
                "_objdump_head": row.get("objdump_head", ""),
                "_compile_hint": row.get("compile_hint", ""),
            }
            instruction = _build_instruction(info)
            rows.append(
                {
                    "example_id": index,
                    "task_id": info["task_id"],
                    "question": instruction,
                    "instruction": instruction,
                    "prompt": [{"role": "user", "content": instruction}],
                    "answer": "",
                    "info": info,
                    "sandbox": self.sandbox_config(info),
                    "program": {"env": {"AGENT_WORKDIR": SRC_DIR}},
                }
            )
        return rows

    def sandbox_config(self, info: dict) -> dict[str, object]:
        lang = info.get("language", "c")
        return {
            "image": _LANGUAGE_IMAGES.get(lang, DEFAULT_IMAGE),
            "cpu_cores": 2,
            "memory_gb": 6,
            "disk_size_gb": _DISK_GB.get(lang, 8),
            "gpu_count": 0,
            "workdir": SRC_DIR,
            "scope": "rollout",
            "timeout_minutes": _SANDBOX_TIMEOUT_MIN.get(lang, 20),
        }

    def get_env_vars(self) -> dict[str, str]:
        return {
            "PATH": (
                # /usr/local/cargo/bin: rust:latest (official image)
                # /root/.cargo/bin: custom primeintellect image (rustup into /root)
                "/usr/local/cargo/bin:/root/.cargo/bin:/usr/local/go/bin"
                ":/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
            ),
            "CARGO_HOME": "/usr/local/cargo",
            "GOPATH": "/root/go",
            "PAGER": "cat",
            "MANPAGER": "cat",
        }

    # ------------------------------------------------------------------
    # Rollout lifecycle
    # ------------------------------------------------------------------

    @vf.setup(priority=250)
    async def setup_programbench_sandbox(self, task, state, sandbox=None) -> None:
        if sandbox is None:
            raise RuntimeError("ProgramBench setup requires an active sandbox.")
        state["_pb_sandbox"] = sandbox
        state["sandbox_id"] = getattr(sandbox, "id", state.get("sandbox_id"))

        info = task["info"]

        await sandbox.execute(f"mkdir -p {SRC_DIR} {TEST_DIR}", timeout=10)

        # Install pytest and tmux. Standard language images (golang:1.22, gcc:13) have
        # python3 but not pytest; ProgramBench tests also use tmux for TTY emulation.
        # apt-get update is required since package lists aren't cached in the base image.
        result = await sandbox.execute(
            "apt-get update -qq 2>&1 | tail -1 && "
            "apt-get install -y --no-install-recommends tmux python3-pip 2>&1 | tail -2 && "
            "python3 -m pip install --quiet --break-system-packages --no-cache-dir pytest pytest-xdist 2>&1 | tail -2 && "
            "python3 -c \"import pytest; print('pytest_ok', pytest.__version__)\"",
            timeout=180,
        )
        if result.exit_code != 0:
            logger.warning(
                "[setup] tooling install FAILED exit=%s out=%s",
                result.exit_code,
                (result.stdout or "")[-500:],
            )
        else:
            logger.debug(
                "[setup] tooling install ok: %s",
                (result.stdout or "").strip().split("\n")[-1],
            )

        # Upload binary from HuggingFace (best-effort; agent can still work from docs).
        await self._upload_binary(sandbox, info)

        # Download test archive and hide it from the agent until scoring.
        await self._setup_tests(sandbox, task, state, info)

    async def _upload_binary(self, sandbox, info: dict) -> None:
        hf_repo = info.get("binary_hf_repo", "")
        hf_filename = info.get("binary_hf_filename", "")
        if not hf_repo or not hf_filename:
            logger.debug(
                "No binary configured for task %s — skipping.", info["task_id"]
            )
            return
        try:
            from huggingface_hub import hf_hub_download

            local_path = await asyncio.to_thread(
                hf_hub_download,
                repo_id=hf_repo,
                filename=hf_filename,
                repo_type="dataset",
                token=os.environ.get("HF_TOKEN"),
            )
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
            state["_pb_test_branch"] = None
            return

        branch = test_branches[0]
        task_id = info["task_id"]

        try:
            from huggingface_hub import hf_hub_download

            local_archive = await asyncio.to_thread(
                hf_hub_download,
                repo_id="programbench/ProgramBench-Tests",
                filename=f"{task_id}/tests/{branch.removesuffix('.tar.gz').removesuffix('.tar')}.tar.gz",
                repo_type="dataset",
                token=os.environ.get("HF_TOKEN"),
            )
        except Exception as exc:
            logger.warning(
                "Test archive download failed for %s/%s: %r", task_id, branch, exc
            )
            state["_pb_test_branch"] = None
            return

        state["_pb_test_branch"] = branch

        if self.hide_tests_from_agent:
            # Keep archive local and upload only at scoring time.
            state["_pb_test_archive_local"] = local_archive
        else:
            remote = f"{TEST_DIR}/tests.tar.gz"
            await sandbox.upload_file(remote, local_archive, timeout=60)
            await sandbox.execute(
                f"tar -xzf {remote} -C {TEST_DIR} && rm {remote}", timeout=30
            )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @vf.reward(weight=1.0)
    async def solved(self, task, state) -> float:
        if state.get("error") is not None:
            return 0.0
        sandbox = state.get("_pb_sandbox")
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
        # Restore hidden test archive.
        await self._restore_tests(sandbox, state)
        if state.get("_pb_test_branch") is None:
            state["eval_error"] = "no_test_branch"
            return 0.0

        # Compile. Prepend language toolchain paths so compile.sh works even if the
        # agent omitted the PATH export (common with smaller models).
        compile_timeout = _COMPILE_TIMEOUT.get(lang, 120)
        compile_result = await sandbox.execute(
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
            return 0.0

        # Verify executable exists.
        check = await sandbox.execute(
            f"test -f {EXECUTABLE_PATH} && echo ok", timeout=10
        )
        if "ok" not in (check.stdout or ""):
            state["compile_success"] = False
            state["eval_error"] = "executable_not_produced"
            return 0.0
        state["compile_success"] = True

        # Place executable in the archive root — tests use EXECUTABLE = "./executable"
        # and expect it relative to their CWD, which is the extracted archive root.
        await sandbox.execute(
            f"cp {EXECUTABLE_PATH} {TEST_DIR}/executable && chmod +x {TEST_DIR}/executable",
            timeout=10,
        )

        # Discover test files (AI-generated tests land in eval/tests/ or similar).
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
            return 0.0

        # Run pytest from the archive root so ./executable resolves correctly.
        pytest_result = await sandbox.execute(
            f"cd {TEST_DIR} && python3 -m pytest {TEST_DIR} --tb=short -q "
            f"--junit-xml={TEST_DIR}/results.xml 2>&1",
            timeout=300,
        )
        state["pytest_log"] = (pytest_result.stdout or "")[:4000]

        # Parse JUnit XML.
        xml_result = await sandbox.execute(
            f"cat {TEST_DIR}/results.xml 2>/dev/null || echo '<empty/>'", timeout=10
        )
        passed, total = _parse_junit_xml(xml_result.stdout or "")
        state["n_tests_passed"] = passed
        state["n_tests_total"] = total
        logger.info(
            "[%s] scored %d/%d tests passed (reward=%.3f)",
            task_id,
            passed,
            total,
            passed / total if total > 0 else 0.0,
        )

        return passed / total if total > 0 else 0.0

    async def _restore_tests(self, sandbox, state) -> None:
        local_archive = state.pop("_pb_test_archive_local", None)
        if not local_archive:
            return
        remote = f"{TEST_DIR}/tests.tar.gz"
        # Ensure TEST_DIR exists (may have been absent if sandbox was restarted).
        await sandbox.execute(f"mkdir -p {TEST_DIR}", timeout=10)
        await sandbox.upload_file(remote, str(local_archive), timeout=120)
        result = await sandbox.execute(
            f"tar -xzf {remote} -C {TEST_DIR} && rm {remote}", timeout=60
        )
        if result.exit_code != 0:
            raise RuntimeError(
                f"Failed to extract test archive: exit_code={result.exit_code}"
            )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    @vf.cleanup(priority=100)
    async def cleanup_pb_state(self, task, state) -> None:
        archive = state.pop("_pb_test_archive_local", None)
        if isinstance(archive, str):
            Path(archive).unlink(missing_ok=True)
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
    environment_timeout: int = 120,
    system_prompt: str | None = None,
    **kwargs: Any,
) -> vf.MiniSWEAgent:
    return vf.MiniSWEAgent(
        agent_workdir=agent_workdir,
        max_turns=max_turns,
        environment_timeout=environment_timeout,
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
    # mini-SWE-agent harness
    max_turns: int | None = None,
    environment_timeout: int = 120,
    system_prompt: str | None = SYSTEM_PROMPT,
) -> vf.Env:
    verifiers.ensure_keys(["HF_TOKEN"])
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
        ),
    )
    taskset = load_taskset(config=config.taskset)
    harness = load_harness(
        config=config.harness,
        agent_workdir=SRC_DIR,
        max_turns=max_turns,
        environment_timeout=environment_timeout,
        system_prompt=system_prompt,
    )
    return vf.Env(taskset=taskset, harness=harness)
