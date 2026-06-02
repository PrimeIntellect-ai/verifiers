"""Scale-SWE TaskSet.

Scale-SWE (https://huggingface.co/datasets/AweAI-Team/Scale-SWE) is a
Python-only SWE dataset whose rows ship task-specific Docker images,
``pre_commands`` for checking out the base state, a gold ``patch``, optional
``f2p_patch`` / ``f2p_script`` evaluation tests, and F2P/P2P pytest ids.

This implementation mirrors AweAgent's ``ScaleSWETask`` +
``ScaleSWEEvaluator`` lifecycle:

1. Run the row's ``pre_commands`` in ``workdir`` before the agent/debug step.
2. Apply the candidate patch with AweAgent's multi-strategy patch helper.
3. Restore likely test files to the row base commit before evaluation.
4. Apply ``f2p_patch`` if present, upload ``f2p_script`` as
   ``test_fail_to_pass.py``, and run merged F2P + P2P pytest ids through an
   injected pytest runner.
5. Score 1.0 only when every expected test id is matched in JUnit XML and
   passed.
"""

import json
import logging
import re
import shlex
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import verifiers as vf
from datasets import load_dataset
from verifiers.envs.experimental.composable import SandboxSpec, SandboxTaskSet

logger = logging.getLogger(__name__)

DATASET_NAME = "AweAI-Team/Scale-SWE"

ENV_VARS_SCALE_SWE = {
    "PATH": (
        "/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:"
        "/opt/conda/envs/testbed/bin:/opt/conda/bin:"
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    ),
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
    "CI": "1",
}

_JUNIT_START = "SCALESWE_JUNIT_XML_START"
_JUNIT_END = "SCALESWE_JUNIT_XML_END"
_ERROR_PREFIX = "SCALESWE_ERROR="
_PYTEST_SUCCESS_MARKER = "<pytest>true</pytest>"

_PYTEST_RUNNER_SCRIPT = """\
import json, sys, os
import pytest

if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        config = json.load(f)
    test_ids = config["test_ids"]
    xml_path = config.get("xml_path", "/tmp/_awe_test_results.xml")
    sys.path.insert(0, os.getcwd())
    sys.argv = ["pytest"]
    args = ["-vv", f"--junitxml={xml_path}", "-o", "addopts=", "--rootdir=."] + test_ids
    ret = pytest.main(args)
    print("<pytest>true</pytest>" if ret == 0 else "<pytest>false</pytest>")
"""


class ScaleSWERubric(vf.Rubric):
    """Scores Scale-SWE tasks by checking merged F2P/P2P pytest results."""

    def __init__(self, taskset: "ScaleSWETaskSet", **kwargs: Any):
        super().__init__(**kwargs)
        self.taskset = taskset
        self.add_reward_func(self.solved)

    async def solved(self, state, info, **kwargs) -> float:
        if state.get("error") is not None:
            return 0.0
        sandbox_client = state.get("sandbox_client")
        sandbox_id = state.get("sandbox_id")
        if not sandbox_client or not sandbox_id:
            return 0.0
        try:
            test_output = await self.taskset._run_tests(
                sandbox_client,
                sandbox_id,
                state,
                state.get("test_timeout", self.taskset.default_test_timeout),
            )
            state["test_output"] = test_output
        except Exception as e:
            logger.warning("Scale-SWE test execution failed: %s", e)
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


class ScaleSWETaskSet(SandboxTaskSet):
    """TaskSet for AweAI-Team/Scale-SWE."""

    default_workdir = "/workspace"
    default_test_timeout = 20 * 60

    def __init__(
        self,
        dataset_name: str = DATASET_NAME,
        split: str = "train",
        filter_fn: str | None = None,
        ds_keep_in_memory: bool = True,
        ds_num_proc: int | None = None,
        timeout_minutes: int | None = None,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.ds_keep_in_memory = ds_keep_in_memory
        self.ds_num_proc = ds_num_proc
        self.timeout_minutes = timeout_minutes
        super().__init__(
            dataset=self._build_dataset,
            name="swe/scaleswe",
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
        return dataset.map(_process_example, remove_columns=dataset.column_names, **_kw)

    def get_instruction(self, info: dict) -> str:
        return info["problem_statement"]

    def get_sandbox_spec(self, info: dict) -> SandboxSpec:
        return SandboxSpec(
            image=info["image_url"],
            timeout_minutes=self.timeout_minutes,
        )

    def get_workdir(self, info: dict) -> str:
        return info["workdir"]

    def get_env_vars(self) -> dict[str, str]:
        return dict(ENV_VARS_SCALE_SWE)

    async def setup(self, state) -> None:
        sandbox_client = state["sandbox_client"]
        sandbox_id = state["sandbox_id"]
        info = state["info"]
        state["test_timeout"] = max(
            int(state.get("test_timeout") or 0), self.default_test_timeout
        )
        workdir = self.get_workdir(info)
        pre_commands = _normalize_pre_commands(info.get("pre_commands", ""))
        if not pre_commands:
            raise RuntimeError("Scale-SWE row missing pre_commands")

        result = await sandbox_client.execute_command(
            sandbox_id,
            pre_commands,
            working_dir=workdir,
            timeout=300,
        )
        if result.exit_code != 0:
            raise RuntimeError(
                "Scale-SWE setup pre_commands failed: "
                f"exit_code={result.exit_code} stderr={(result.stderr or '')[-1000:]}"
            )

        await self._verify_setup_state(sandbox_client, sandbox_id, workdir, info)

    async def _verify_setup_state(
        self,
        sandbox_client: Any,
        sandbox_id: str,
        workdir: str,
        info: dict,
    ) -> None:
        parent_commit = info.get("parent_commit") or info.get("base_commit") or ""
        script = f"""
set +e
fail=0
expected={shlex.quote(parent_commit)}
head=$(git rev-parse HEAD 2>&1)
if [ "$head" != "$expected" ]; then
  echo "HEAD mismatch: expected=$expected actual=$head"
  fail=1
fi
branch=$(git rev-parse --abbrev-ref HEAD 2>&1)
if [ "$branch" != "scaleswe" ]; then
  echo "branch mismatch: expected=scaleswe actual=$branch"
  fail=1
fi
email=$(git config user.email 2>/dev/null)
if [ "$email" != "scaleswe@example.com" ]; then
  echo "git user.email mismatch: $email"
  fail=1
fi
name=$(git config user.name 2>/dev/null)
if [ "$name" != "scaleswe-engine" ]; then
  echo "git user.name mismatch: $name"
  fail=1
fi
extra_refs=$(git show-ref 2>/dev/null | awk '{{print $2}}' | grep -v '^refs/heads/scaleswe$')
if [ -n "$extra_refs" ]; then
  echo "unexpected refs after setup:"
  echo "$extra_refs"
  fail=1
fi
reflog_entry=$(git reflog --all --format=%H 2>/dev/null | head -n 1)
if [ -n "$reflog_entry" ]; then
  echo "reflog not expired"
  fail=1
fi
exit "$fail"
"""
        result = await sandbox_client.execute_command(
            sandbox_id,
            script,
            working_dir=workdir,
            timeout=60,
        )
        if result.exit_code != 0:
            output = "\n".join(part for part in (result.stdout, result.stderr) if part)
            raise RuntimeError(
                f"Scale-SWE setup verification failed:\n{output[-2000:]}"
            )

    async def _run_tests(
        self,
        sandbox_client: Any,
        sandbox_id: str,
        state: dict,
        test_timeout: int,
    ) -> str:
        info = state["info"]
        workdir = self.get_workdir(info)
        await self._restore_test_files(sandbox_client, sandbox_id, workdir, info)

        f2p_patch = info.get("f2p_patch") or ""
        if f2p_patch.strip():
            try:
                await self._apply_patch_file(
                    sandbox_client, sandbox_id, workdir, f2p_patch, "f2p_patch"
                )
            except RuntimeError as e:
                return f"{_ERROR_PREFIX}f2p_patch_failed\n{e}"

        f2p_script = info.get("f2p_script") or ""
        if f2p_script:
            await _upload_content(
                sandbox_client,
                sandbox_id,
                f"{workdir.rstrip('/')}/test_fail_to_pass.py",
                f2p_script,
            )

        f2p_ids = _parse_test_ids(info.get("FAIL_TO_PASS"))
        p2p_ids = _parse_test_ids(info.get("PASS_TO_PASS"))
        test_ids = f2p_ids + p2p_ids
        if not test_ids:
            return f"{_ERROR_PREFIX}no_test_ids"

        await _upload_content(
            sandbox_client,
            sandbox_id,
            "/tmp/_awe_pytest_runner.py",
            _PYTEST_RUNNER_SCRIPT,
        )
        await _upload_content(
            sandbox_client,
            sandbox_id,
            "/tmp/_awe_test_config.json",
            json.dumps(
                {"test_ids": test_ids, "xml_path": "/tmp/_awe_test_results.xml"}
            ),
        )

        env_str = _export_env(self.get_env_vars())
        command = (
            f"{env_str} python /tmp/_awe_pytest_runner.py "
            "/tmp/_awe_test_config.json > /tmp/_awe_test_output.txt 2>&1"
        )
        run_result = await sandbox_client.run_background_job(
            sandbox_id,
            command,
            timeout=test_timeout,
            working_dir=workdir,
        )
        output_result = await sandbox_client.execute_command(
            sandbox_id,
            "cat /tmp/_awe_test_output.txt 2>/dev/null || true",
            timeout=300,
        )
        xml_result = await sandbox_client.execute_command(
            sandbox_id,
            "cat /tmp/_awe_test_results.xml 2>/dev/null || true",
            timeout=300,
        )

        raw_output = output_result.stdout or ""
        if output_result.stderr:
            raw_output += f"\n{output_result.stderr}"
        if run_result.exit_code != 0:
            raw_output += f"\nSCALESWE_PYTEST_EXIT={run_result.exit_code}"
        if xml_result.stdout:
            raw_output += f"\n{_JUNIT_START}\n{xml_result.stdout}\n{_JUNIT_END}\n"
        return raw_output

    async def _restore_test_files(
        self, sandbox_client: Any, sandbox_id: str, workdir: str, info: dict
    ) -> None:
        base_commit = info.get("parent_commit") or info.get("base_commit") or ""
        if not base_commit:
            raise RuntimeError("Scale-SWE row missing parent_commit/base_commit")

        command = f"""
base={shlex.quote(base_commit)}
git checkout "$base" -- tests/ test/ Test/ Tests/ 2>/dev/null || true
git ls-tree -r --name-only "$base" 2>/dev/null | while IFS= read -r path; do
  case "$path" in
    test_*.py|*/test_*.py|*_test.py|*/*_test.py|conftest.py|*/conftest.py)
      git checkout "$base" -- "$path" 2>/dev/null || true
      ;;
  esac
done
git ls-files 2>/dev/null | while IFS= read -r path; do
  case "$path" in
    tests/*|test/*|Test/*|Tests/*|test_*.py|*/test_*.py|*_test.py|*/*_test.py|conftest.py|*/conftest.py)
      if ! git cat-file -e "$base:$path" 2>/dev/null; then
        rm -f -- "$path"
        git rm -q --cached -- "$path" 2>/dev/null || true
      fi
      ;;
  esac
done
"""
        await sandbox_client.execute_command(
            sandbox_id, command, working_dir=workdir, timeout=120
        )

    def _calculate_reward(self, test_output: str, info: dict) -> float:
        if not test_output or test_output.startswith(_ERROR_PREFIX):
            return 0.0
        expected = _parse_test_ids(info.get("FAIL_TO_PASS")) + _parse_test_ids(
            info.get("PASS_TO_PASS")
        )
        if not expected:
            return 0.0
        if _PYTEST_SUCCESS_MARKER in test_output:
            return 1.0
        xml_content = _extract_between(test_output, _JUNIT_START, _JUNIT_END)
        if not xml_content:
            return 0.0
        all_passed, details = _parse_junit_xml(xml_content, expected)
        if not all_passed:
            logger.info(
                "Scale-SWE test mismatch for %s: %s",
                info.get("instance_id"),
                details,
            )
            return 0.0
        return 1.0

    async def _apply_patch_file(
        self,
        sandbox_client: Any,
        sandbox_id: str,
        workdir: str,
        patch: str,
        label: str,
    ) -> None:
        remote_path = "/tmp/_awe_agent.patch"
        await _upload_content(sandbox_client, sandbox_id, remote_path, patch)

        strategies = [
            (f"git apply --verbose {remote_path}", False),
            (
                "git apply --verbose --ignore-space-change --ignore-whitespace "
                f"{remote_path}",
                False,
            ),
            (f"patch --batch --fuzz=5 -p1 -i {remote_path}", False),
            (f"git apply --verbose --reject {remote_path}", True),
            (
                "git apply --verbose --reject --ignore-space-change "
                f"--ignore-whitespace {remote_path}",
                True,
            ),
            (
                "git apply --verbose --reject --ignore-space-change "
                f"--ignore-whitespace --allow-empty {remote_path}",
                True,
            ),
        ]

        last_result = None
        for command, is_reject in strategies:
            result = await sandbox_client.execute_command(
                sandbox_id, command, working_dir=workdir, timeout=120
            )
            if result.exit_code == 0:
                return
            if is_reject and result.exit_code == 1:
                return
            last_result = result

        stderr = (getattr(last_result, "stderr", "") or "")[-1000:]
        stdout = (getattr(last_result, "stdout", "") or "")[-1000:]
        raise RuntimeError(
            f"{label} apply failed: exit_code={getattr(last_result, 'exit_code', '?')} "
            f"stdout={stdout} stderr={stderr}"
        )

    async def _apply_gold_patch(
        self, sandbox_client: Any, sandbox_id: str, state: dict
    ) -> None:
        info = state["info"]
        patch = info.get("patch", "")
        if not patch or not patch.strip():
            raise RuntimeError("No gold patch in info['patch']")
        await self._apply_patch_file(
            sandbox_client, sandbox_id, self.get_workdir(info), patch, "gold"
        )

    def get_rubric(self):
        return ScaleSWERubric(self)

    async def validate_instance(self, state) -> bool:
        sandbox_client = state["sandbox_client"]
        sandbox_id = state["sandbox_id"]
        await self._apply_gold_patch(sandbox_client, sandbox_id, state)
        test_output = await self._run_tests(
            sandbox_client,
            sandbox_id,
            state,
            state.get("test_timeout", self.default_test_timeout),
        )
        state["test_output"] = test_output
        info = state.get("info") or {}
        return self._calculate_reward(test_output, info) > 0


def _process_example(x: dict) -> dict:
    return {
        "question": x["problem_statement"],
        "info": {**x},
        "answer": "",
    }


def _normalize_pre_commands(raw: Any) -> str:
    if not isinstance(raw, str):
        return ""
    return raw.strip().removesuffix("\\n")


def _parse_test_ids(raw: str | list[str] | None) -> list[str]:
    if not raw:
        return []
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if t]
    raw = raw.strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(t).strip() for t in parsed if t]
        if isinstance(parsed, str) and parsed:
            return [parsed]
    except (json.JSONDecodeError, TypeError):
        pass
    return [raw]


def _normalize_for_match(value: str) -> str:
    parts = value.strip().split("::")
    if parts and parts[0].endswith(".py"):
        parts[0] = parts[0][:-3]
    return ".".join(parts).replace("/", ".").strip(".")


def _fingerprint(value: str) -> str:
    return re.sub(r"\s+", "", value)


def _parse_junit_xml(
    xml_content: str,
    expected_tests: list[str],
) -> tuple[bool, dict[str, object]]:
    details: dict[str, object] = {
        "matched": {},
        "unmatched_expected": list(expected_tests),
        "xml_errors": [],
    }
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as e:
        details["xml_errors"] = [str(e)]
        return False, details

    exact_set = set(expected_tests)
    norm_map = {_normalize_for_match(t): t for t in expected_tests}
    fp_map = {_fingerprint(_normalize_for_match(t)): t for t in expected_tests}

    matched: dict[str, str] = {}
    found_expected: set[str] = set()

    for testcase in root.iter("testcase"):
        name = testcase.get("name", "")
        classname = testcase.get("classname", "")
        file_attr = testcase.get("file", "")
        if testcase.find("skipped") is not None:
            continue
        status = (
            "failed"
            if testcase.find("failure") is not None
            or testcase.find("error") is not None
            else "passed"
        )

        candidate1 = f"{file_attr}::{name}" if file_attr else ""
        if candidate1 in exact_set:
            matched[candidate1] = status
            found_expected.add(candidate1)
            continue

        candidate2 = _normalize_for_match(f"{classname}.{name}")
        if candidate2 in norm_map:
            original = norm_map[candidate2]
            matched[original] = status
            found_expected.add(original)
            continue

        candidate3 = _fingerprint(candidate2)
        if candidate3 in fp_map:
            original = fp_map[candidate3]
            matched[original] = status
            found_expected.add(original)
            continue

        candidate4 = f"{classname.replace('.', '/')}.py::{name}"
        if candidate4 in exact_set:
            matched[candidate4] = status
            found_expected.add(candidate4)

    unmatched = [test for test in expected_tests if test not in found_expected]
    all_passed = (
        len(found_expected) > 0
        and all(status == "passed" for status in matched.values())
        and len(unmatched) == 0
    )
    details["matched"] = matched
    details["unmatched_expected"] = unmatched
    details["total_matched"] = len(matched)
    details["total_expected"] = len(expected_tests)
    return all_passed, details


def _extract_between(text: str, start: str, end: str) -> str:
    if start not in text or end not in text:
        return ""
    return text.split(start, 1)[1].split(end, 1)[0].strip()


def _export_env(env_vars: dict[str, str]) -> str:
    if not env_vars:
        return ""
    return " ".join(
        f"export {key}={shlex.quote(value)};" for key, value in env_vars.items()
    )


async def _upload_content(
    sandbox_client: Any,
    sandbox_id: str,
    remote_path: str,
    content: str,
) -> None:
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write(content)
        f.flush()
        local_path = f.name
    try:
        await sandbox_client.upload_file(sandbox_id, remote_path, local_path)
    finally:
        Path(local_path).unlink(missing_ok=True)
