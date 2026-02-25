"""Tests for the RLMEnv class (filesystem-based, local-only)."""

import ast
import base64
import contextlib
import io
import json
import os
import pickle
import shutil
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from datasets import Dataset

import verifiers as vf
from verifiers.envs.experimental import rlm_env as rlm_module
from verifiers.envs.experimental.rlm_env import (
    RLMCodeExecutionTimeout,
    RLMEnv,
    RLMSessionError,
    RLMSetupError,
    RLMWorkerError,
    RLMWorkerPaths,
    RLMWorkerRecoveryError,
    SubLLMEmptyModelResponseError,
    _InterceptionPool,
)

# =============================================================================
# Helpers
# =============================================================================


def make_dataset(info: dict) -> Dataset:
    return Dataset.from_dict(
        {
            "question": ["What is 2+2?"],
            "answer": ["4"],
            "info": [info],
        }
    )


def build_env(dataset: Dataset, **kwargs) -> RLMEnv:
    with patch("verifiers.envs.environment.signal.signal"):
        return RLMEnv(dataset=dataset, **kwargs)


def extract_bash_helper_source() -> str:
    template = rlm_module._RLM_BASH_TOOL_HELPER_SCRIPT
    if "def main" in template:
        return template
    return template


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rlm_env() -> RLMEnv:
    dataset = make_dataset({})
    return build_env(
        dataset,
        max_iterations=10,
        max_output_length=1000,
        repl_language="python",
        interception_url="http://test.invalid",
    )


@pytest.fixture
def rlm_env_with_sub_tools() -> RLMEnv:
    def sample_tool(x: int, y: int) -> int:
        """Add two numbers together."""
        return x + y

    def another_tool(text: str) -> str:
        """Reverse a string."""
        return text[::-1]

    dataset = make_dataset({})
    return build_env(
        dataset,
        sub_tools=[sample_tool, another_tool],
        sub_tool_max_turns=3,
        repl_language="python",
        interception_url="http://test.invalid",
    )


@pytest.fixture
def rlm_env_bash() -> RLMEnv:
    dataset = make_dataset({})
    return build_env(
        dataset,
        max_iterations=10,
        max_output_length=1000,
        repl_language="bash",
        interception_url="http://test.invalid",
    )


@pytest.fixture
def context_dir(tmp_path: Path) -> Path:
    root = tmp_path / "context_src"
    root.mkdir()
    (root / "data.txt").write_text("hello", encoding="utf-8")
    nested = root / "nested"
    nested.mkdir()
    (nested / "value.json").write_text('{"a": 1}', encoding="utf-8")
    return root


# =============================================================================
# 1. Pure Utility Functions
# =============================================================================


class TestFormatExecutionOutput:
    """Tests for _format_execution_output method."""

    def test_format_with_stdout(self, rlm_env):
        result = {
            "status": "ok",
            "stdout": "Hello, world!",
            "stderr": "",
            "result": None,
            "execution_count": 1,
        }
        output = rlm_env._format_execution_output(result)
        assert output == "Hello, world!"

    def test_format_with_stderr(self, rlm_env):
        result = {
            "status": "ok",
            "stdout": "output",
            "stderr": "warning message",
            "result": None,
            "execution_count": 1,
        }
        output = rlm_env._format_execution_output(result)
        assert "output" in output
        assert "stderr:" in output
        assert "warning message" in output

    def test_format_with_result_value(self, rlm_env):
        result = {
            "status": "ok",
            "stdout": "",
            "stderr": "",
            "result": "42",
            "execution_count": 3,
        }
        output = rlm_env._format_execution_output(result)
        assert "Out[3]: 42" in output

    def test_format_error_status(self, rlm_env):
        result = {
            "status": "error",
            "stdout": "",
            "stderr": "",
            "result": "Traceback (most recent call last):\n  NameError: name 'x' is not defined",
            "execution_count": 1,
        }
        output = rlm_env._format_execution_output(result)
        assert "Traceback" in output
        assert "NameError" in output

    def test_truncate_long_output(self, rlm_env):
        long_output = "x" * 2000
        result = {
            "status": "ok",
            "stdout": long_output,
            "stderr": "",
            "result": None,
            "execution_count": 1,
        }
        output = rlm_env._format_execution_output(result)
        assert len(output) <= rlm_env.max_output_length + 50
        assert "[output truncated]" in output

    def test_empty_output(self, rlm_env):
        result = {
            "status": "ok",
            "stdout": "",
            "stderr": "",
            "result": None,
            "execution_count": 1,
        }
        output = rlm_env._format_execution_output(result)
        assert output == "(no output)"


class TestGenerateSubToolsDocumentation:
    def test_empty_when_no_sub_tools(self, rlm_env):
        docs = rlm_env._generate_sub_tools_documentation()
        assert docs == ""

    def test_generate_docs_for_tools(self, rlm_env_with_sub_tools):
        docs = rlm_env_with_sub_tools._generate_sub_tools_documentation()
        assert "Sub-LLM Tools" in docs
        assert "sample_tool" in docs
        assert "another_tool" in docs
        assert "Add two numbers" in docs
        assert "Reverse a string" in docs

    def test_docs_include_parameters(self, rlm_env_with_sub_tools):
        docs = rlm_env_with_sub_tools._generate_sub_tools_documentation()
        assert "Parameters" in docs
        assert "`x`" in docs or "x" in docs
        assert "`y`" in docs or "y" in docs


# =============================================================================
# 2. Context Filesystem Setup
# =============================================================================


class TestContextFilesystemSetup:
    @pytest.mark.asyncio
    async def test_setup_state_copies_context_dir(self, context_dir: Path):
        dataset = make_dataset({"context_dir": str(context_dir)})
        env = build_env(dataset, interception_url="http://test.invalid")
        env._ensure_interception_server = AsyncMock()
        env._executor.prepare_filesystem = AsyncMock()
        env._executor.setup = AsyncMock()

        state = {
            "info": {"context_dir": str(context_dir)},
            "model": "m",
            "client": MagicMock(),
        }
        result = await env.setup_state(state)

        try:
            fs_root = Path(result["rlm_fs_root"])
            control_dir = Path(result["rlm_control_dir"])
            rollout_dir = Path(result["rlm_rollout_dir"])

            assert fs_root.is_dir()
            assert (fs_root / "data.txt").read_text(encoding="utf-8") == "hello"
            assert fs_root.parent == control_dir.parent == rollout_dir
            assert fs_root.name == "rlm_fs"
            assert control_dir.name == "rlm_control"
            assert result["rlm_fs_has_data"] is True
            assert result["rlm_fs_source"] == str(context_dir)
        finally:
            await env.cleanup_rlm_state(result)

    @pytest.mark.asyncio
    async def test_setup_state_writes_builtin_context_json(self):
        dataset = make_dataset({"context": {"a": 1}})
        env = build_env(dataset, interception_url="http://test.invalid")
        env._ensure_interception_server = AsyncMock()
        env._executor.prepare_filesystem = AsyncMock()
        env._executor.setup = AsyncMock()

        state = {"info": {"context": {"a": 1}}, "model": "m", "client": MagicMock()}
        result = await env.setup_state(state)
        try:
            fs_root = Path(result["rlm_fs_root"])
            context_file = fs_root / "context.json"
            assert context_file.exists()
            assert json.loads(context_file.read_text(encoding="utf-8")) == {"a": 1}
            assert result["rlm_fs_has_data"] is True
        finally:
            await env.cleanup_rlm_state(result)

    @pytest.mark.asyncio
    async def test_setup_state_writes_builtin_context_text(self):
        dataset = make_dataset({"context": "hello"})
        env = build_env(dataset, interception_url="http://test.invalid")
        env._ensure_interception_server = AsyncMock()
        env._executor.prepare_filesystem = AsyncMock()
        env._executor.setup = AsyncMock()

        state = {"info": {"context": "hello"}, "model": "m", "client": MagicMock()}
        result = await env.setup_state(state)
        try:
            fs_root = Path(result["rlm_fs_root"])
            context_file = fs_root / "context.txt"
            assert context_file.exists()
            assert context_file.read_text(encoding="utf-8") == "hello"
            assert result["rlm_fs_has_data"] is True
        finally:
            await env.cleanup_rlm_state(result)

    @pytest.mark.asyncio
    async def test_setup_state_rejects_symlinks(self, tmp_path: Path):
        src = tmp_path / "context_src"
        src.mkdir()
        (src / "real.txt").write_text("hello", encoding="utf-8")
        try:
            os.symlink(src / "real.txt", src / "link.txt")
        except OSError:
            pytest.skip("symlinks not supported on this platform")

        dataset = make_dataset({"context_dir": str(src)})
        env = build_env(dataset, interception_url="http://test.invalid")
        env._ensure_interception_server = AsyncMock()
        env._executor.prepare_filesystem = AsyncMock()
        env._executor.setup = AsyncMock()

        state = {"info": {"context_dir": str(src)}, "model": "m", "client": MagicMock()}
        with pytest.raises(ValueError, match="symlink"):
            await env.setup_state(state)

    @pytest.mark.asyncio
    async def test_setup_state_respects_size_limit(self, tmp_path: Path):
        src = tmp_path / "context_src"
        src.mkdir()
        (src / "big.txt").write_bytes(b"0123456789")

        dataset = make_dataset({"context_dir": str(src)})
        env = build_env(
            dataset, filesystem_copy_max_bytes=5, interception_url="http://test.invalid"
        )
        env._ensure_interception_server = AsyncMock()
        env._executor.prepare_filesystem = AsyncMock()
        env._executor.setup = AsyncMock()

        state = {"info": {"context_dir": str(src)}, "model": "m", "client": MagicMock()}
        with pytest.raises(ValueError, match="exceeds size limit"):
            await env.setup_state(state)

    @pytest.mark.asyncio
    async def test_setup_state_no_context_creates_empty_dir(self):
        dataset = make_dataset({})
        env = build_env(dataset, interception_url="http://test.invalid")
        env._ensure_interception_server = AsyncMock()
        env._executor.prepare_filesystem = AsyncMock()
        env._executor.setup = AsyncMock()

        state = {"info": {}, "model": "m", "client": MagicMock()}
        result = await env.setup_state(state)
        try:
            fs_root = Path(result["rlm_fs_root"])
            assert fs_root.exists()
            assert list(fs_root.iterdir()) == []
            assert result["rlm_fs_has_data"] is False
        finally:
            await env.cleanup_rlm_state(result)

    @pytest.mark.asyncio
    async def test_system_prompt_mentions_working_dir_and_empty_context(self):
        dataset = make_dataset({})
        env = build_env(dataset, interception_url="http://test.invalid")
        env._ensure_interception_server = AsyncMock()
        env._executor.prepare_filesystem = AsyncMock()
        env._executor.setup = AsyncMock()

        state = {"info": {}, "model": "m", "client": MagicMock()}
        result = await env.setup_state(state)
        try:
            prompt = result["rlm_system_prompt"]
            assert "filesystem available" in prompt
            assert "Working directory:" not in prompt
            assert "No extra data was provided" not in prompt
        finally:
            await env.cleanup_rlm_state(result)


class TestFilesystemCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_removes_filesystem_by_default(self, tmp_path: Path):
        dataset = make_dataset({"context": "hello"})
        env = build_env(dataset, interception_url="http://test.invalid")
        env._ensure_interception_server = AsyncMock()
        env._executor.prepare_filesystem = AsyncMock()
        env._executor.setup = AsyncMock()

        state = {"info": {"context": "hello"}, "model": "m", "client": MagicMock()}
        result = await env.setup_state(state)
        rollout_dir = Path(result["rlm_rollout_dir"])
        assert rollout_dir.exists()

        await env.cleanup_rlm_state(result)
        assert not rollout_dir.exists()

    @pytest.mark.asyncio
    async def test_cleanup_keeps_filesystem_when_configured(self):
        dataset = make_dataset({"context": "hello"})
        env = build_env(
            dataset,
            retain_filesystem_after_rollout=True,
            interception_url="http://test.invalid",
        )
        env._ensure_interception_server = AsyncMock()
        env._executor.prepare_filesystem = AsyncMock()
        env._executor.setup = AsyncMock()

        state = {"info": {"context": "hello"}, "model": "m", "client": MagicMock()}
        result = await env.setup_state(state)
        rollout_dir = Path(result["rlm_rollout_dir"])
        assert rollout_dir.exists()

        try:
            await env.cleanup_rlm_state(result)
            assert rollout_dir.exists()
        finally:
            shutil.rmtree(rollout_dir, ignore_errors=True)


class TestBashPrompt:
    @pytest.mark.asyncio
    async def test_bash_prompt_mentions_env_vars(self, rlm_env_bash):
        env = rlm_env_bash
        env._ensure_interception_server = AsyncMock()
        env._executor.prepare_filesystem = AsyncMock()
        env._executor.setup = AsyncMock()

        state = {"info": {}, "model": "m", "client": MagicMock()}
        result = await env.setup_state(state)
        try:
            prompt = result["rlm_system_prompt"]
            assert "RLM_READY" in prompt
            assert "RLM_CONTENT" in prompt
        finally:
            await env.cleanup_rlm_state(result)


class TestPromptVerbosity:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "verbosity, expected_snippets, unexpected_snippets",
        [
            (
                "light",
                [
                    "You have the `call_python_repl` tool and a filesystem available to you."
                ],
                [
                    "This is an iterative environment.",
                    "Critical: This is an ITERATIVE environment",
                ],
            ),
            (
                "medium",
                [
                    "You have the `call_python_repl` tool and a filesystem available to you.",
                    "This is an iterative environment.",
                ],
                ["Critical: This is an ITERATIVE environment"],
            ),
            (
                "heavy",
                [
                    "iterative Python REPL where you explore data step by step.",
                    "Critical: This is an ITERATIVE environment",
                ],
                ["This is an iterative environment."],
            ),
        ],
    )
    async def test_root_prompt_verbosity_python(
        self,
        verbosity: str,
        expected_snippets: list[str],
        unexpected_snippets: list[str],
    ):
        dataset = make_dataset({})
        env = build_env(
            dataset,
            repl_language="python",
            root_prompt_verbosity=verbosity,
            interception_url="http://test.invalid",
        )
        env._ensure_interception_server = AsyncMock()
        env._executor.prepare_filesystem = AsyncMock()
        env._executor.setup = AsyncMock()

        state = {"info": {}, "model": "m", "client": MagicMock()}
        result = await env.setup_state(state)
        try:
            prompt = result["rlm_system_prompt"]
            for snippet in expected_snippets:
                assert snippet in prompt
            for snippet in unexpected_snippets:
                assert snippet not in prompt
        finally:
            await env.cleanup_rlm_state(result)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("verbosity", ["light", "medium", "heavy"])
    async def test_sub_prompt_verbosity(self, verbosity: str, rlm_env: RLMEnv):
        env = rlm_env
        env.sub_prompt_verbosity = verbosity
        env.sub_tool_max_turns = 7

        captured: dict[str, Any] = {}

        async def _fake_run_sub_llm(state, client, model, messages):
            captured["messages"] = messages
            return {
                "final_content": "ok",
                "turns": [
                    {
                        "prompt_messages": [{"role": "user", "content": "hi"}],
                        "response": {},
                        "tool_call_count": 0,
                    }
                ],
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "tool_call_count": 0,
                "num_turns": 1,
                "max_turns_reached": False,
            }

        env._run_sub_llm = AsyncMock(side_effect=_fake_run_sub_llm)

        await env._run_sub_llm_request(
            state_ref={},
            client=MagicMock(),
            sub_model="m",
            messages=[{"role": "user", "content": "task"}],
            batch_id="b",
            request_id="r",
            parent_turn=0,
        )

        expected = rlm_module._SUB_LLM_SYSTEM_PROMPT_STORE[verbosity].format(
            num_turns=env.sub_tool_max_turns
        )
        assert captured["messages"][0]["role"] == "system"
        assert captured["messages"][0]["content"] == expected


class TestBashReplOutput:
    @pytest.mark.asyncio
    async def test_bash_output_is_raw(self, rlm_env_bash):
        rlm_env_bash._execute_code = AsyncMock(
            return_value={
                "status": "ok",
                "stdout": "output",
                "stderr": "warning",
                "result": None,
                "execution_count": 1,
                "answer": {"ready": False, "content": ""},
            }
        )

        state = {"trajectory": [], "context_warning_sent": False}
        output = await rlm_env_bash.call_bash_repl("echo hi", state)

        assert "output" in output
        assert "warning" in output
        assert "stderr:" not in output
        assert "Out[" not in output
        assert "[Execution time" not in output


class TestBashWorkerScript:
    def test_rendered_bash_worker_is_valid_python(self, tmp_path: Path):
        paths = RLMWorkerPaths(
            base_dir=str(tmp_path),
            command_fifo=str(tmp_path / "cmd"),
            response_fifo=str(tmp_path / "res"),
            ready_flag=str(tmp_path / "ready"),
            worker_path=str(tmp_path / "worker.py"),
            worker_pid_file=str(tmp_path / "worker.pid"),
            context_file=str(tmp_path / "context.json"),
            answer_file=str(tmp_path / "answer.json"),
            log_file=str(tmp_path / "worker.log"),
        )
        script = rlm_module._render_worker_script(paths, repl_language="bash")
        ast.parse(script)

    def test_bash_worker_escapes_exit_code_marker(self, tmp_path: Path):
        paths = RLMWorkerPaths(
            base_dir=str(tmp_path),
            command_fifo=str(tmp_path / "cmd"),
            response_fifo=str(tmp_path / "res"),
            ready_flag=str(tmp_path / "ready"),
            worker_path=str(tmp_path / "worker.py"),
            worker_pid_file=str(tmp_path / "worker.pid"),
            context_file=str(tmp_path / "context.json"),
            answer_file=str(tmp_path / "answer.json"),
            log_file=str(tmp_path / "worker.log"),
        )
        script = rlm_module._render_worker_script(paths, repl_language="bash")
        assert "$?" in script
        assert "__RLM_ENV__" in script


class TestBashToolHelper:
    def _run_helper(
        self,
        argv: list[str],
        stdin_data: str = "",
        response_data: dict | None = None,
    ) -> tuple[str, str, int, dict | None]:
        helper_source = extract_bash_helper_source()
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        env = {
            "RLM_ROOT_TOOL_URL": "http://example.invalid/",
            "RLM_ROOT_TOOL_SERIALIZATION": "pickle",
        }
        captured_payload: dict | None = None
        with patch("urllib.request.urlopen") as mock_urlopen:

            def _capture_request(req, timeout=300):
                nonlocal captured_payload
                data = json.loads(req.data.decode("utf-8"))
                args = list(pickle.loads(base64.b64decode(data["args"])))
                kwargs = pickle.loads(base64.b64decode(data["kwargs"]))
                captured_payload = {
                    "tool_name": data.get("tool_name"),
                    "args": args,
                    "kwargs": kwargs,
                }
                return response

            response = MagicMock()
            response.__enter__.return_value = response
            response.__exit__.return_value = None
            if response_data is None:
                response_data = {
                    "result": base64.b64encode(pickle.dumps(["ok"])).decode("ascii"),
                    "error": None,
                }
            response.read.return_value = json.dumps(response_data).encode("utf-8")
            mock_urlopen.return_value.__enter__.return_value = response
            mock_urlopen.side_effect = _capture_request
            namespace = {"__name__": "__main__"}
            with (
                patch.dict(os.environ, env, clear=False),
                patch("sys.argv", ["rlm_root_tool.py", *argv]),
                patch("sys.stdin", io.StringIO(stdin_data)),
                contextlib.redirect_stdout(stdout_buffer),
                contextlib.redirect_stderr(stderr_buffer),
            ):
                try:
                    exec(helper_source, namespace, namespace)
                except SystemExit as exc:
                    code = exc.code if isinstance(exc.code, int) else 1
                else:
                    code = 0
        return (
            stdout_buffer.getvalue(),
            stderr_buffer.getvalue(),
            code,
            captured_payload,
        )

    def test_llm_batch_json_arg(self):
        payload = json.dumps({"prompts": ["alpha", "beta"]})
        stdout, stderr, code, captured = self._run_helper(
            ["--tool", "llm_batch", "--json", payload]
        )
        assert code == 0
        assert stderr == ""
        assert "ok" in stdout
        assert captured is not None
        assert captured["tool_name"] == "llm_batch"
        assert captured["args"][0] == ["alpha", "beta"]
        assert captured["kwargs"] == {}

    def test_tool_json_args_kwargs(self):
        payload = json.dumps({"args": [1, 2], "kwargs": {"x": "y"}})
        stdout, stderr, code, captured = self._run_helper(
            ["--tool", "other_tool", "--json", payload]
        )
        assert code == 0
        assert stderr == ""
        assert "ok" in stdout
        assert captured is not None
        assert captured["tool_name"] == "other_tool"
        assert captured["args"] == [1, 2]
        assert captured["kwargs"] == {"x": "y"}

    def test_llm_batch_positional_args(self):
        stdout, stderr, code, captured = self._run_helper(
            ["--tool", "llm_batch", "first", "second"]
        )
        assert code == 0
        assert stderr == ""
        assert "ok" in stdout
        assert captured is not None
        assert captured["args"][0] == ["first", "second"]

    def test_llm_batch_json_stdin(self):
        payload = json.dumps({"prompts": ["stdin"]})
        stdout, stderr, code, captured = self._run_helper(
            ["--tool", "llm_batch", "--json"], stdin_data=payload
        )
        assert code == 0
        assert stderr == ""
        assert "ok" in stdout
        assert captured is not None
        assert captured["args"][0] == ["stdin"]

    def test_tool_json_kwargs_only(self):
        payload = json.dumps({"flag": True, "name": "test"})
        stdout, stderr, code, captured = self._run_helper(
            ["--tool", "other_tool", "--json", payload]
        )
        assert code == 0
        assert stderr == ""
        assert captured is not None
        assert captured["args"] == []
        assert captured["kwargs"] == {"flag": True, "name": "test"}

    def test_tool_json_list_args(self):
        payload = json.dumps([1, "two", False])
        stdout, stderr, code, captured = self._run_helper(
            ["--tool", "other_tool", "--json", payload]
        )
        assert code == 0
        assert stderr == ""
        assert captured is not None
        assert captured["args"] == [1, "two", False]
        assert captured["kwargs"] == {}

    def test_tool_json_scalar_arg(self):
        payload = json.dumps("solo")
        stdout, stderr, code, captured = self._run_helper(
            ["--tool", "other_tool", "--json", payload]
        )
        assert code == 0
        assert stderr == ""
        assert captured is not None
        assert captured["args"] == ["solo"]
        assert captured["kwargs"] == {}

    def test_tool_json_extra_args_error(self):
        payload = json.dumps({"args": [1]})
        stdout, stderr, code, captured = self._run_helper(
            ["--tool", "other_tool", "--json", payload, "extra"]
        )
        assert code != 0
        assert "does not accept extra args" in stderr
        assert captured is None

    def test_llm_batch_json_extra_args_error(self):
        payload = json.dumps({"prompts": ["x"]})
        stdout, stderr, code, captured = self._run_helper(
            ["--tool", "llm_batch", "--json", payload, "extra"]
        )
        assert code != 0
        assert "does not accept extra args" in stderr
        assert captured is None

    def test_tool_json_invalid_error(self):
        stdout, stderr, code, captured = self._run_helper(
            ["--tool", "other_tool", "--json", "{invalid"]
        )
        assert code != 0
        assert "Invalid JSON payload" in stderr
        assert captured is None

    def test_llm_batch_output_headers_with_metadata(self):
        payload = json.dumps({"prompts": ["one", "two"]})
        response_data = {
            "result": base64.b64encode(pickle.dumps(["first", "second"])).decode(
                "ascii"
            ),
            "error": None,
            "print_lines": [
                "llm_batch: 2 call(s) in 0.10s",
                "  [0]: 5 tokens, 0 tool calls, 0.01s ✓",
                "  [1]: 6 tokens, 1 tool calls, 0.02s ✓",
            ],
        }
        stdout, stderr, code, captured = self._run_helper(
            ["--tool", "llm_batch", "--json", payload], response_data=response_data
        )
        assert code == 0
        assert stderr == ""
        assert "llm_batch: 2 call(s) in 0.10s" in stdout
        assert "----- llm_batch[0]" in stdout
        assert "----- llm_batch[1]" in stdout
        assert "first" in stdout
        assert "second" in stdout


# =============================================================================
# 3. Initialization and Configuration
# =============================================================================


class TestRLMEnvInitialization:
    def test_default_repl_language_is_bash(self):
        dataset = make_dataset({})
        env = build_env(dataset)

        assert getattr(env, "repl_language", None) == "bash"
        assert "call_bash_repl" in env.tool_map
        assert "call_python_repl" not in env.tool_map

    def test_python_repl_tool_registered(self):
        dataset = make_dataset({})
        env = build_env(dataset, repl_language="python")

        assert "call_python_repl" in env.tool_map
        assert "call_bash_repl" not in env.tool_map

    def test_default_initialization(self):
        dataset = make_dataset({})
        env = build_env(dataset, repl_language="python")

        assert env.sub_model is None
        assert env.sub_tools == []
        assert env.max_iterations == 50
        assert env.max_output_length == 8192
        assert env.max_sub_llm_parallelism == 5
        assert env.sub_llm_stagger_ms == 200
        assert env.sub_llm_stagger_jitter_ms == 50
        assert env.context_key == "context"
        assert env.context_dir_key == "context_dir"

    def test_custom_configuration(self):
        def dummy_tool(x: int) -> int:
            return x * 2

        dataset = make_dataset({})
        env = build_env(
            dataset,
            sub_model="gpt-4",
            sub_tools=[dummy_tool],
            max_iterations=20,
            max_output_length=4096,
            max_sub_llm_parallelism=10,
            sub_llm_stagger_ms=15,
            sub_llm_stagger_jitter_ms=5,
            context_key="custom_context",
            context_dir_key="custom_context_dir",
            repl_language="python",
        )

        assert env.sub_model == "gpt-4"
        assert len(env.sub_tools) == 1
        assert env.max_iterations == 20
        assert env.max_output_length == 4096
        assert env.max_sub_llm_parallelism == 10
        assert env.sub_llm_stagger_ms == 15
        assert env.sub_llm_stagger_jitter_ms == 5
        assert env.context_key == "custom_context"
        assert env.context_dir_key == "custom_context_dir"

    def test_system_prompt_customization(self):
        custom_prompt = "You are a custom RLM assistant."
        dataset = make_dataset({})
        env = build_env(dataset, system_prompt=custom_prompt, repl_language="python")
        assert env.custom_system_prompt == custom_prompt

    def test_bash_tool_removed(self, rlm_env):
        assert "bash" not in rlm_env.tool_map


class TestToolSplitConfiguration:
    def test_tool_name_collision_raises(self):
        def tool_a() -> str:
            return "a"

        def tool_b() -> str:
            return "b"

        tool_b.__name__ = tool_a.__name__

        dataset = make_dataset({})
        with pytest.raises(ValueError, match="collision"):
            build_env(dataset, tools=[tool_a, tool_b])

    def test_fixed_tool_override_raises(self):
        def llm_batch() -> str:  # pragma: no cover - name collision test
            return "override"

        dataset = make_dataset({})
        with pytest.raises(ValueError, match="llm_batch"):
            build_env(dataset, tools=[llm_batch])

    def test_tools_not_exposed_as_environment_tool_defs(self):
        def shared_tool() -> str:
            return "shared"

        def root_tool() -> str:
            return "root"

        def sub_tool() -> str:
            return "sub"

        dataset = make_dataset({})
        env = build_env(
            dataset, tools=[shared_tool], root_tools=[root_tool], sub_tools=[sub_tool]
        )

        tool_names = {tool.name for tool in env.tool_defs}
        assert "shared_tool" not in tool_names
        assert "root_tool" not in tool_names
        assert "sub_tool" not in tool_names

    @pytest.mark.asyncio
    async def test_root_and_sub_tools_documented_and_ordered(self):
        def shared_tool() -> str:
            """Shared tool."""
            return "shared"

        def root_tool() -> str:
            """Root-only tool."""
            return "root"

        def sub_tool() -> str:
            """Sub-only tool."""
            return "sub"

        dataset = make_dataset({})
        env = build_env(
            dataset,
            tools=[shared_tool],
            root_tools=[root_tool],
            sub_tools=[sub_tool],
            interception_url="http://test.invalid",
        )
        env._ensure_interception_server = AsyncMock()
        env._executor.prepare_filesystem = AsyncMock()
        env._executor.setup = AsyncMock()

        state = {"info": {}, "model": "test-model", "client": MagicMock()}
        result = await env.setup_state(state)
        try:
            prompt = result["rlm_system_prompt"]
            assert "Root REPL Tools" in prompt
            assert "Sub-LLM Tools" in prompt

            root_index = prompt.find("Root REPL Tools")
            sub_index = prompt.find("Sub-LLM Tools")
            assert root_index != -1
            assert sub_index != -1
            assert root_index < sub_index

            root_section = prompt[root_index:sub_index]
            sub_section = prompt[sub_index:]

            assert "llm_batch" in root_section
            assert root_section.find("llm_batch") < root_section.find("shared_tool")
            assert root_section.find("shared_tool") < root_section.find("root_tool")

            assert "shared_tool" in sub_section
            assert "sub_tool" in sub_section
            assert "root_tool" not in sub_section
            assert sub_section.find("shared_tool") < sub_section.find("sub_tool")

            assert result["rlm_shared_tools"] == ["shared_tool"]
            assert result["rlm_root_tools"] == [
                "llm_batch",
                "shared_tool",
                "root_tool",
            ]
            assert result["rlm_sub_tools"] == ["shared_tool", "sub_tool"]
        finally:
            await env.cleanup_rlm_state(result)


# =============================================================================
# 4. Stop Conditions
# =============================================================================


class TestStopConditions:
    @pytest.mark.asyncio
    async def test_answer_ready_true(self, rlm_env):
        state = {"final_answer": "42"}
        result = await rlm_env.answer_ready(state)
        assert result is True

    @pytest.mark.asyncio
    async def test_answer_ready_false(self, rlm_env):
        state = {}
        result = await rlm_env.answer_ready(state)
        assert result is False


# =============================================================================
# 5. Context Limit Warning
# =============================================================================


class TestContextLimitWarning:
    @pytest.mark.asyncio
    async def test_no_warning_when_max_seq_len_not_set(self, rlm_env):
        rlm_env.max_seq_len = None
        rlm_env._execute_code = AsyncMock(
            return_value={
                "status": "ok",
                "stdout": "output",
                "stderr": "",
                "result": None,
                "execution_count": 1,
                "answer": {"ready": False, "content": ""},
            }
        )

        state = {"trajectory": [], "context_warning_sent": False}
        output = await rlm_env.call_python_repl("print('test')", state)

        assert "[CONTEXT LIMIT WARNING]" not in output
        assert state["context_warning_sent"] is False

    @pytest.mark.asyncio
    async def test_warning_at_threshold(self, rlm_env):
        rlm_env.max_seq_len = 10000
        rlm_env._execute_code = AsyncMock(
            return_value={
                "status": "ok",
                "stdout": "output",
                "stderr": "",
                "result": None,
                "execution_count": 1,
                "answer": {"ready": False, "content": ""},
            }
        )

        mock_response = MagicMock()
        mock_response.usage = MagicMock(prompt_tokens=8000)
        state = {
            "trajectory": [{"response": mock_response}],
            "context_warning_sent": False,
        }

        output = await rlm_env.call_python_repl("print('test')", state)

        assert "[CONTEXT LIMIT WARNING]" in output
        assert "8,000" in output
        assert "10,000" in output
        assert "80%" in output
        assert state["context_warning_sent"] is True

    @pytest.mark.asyncio
    async def test_bash_warning_at_threshold(self, rlm_env_bash):
        rlm_env_bash.max_seq_len = 10000
        rlm_env_bash._execute_code = AsyncMock(
            return_value={
                "status": "ok",
                "stdout": "output",
                "stderr": "",
                "result": None,
                "execution_count": 1,
                "answer": {"ready": False, "content": ""},
            }
        )

        mock_response = MagicMock()
        mock_response.usage = MagicMock(prompt_tokens=8000)
        state = {
            "trajectory": [{"response": mock_response}],
            "context_warning_sent": False,
        }

        output = await rlm_env_bash.call_bash_repl("echo test", state)

        assert "[CONTEXT LIMIT WARNING]" in output
        assert "8,000" in output
        assert "10,000" in output
        assert "80%" in output
        assert "RLM_READY=1" in output
        assert state["context_warning_sent"] is True


# =============================================================================
# 6. Sub-LLM Tool Infrastructure
# =============================================================================


class TestCallSubTool:
    @pytest.mark.asyncio
    async def test_executes_tool_successfully(self, rlm_env_with_sub_tools):
        result = await rlm_env_with_sub_tools._call_sub_tool(
            "sample_tool", {"x": 2, "y": 3}, "call_123"
        )

        assert result["role"] == "tool"
        assert result["content"] == "5"
        assert result["tool_call_id"] == "call_123"

    @pytest.mark.asyncio
    async def test_handles_tool_error(self, rlm_env_with_sub_tools):
        result = await rlm_env_with_sub_tools._call_sub_tool(
            "sample_tool", {"x": "not_an_int", "y": 3}, "call_456"
        )

        assert result["role"] == "tool"
        assert "Error" in result["content"]
        assert result["tool_call_id"] == "call_456"


class TestRunSubLLMWithTools:
    @pytest.mark.asyncio
    async def test_completes_without_tool_calls(self, rlm_env_with_sub_tools):
        from verifiers.types import Response, ResponseMessage

        mock_client = MagicMock()
        mock_client.get_response = AsyncMock(
            return_value=Response(
                id="mock",
                created=0,
                model="gpt-4",
                usage=None,
                message=ResponseMessage(
                    content="Final answer",
                    reasoning_content=None,
                    finish_reason="stop",
                    is_truncated=False,
                    tokens=None,
                    tool_calls=None,
                ),
            )
        )

        messages = [{"role": "user", "content": "Test"}]
        state = {}
        result = await rlm_env_with_sub_tools._run_sub_llm(
            state, mock_client, "gpt-4", messages
        )

        assert result["final_content"] == "Final answer"
        assert result["tool_call_count"] == 0
        assert result["num_turns"] == 1
        assert result["max_turns_reached"] is False
        assert len(result["turns"]) == 1

    @pytest.mark.asyncio
    async def test_executes_tool_calls(self, rlm_env_with_sub_tools):
        from verifiers.types import Response, ResponseMessage, ToolCall

        resp1 = Response(
            id="mock1",
            created=0,
            model="gpt-4",
            usage=None,
            message=ResponseMessage(
                content=None,
                reasoning_content=None,
                finish_reason="stop",
                is_truncated=False,
                tokens=None,
                tool_calls=[
                    ToolCall(
                        id="call_1", name="sample_tool", arguments='{"x": 2, "y": 3}'
                    )
                ],
            ),
        )
        resp2 = Response(
            id="mock2",
            created=0,
            model="gpt-4",
            usage=None,
            message=ResponseMessage(
                content="The result is 5",
                reasoning_content=None,
                finish_reason="stop",
                is_truncated=False,
                tokens=None,
                tool_calls=None,
            ),
        )
        mock_client = MagicMock()
        mock_client.get_response = AsyncMock(side_effect=[resp1, resp2])

        messages = [{"role": "user", "content": "Add 2 and 3"}]
        state = {}
        await rlm_env_with_sub_tools._run_sub_llm(state, mock_client, "gpt-4", messages)

        assert mock_client.get_response.call_count == 2


# =============================================================================
# 7. Sub-LLM Request Paths
# =============================================================================


class TestSubLLMRequestPaths:
    @pytest.mark.asyncio
    async def test_sub_llm_ignores_interleaving_and_uses_chat(self, rlm_env):
        from verifiers.types import Response, ResponseMessage

        mock_client = MagicMock()
        mock_client.get_response = AsyncMock(
            return_value=Response(
                id="mock",
                created=0,
                model="gpt-4",
                usage=None,
                message=ResponseMessage(
                    content="ok",
                    reasoning_content=None,
                    finish_reason="stop",
                    is_truncated=False,
                    tokens=None,
                    tool_calls=None,
                ),
            )
        )

        messages = [{"role": "user", "content": "hi"}]
        state = {"sampling_args": {"max_tokens": 7}}

        await rlm_env._call_sub_llm_api(state, mock_client, "gpt-4", messages)

        mock_client.get_response.assert_awaited_once()
        call_kwargs = mock_client.get_response.call_args.kwargs
        # sampling_args should have max_tokens (from state["sampling_args"]["max_tokens"])
        assert call_kwargs["sampling_args"]["max_tokens"] == 7


# =============================================================================
# 8. llm_batch Prompt Validation
# =============================================================================


class TestLLMBatchPromptValidation:
    @pytest.mark.asyncio
    async def test_llm_batch_rejects_non_string_prompts(self, rlm_env):
        context = {
            "client": MagicMock(),
            "sub_model": "gpt-4",
            "state": {"trajectory": []},
        }

        contents, _ = await rlm_env._root_llm_batch(
            context, [{"role": "user", "content": "hi"}]
        )
        assert "must be a string" in contents[0]

        contents, _ = await rlm_env._root_llm_batch(
            context, [[{"role": "user", "content": "hi"}]]
        )
        assert "must be a string" in contents[0]


# =============================================================================
# 9. Root Tool Serialization (pickle)
# =============================================================================


class TestRootToolSerialization:
    @pytest.mark.asyncio
    async def test_root_tool_request_uses_pickle(self):
        def echo_tool(value):
            return value

        dataset = make_dataset({})
        env = build_env(dataset, root_tools=[echo_tool])

        rollout_id = "rlm_root_tool_test"
        state = {}
        env.active_rollouts[rollout_id] = {
            "client": MagicMock(),
            "model": "test-model",
            "sub_model": "test-model",
            "state": state,
        }

        payload = {"value": 123}
        args_payload = base64.b64encode(pickle.dumps((payload,))).decode("ascii")
        kwargs_payload = base64.b64encode(pickle.dumps({})).decode("ascii")

        mock_request = MagicMock()
        mock_request.match_info = {"rollout_id": rollout_id}
        mock_request.json = AsyncMock(
            return_value={
                "tool_name": "echo_tool",
                "serialization": "pickle",
                "args": args_payload,
                "kwargs": kwargs_payload,
            }
        )

        response = await env._handle_root_tool_request(mock_request)
        assert response.status == 200

        response_data = json.loads(response.text)
        result_payload = response_data["result"]
        decoded = pickle.loads(base64.b64decode(result_payload))
        assert decoded == payload


# =============================================================================
# 10. Context Limit Configuration
# =============================================================================


class TestContextLimitConfiguration:
    def test_default_threshold(self, rlm_env):
        assert rlm_env.context_warning_threshold == 0.80

    def test_custom_threshold(self):
        dataset = make_dataset({})
        env = build_env(dataset, context_warning_threshold=0.70)
        assert env.context_warning_threshold == 0.70


# =============================================================================
# 11. Sub-LLM Metrics with Tools
# =============================================================================


class TestSubLLMMetricsWithTools:
    @pytest.mark.asyncio
    async def test_accumulates_tokens_across_tool_turns(self, rlm_env_with_sub_tools):
        from verifiers.types import Response, ResponseMessage, ToolCall, Usage

        resp1 = Response(
            id="mock1",
            created=0,
            model="gpt-4",
            usage=Usage(
                prompt_tokens=50,
                reasoning_tokens=0,
                completion_tokens=30,
                total_tokens=80,
            ),
            message=ResponseMessage(
                content=None,
                reasoning_content=None,
                finish_reason="stop",
                is_truncated=False,
                tokens=None,
                tool_calls=[
                    ToolCall(
                        id="call_1", name="sample_tool", arguments='{"x": 2, "y": 3}'
                    )
                ],
            ),
        )
        resp2 = Response(
            id="mock2",
            created=0,
            model="gpt-4",
            usage=Usage(
                prompt_tokens=100,
                reasoning_tokens=0,
                completion_tokens=20,
                total_tokens=120,
            ),
            message=ResponseMessage(
                content="The result is 5",
                reasoning_content=None,
                finish_reason="stop",
                is_truncated=False,
                tokens=None,
                tool_calls=None,
            ),
        )
        mock_client = MagicMock()
        mock_client.get_response = AsyncMock(side_effect=[resp1, resp2])

        messages = [{"role": "user", "content": "Add 2 and 3"}]
        state = {}
        result = await rlm_env_with_sub_tools._run_sub_llm(
            state, mock_client, "gpt-4", messages
        )

        assert result["total_prompt_tokens"] == 150
        assert result["total_completion_tokens"] == 50
        assert result["tool_call_count"] == 1
        assert result["num_turns"] == 2
        assert result["max_turns_reached"] is False
        assert len(result["turns"]) == 2


# =============================================================================
# 12. Sub-LLM Trajectory Steps
# =============================================================================


class TestSubLLMTrajectorySteps:
    @pytest.mark.asyncio
    async def test_include_sub_llm_in_trajectory_default(self, rlm_env):
        assert rlm_env.include_sub_llm_in_trajectory is False

    def test_interleaved_allowed_when_sub_llm_in_trajectory(self):
        dataset = make_dataset({})
        env = build_env(
            dataset,
            include_sub_llm_in_trajectory=True,
        )
        assert env.include_sub_llm_in_trajectory is True

    @pytest.mark.asyncio
    async def test_sub_llm_steps_added_to_trajectory(self, rlm_env):
        rlm_env.include_sub_llm_in_trajectory = True
        state = {"trajectory": [], "sampling_args": {}}

        mock_message = MagicMock()
        mock_message.tool_calls = None
        mock_message.content = "ok"
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=mock_message)]
        mock_response.usage = MagicMock(prompt_tokens=1, completion_tokens=1)

        result = {
            "final_content": "ok",
            "turns": [
                {
                    "prompt_messages": [{"role": "user", "content": "hi"}],
                    "response": mock_response,
                    "tool_call_count": 0,
                }
            ],
            "total_prompt_tokens": 1,
            "total_completion_tokens": 1,
            "tool_call_count": 0,
            "num_turns": 1,
            "max_turns_reached": False,
        }

        token_payload = {
            "prompt_ids": [1],
            "prompt_mask": [0],
            "completion_ids": [2],
            "completion_mask": [1],
            "completion_logprobs": [0.0],
            "overlong_prompt": False,
            "is_truncated": False,
        }

        with (
            patch.object(rlm_env, "_run_sub_llm", new=AsyncMock(return_value=result)),
            patch(
                "verifiers.envs.experimental.rlm_env.parse_response_tokens",
                new=AsyncMock(return_value=token_payload),
            ),
            patch(
                "verifiers.envs.experimental.rlm_env.parse_response_message",
                new=AsyncMock(return_value=[{"role": "assistant", "content": "ok"}]),
            ),
        ):
            await rlm_env._run_sub_llm_request(
                state_ref=state,
                client=MagicMock(),
                sub_model="gpt-4",
                messages=[{"role": "user", "content": "hi"}],
                batch_id="b1",
                request_id="r1",
                parent_turn=0,
            )

        assert len(state["trajectory"]) == 1
        assert state["trajectory"][0]["trajectory_id"] == "b1_r1"
        assert state["trajectory"][0]["extras"]["is_sub_llm_call"] is True


# =============================================================================
# 13. Tunnel Utils (kept for coverage)
# =============================================================================


class TestExtractTunnelUrlFromLine:
    def test_extract_valid_url(self):
        from verifiers.utils.tunnel_utils import extract_tunnel_url_from_line

        line = (
            "2024-01-01 12:00:00 INF https://random-words.trycloudflare.com registered"
        )
        url = extract_tunnel_url_from_line(line)
        assert url == "https://random-words.trycloudflare.com"

    def test_return_none_for_no_url(self):
        from verifiers.utils.tunnel_utils import extract_tunnel_url_from_line

        line = "Starting cloudflared tunnel..."
        url = extract_tunnel_url_from_line(line)
        assert url is None

    def test_handle_trailing_characters(self):
        from verifiers.utils.tunnel_utils import extract_tunnel_url_from_line

        line = "https://test-tunnel.trycloudflare.com/path?query=1 some text"
        url = extract_tunnel_url_from_line(line)
        assert url is not None
        assert url.startswith("https://")
        assert ".trycloudflare.com" in url

    def test_no_https_prefix(self):
        from verifiers.utils.tunnel_utils import extract_tunnel_url_from_line

        line = "something.trycloudflare.com without https"
        url = extract_tunnel_url_from_line(line)
        assert url is None


# =============================================================================
# 14. RLM Exception Hierarchy
# =============================================================================


class TestExceptionHierarchy:
    """Verify that RLM exceptions inherit from the correct verifiers base classes."""

    def test_rlm_session_error_is_sandbox_error(self):
        assert issubclass(RLMSessionError, vf.SandboxError)

    def test_rlm_setup_error_is_sandbox_error(self):
        assert issubclass(RLMSetupError, vf.SandboxError)

    def test_rlm_worker_error_is_sandbox_error(self):
        assert issubclass(RLMWorkerError, vf.SandboxError)

    def test_rlm_worker_recovery_error_is_worker_error(self):
        assert issubclass(RLMWorkerRecoveryError, RLMWorkerError)

    def test_rlm_code_execution_timeout_is_tool_call_error(self):
        assert issubclass(RLMCodeExecutionTimeout, vf.ToolCallError)

    def test_sub_llm_empty_response_is_empty_model_response_error(self):
        assert issubclass(SubLLMEmptyModelResponseError, vf.EmptyModelResponseError)

    def test_all_are_vf_errors(self):
        """All RLM exceptions should be caught by the rollout loop's except vf.Error."""
        for exc_cls in (
            RLMSessionError,
            RLMSetupError,
            RLMWorkerError,
            RLMWorkerRecoveryError,
            RLMCodeExecutionTimeout,
            SubLLMEmptyModelResponseError,
        ):
            assert issubclass(exc_cls, vf.Error), (
                f"{exc_cls.__name__} is not a vf.Error"
            )


class TestRLMSessionErrorRaised:
    """Test that RLMSessionError is raised when sessions/sandboxes are not initialized."""

    def test_get_session_missing_rollout_id(self, rlm_env):
        executor = rlm_env._executor
        state = {}
        with pytest.raises(RLMSessionError, match="Sandbox session not initialized"):
            executor._get_session(state)

    def test_get_session_unknown_rollout_id(self, rlm_env):
        executor = rlm_env._executor
        state = {"rollout_id": "nonexistent"}
        with pytest.raises(RLMSessionError, match="Sandbox session not initialized"):
            executor._get_session(state)


class TestRLMCodeExecutionTimeoutHandling:
    """Test the abort and recovery paths for code execution timeout."""

    @pytest.mark.asyncio
    async def test_abort_on_timeout_raises_timeout_directly(self, rlm_env):
        rlm_env.abort_on_code_timeout = True
        rlm_env._executor.execute = AsyncMock(
            side_effect=RLMCodeExecutionTimeout("timed out")
        )
        rlm_env._executor.prepare_filesystem = AsyncMock()
        rlm_env._executor.setup = AsyncMock()

        state = {"rlm_worker_ready": True, "_exec_seq": 0}
        with pytest.raises(RLMCodeExecutionTimeout):
            await rlm_env._execute_code("import time; time.sleep(999)", state)

    @pytest.mark.asyncio
    async def test_recovery_failure_raises_worker_recovery_error(self, rlm_env):
        rlm_env.abort_on_code_timeout = False
        rlm_env._executor.execute = AsyncMock(
            side_effect=RLMCodeExecutionTimeout("timed out")
        )
        rlm_env._executor.prepare_filesystem = AsyncMock()
        rlm_env._executor.setup = AsyncMock()
        rlm_env._recover_from_code_timeout = AsyncMock(return_value=False)

        state = {"rlm_worker_ready": True, "_exec_seq": 0}
        with pytest.raises(RLMWorkerRecoveryError, match="could not be restarted"):
            await rlm_env._execute_code("import time; time.sleep(999)", state)

    @pytest.mark.asyncio
    async def test_recovery_success_returns_error_result(self, rlm_env):
        rlm_env.abort_on_code_timeout = False
        rlm_env._executor.execute = AsyncMock(
            side_effect=RLMCodeExecutionTimeout("timed out")
        )
        rlm_env._executor.prepare_filesystem = AsyncMock()
        rlm_env._executor.setup = AsyncMock()
        rlm_env._recover_from_code_timeout = AsyncMock(return_value=True)

        state = {"rlm_worker_ready": True, "_exec_seq": 0}
        result = await rlm_env._execute_code("slow_code()", state)
        assert result["status"] == "error"
        assert "timed out" in result["result"]


class TestSubLLMEmptyModelResponseErrorRaised:
    """Test that SubLLMEmptyModelResponseError is raised for empty sub-LLM responses."""

    @pytest.mark.asyncio
    async def test_empty_response_from_sub_llm(self, rlm_env):
        with patch.object(
            rlm_env,
            "get_model_response",
            new=AsyncMock(
                side_effect=vf.EmptyModelResponseError("Model returned no response")
            ),
        ):
            state = {"sampling_args": {}}
            messages = [{"role": "user", "content": "hello"}]
            with pytest.raises(SubLLMEmptyModelResponseError, match="no response"):
                await rlm_env._call_sub_llm_api(state, MagicMock(), "gpt-4", messages)

    @pytest.mark.asyncio
    async def test_sub_llm_empty_response_chains_cause(self, rlm_env):
        original = vf.EmptyModelResponseError("original error")
        with patch.object(
            rlm_env,
            "get_model_response",
            new=AsyncMock(side_effect=original),
        ):
            state = {"sampling_args": {}}
            messages = [{"role": "user", "content": "hello"}]
            with pytest.raises(SubLLMEmptyModelResponseError) as exc_info:
                await rlm_env._call_sub_llm_api(state, MagicMock(), "gpt-4", messages)
            assert exc_info.value.__cause__ is original


# =============================================================================
# _InterceptionPool tests
# =============================================================================


class TestInterceptionPool:
    """Tests for the shared _InterceptionPool."""

    @pytest.fixture(autouse=True)
    def fresh_pool(self):
        """Each test gets a fresh pool."""
        self.pool = _InterceptionPool()

    @pytest.mark.asyncio
    async def test_acquire_creates_server(self):
        await self.pool.acquire_server(0, "127.0.0.1")
        assert 0 in self.pool._entries
        entry = self.pool._entries[0]
        assert entry.refcount == 1
        assert entry.server_app is not None
        assert entry.server_site is not None
        await self.pool.release(0)

    @pytest.mark.asyncio
    async def test_acquire_twice_increments_refcount(self):
        await self.pool.acquire_server(0, "127.0.0.1")
        await self.pool.acquire_server(0, "127.0.0.1")
        assert self.pool._entries[0].refcount == 2
        await self.pool.release(0)
        assert self.pool._entries[0].refcount == 1
        await self.pool.release(0)
        assert 0 not in self.pool._entries

    @pytest.mark.asyncio
    async def test_release_destroys_at_zero(self):
        await self.pool.acquire_server(0, "127.0.0.1")
        await self.pool.release(0)
        assert 0 not in self.pool._entries

    @pytest.mark.asyncio
    async def test_double_release_does_not_crash(self):
        await self.pool.acquire_server(0, "127.0.0.1")
        await self.pool.release(0)
        await self.pool.release(0)  # should not raise
        assert 0 not in self.pool._entries

    @pytest.mark.asyncio
    async def test_register_and_unregister_rollout(self):
        await self.pool.acquire_server(0, "127.0.0.1")
        sentinel = object()
        self.pool.register_rollout(0, "roll_1", sentinel)
        assert self.pool._entries[0].rollout_dispatch["roll_1"] is sentinel
        self.pool.unregister_rollout(0, "roll_1")
        assert "roll_1" not in self.pool._entries[0].rollout_dispatch
        await self.pool.release(0)

    @pytest.mark.asyncio
    async def test_unregister_missing_rollout_does_not_crash(self):
        await self.pool.acquire_server(0, "127.0.0.1")
        self.pool.unregister_rollout(0, "nonexistent")  # should not raise
        await self.pool.release(0)

    @pytest.mark.asyncio
    async def test_dispatch_routes_to_correct_env(self):
        from aiohttp import web
        from aiohttp.test_utils import TestClient, TestServer

        await self.pool.acquire_server(0, "127.0.0.1")
        entry = self.pool._entries[0]

        # Mock two env instances with different handlers
        env_a = MagicMock()
        env_a._handle_sub_llm_request = AsyncMock(
            return_value=web.json_response({"from": "a"})
        )
        env_b = MagicMock()
        env_b._handle_sub_llm_request = AsyncMock(
            return_value=web.json_response({"from": "b"})
        )

        self.pool.register_rollout(0, "roll_a", env_a)
        self.pool.register_rollout(0, "roll_b", env_b)

        server = TestServer(entry.server_app)
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.post("/rollout/roll_a/v1/chat/completions", json={})
            assert resp.status == 200
            env_a._handle_sub_llm_request.assert_called_once()
            env_b._handle_sub_llm_request.assert_not_called()
        finally:
            await client.close()

        await self.pool.release(0)

    @pytest.mark.asyncio
    async def test_dispatch_returns_404_for_unknown_rollout(self):
        from aiohttp.test_utils import TestClient, TestServer

        await self.pool.acquire_server(0, "127.0.0.1")
        entry = self.pool._entries[0]

        server = TestServer(entry.server_app)
        client = TestClient(server)
        await client.start_server()
        try:
            resp = await client.post("/rollout/unknown/v1/chat/completions", json={})
            assert resp.status == 404
        finally:
            await client.close()

        await self.pool.release(0)

    @pytest.mark.asyncio
    async def test_get_tunnel_url_without_server_raises(self):
        with pytest.raises(RuntimeError, match="No server on port"):
            await self.pool.get_tunnel_url(9999)

    @pytest.mark.asyncio
    async def test_get_tunnel_url_reuses_tunnel(self):
        await self.pool.acquire_server(0, "127.0.0.1")
        mock_tunnel = MagicMock()
        mock_tunnel.is_running = True
        mock_tunnel.url = "https://test-tunnel.example.com"
        mock_tunnel.start = AsyncMock(return_value="https://test-tunnel.example.com")
        mock_tunnel.stop = AsyncMock()

        with patch(
            "verifiers.envs.experimental.rlm_env.Tunnel",
            return_value=mock_tunnel,
        ):
            url1 = await self.pool.get_tunnel_url(0)
            url2 = await self.pool.get_tunnel_url(0)
            assert url1 == url2
            # Tunnel constructor called only once
            mock_tunnel.start.assert_called_once()

        await self.pool.release(0)

    @pytest.mark.asyncio
    async def test_get_tunnel_url_recreates_dead_tunnel(self):
        await self.pool.acquire_server(0, "127.0.0.1")

        dead_tunnel = MagicMock()
        dead_tunnel.is_running = False
        dead_tunnel.stop = AsyncMock()

        new_tunnel = MagicMock()
        new_tunnel.is_running = True
        new_tunnel.url = "https://new-tunnel.example.com"
        new_tunnel.start = AsyncMock(return_value="https://new-tunnel.example.com")

        # Inject a dead tunnel into the entry
        entry = self.pool._entries[0]
        entry.tunnel = dead_tunnel
        entry.tunnel_url = "https://old-tunnel.example.com"

        with patch(
            "verifiers.envs.experimental.rlm_env.Tunnel",
            return_value=new_tunnel,
        ):
            url = await self.pool.get_tunnel_url(0)
            assert url == "https://new-tunnel.example.com"
            dead_tunnel.stop.assert_called_once()
            new_tunnel.start.assert_called_once()

        await self.pool.release(0)

    @pytest.mark.asyncio
    async def test_concurrent_acquire_no_double_increment(self):
        """Verify that concurrent rollouts in one RLMEnv don't double-acquire."""
        dataset = make_dataset({})
        env = build_env(
            dataset,
            interception_port=18080,
            interception_url="http://test.invalid",
        )
        # Replace the global pool with our fresh one
        with patch("verifiers.envs.experimental.rlm_env._interception_pool", self.pool):
            # Simulate concurrent _ensure_interception_server calls
            import asyncio

            await asyncio.gather(
                env._ensure_interception_server(),
                env._ensure_interception_server(),
                env._ensure_interception_server(),
            )
            assert self.pool._entries[18080].refcount == 1
            assert env._has_pool_ref is True
            await self.pool.release(18080)


# =============================================================================
# Message History Upload
# =============================================================================


class TestMessageHistory:
    """Tests for expose_message_history feature."""

    @pytest.fixture
    def env_with_history(self) -> RLMEnv:
        dataset = make_dataset({})
        return build_env(
            dataset,
            repl_language="python",
            expose_message_history=True,
            interception_url="http://test.invalid",
        )

    @pytest.fixture
    def env_without_history(self) -> RLMEnv:
        dataset = make_dataset({})
        return build_env(
            dataset,
            repl_language="python",
            expose_message_history=False,
            interception_url="http://test.invalid",
        )

    def _make_state_with_trajectory(
        self, env: RLMEnv, messages_per_step: int = 2, num_steps: int = 1
    ) -> dict:
        """Build a state dict with a realistic trajectory."""
        trajectory_id = "main_traj"
        trajectory = []
        for step_idx in range(num_steps):
            prompt_msgs = [
                vf.UserMessage(content=f"Step {step_idx} user message {i}")
                for i in range(messages_per_step)
            ]
            completion_msgs = [
                vf.AssistantMessage(content=f"Step {step_idx} assistant response")
            ]
            trajectory.append(
                {
                    "prompt": prompt_msgs,
                    "completion": completion_msgs,
                    "response": None,
                    "tokens": None,
                    "reward": None,
                    "advantage": None,
                    "is_truncated": False,
                    "trajectory_id": trajectory_id,
                    "extras": {},
                }
            )
        return {
            "trajectory": trajectory,
            "trajectory_id": trajectory_id,
            "_messages_uploaded_count": 0,
        }

    def test_build_message_history_empty_trajectory(self, env_with_history):
        state = {
            "trajectory": [],
            "trajectory_id": "main",
            "_messages_uploaded_count": 0,
        }
        result = env_with_history._build_message_history(state)
        assert result == []

    def test_build_message_history_one_step(self, env_with_history):
        state = self._make_state_with_trajectory(
            env_with_history, messages_per_step=1, num_steps=1
        )
        result = env_with_history._build_message_history(state)
        # 1 prompt message + 1 completion message = 2 messages
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Step 0 user message 0"
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "Step 0 assistant response"

    def test_build_message_history_multi_step(self, env_with_history):
        state = self._make_state_with_trajectory(
            env_with_history, messages_per_step=2, num_steps=3
        )
        result = env_with_history._build_message_history(state)
        # Last step: 2 prompt messages + 1 completion = 3
        assert len(result) == 3
        # Should be from the last step
        assert result[0]["content"] == "Step 2 user message 0"
        assert result[1]["content"] == "Step 2 user message 1"
        assert result[2]["content"] == "Step 2 assistant response"

    def test_build_message_history_skips_sub_llm_steps(self, env_with_history):
        main_id = "main_traj"
        sub_id = "sub_batch_1"
        trajectory = [
            {
                "prompt": [vf.UserMessage(content="main prompt")],
                "completion": [vf.AssistantMessage(content="main response")],
                "response": None,
                "tokens": None,
                "reward": None,
                "advantage": None,
                "is_truncated": False,
                "trajectory_id": main_id,
                "extras": {},
            },
            {
                "prompt": [vf.UserMessage(content="sub prompt")],
                "completion": [vf.AssistantMessage(content="sub response")],
                "response": None,
                "tokens": None,
                "reward": None,
                "advantage": None,
                "is_truncated": False,
                "trajectory_id": sub_id,
                "extras": {"is_sub_llm_call": True},
            },
        ]
        state = {
            "trajectory": trajectory,
            "trajectory_id": main_id,
            "_messages_uploaded_count": 0,
        }
        result = env_with_history._build_message_history(state)
        # Should only include the main step's messages
        assert len(result) == 2
        assert result[0]["content"] == "main prompt"
        assert result[1]["content"] == "main response"

    def test_incremental_delta_computation(self, env_with_history):
        """Verify that only new messages are uploaded on subsequent calls."""
        main_id = "main_traj"
        # Step 0: 1 user + 1 assistant = 2 messages
        step0 = {
            "prompt": [vf.UserMessage(content="q0")],
            "completion": [vf.AssistantMessage(content="a0")],
            "response": None,
            "tokens": None,
            "reward": None,
            "advantage": None,
            "is_truncated": False,
            "trajectory_id": main_id,
            "extras": {},
        }
        state = {
            "trajectory": [step0],
            "trajectory_id": main_id,
            "_messages_uploaded_count": 0,
        }

        # First call: all messages are new
        messages = env_with_history._build_message_history(state)
        uploaded_count = state["_messages_uploaded_count"]
        new_messages = messages[uploaded_count:]
        assert len(new_messages) == 2

        # Simulate upload completing
        state["_messages_uploaded_count"] = len(messages)

        # Step 1: prompt = [q0, a0, tool_result], completion = [a1] = 4 messages total
        step1 = {
            "prompt": [
                vf.UserMessage(content="q0"),
                vf.AssistantMessage(content="a0"),
                vf.ToolMessage(tool_call_id="tc1", content="tool output"),
            ],
            "completion": [vf.AssistantMessage(content="a1")],
            "response": None,
            "tokens": None,
            "reward": None,
            "advantage": None,
            "is_truncated": False,
            "trajectory_id": main_id,
            "extras": {},
        }
        state["trajectory"].append(step1)

        # Second call: only the new messages
        messages = env_with_history._build_message_history(state)
        new_messages = messages[state["_messages_uploaded_count"] :]
        # Delta = 4 total - 2 already uploaded = 2 new messages
        assert len(new_messages) == 2
        assert new_messages[0]["role"] == "tool"
        assert new_messages[1]["role"] == "assistant"
        assert new_messages[1]["content"] == "a1"

    @pytest.mark.asyncio
    async def test_upload_creates_file_on_first_call_with_empty_trajectory(
        self, env_with_history
    ):
        """First call with no trajectory should touch .messages to create it."""
        state = {
            "trajectory": [],
            "trajectory_id": "main",
            "_messages_uploaded_count": 0,
            "rollout_id": "test_rollout",
        }

        mock_session = MagicMock()
        mock_session.sandbox_id = "sandbox_123"
        mock_session.sandbox_fs_root = "/tmp/rlm_test/rlm_fs"
        env_with_history._executor._get_session = MagicMock(return_value=mock_session)
        env_with_history._executor._execute_sandbox_command = AsyncMock()

        await env_with_history._upload_message_history(state)

        env_with_history._executor._execute_sandbox_command.assert_called_once()
        cmd = env_with_history._executor._execute_sandbox_command.call_args[0][1]
        assert ".messages" in cmd
        assert "touch" in cmd
        assert state["_messages_uploaded_count"] == 0

    @pytest.mark.asyncio
    async def test_upload_message_history_calls_sandbox_command(self, env_with_history):
        """Verify _upload_message_history sends base64-encoded JSONL via sandbox command."""
        state = self._make_state_with_trajectory(
            env_with_history, messages_per_step=1, num_steps=1
        )
        state["rollout_id"] = "test_rollout"

        mock_session = MagicMock()
        mock_session.sandbox_id = "sandbox_123"
        mock_session.sandbox_fs_root = "/tmp/rlm_test/rlm_fs"
        env_with_history._executor._get_session = MagicMock(return_value=mock_session)
        env_with_history._executor._execute_sandbox_command = AsyncMock()

        await env_with_history._upload_message_history(state)

        # Should have called sandbox command
        env_with_history._executor._execute_sandbox_command.assert_called_once()
        call_args = env_with_history._executor._execute_sandbox_command.call_args
        assert call_args[0][0] == "sandbox_123"
        cmd = call_args[0][1]
        assert ".messages" in cmd
        assert "base64" in cmd

        # Counter should be updated
        assert state["_messages_uploaded_count"] == 2

    @pytest.mark.asyncio
    async def test_upload_skips_when_no_new_messages(self, env_with_history):
        """Verify no sandbox command when all messages already uploaded."""
        state = self._make_state_with_trajectory(
            env_with_history, messages_per_step=1, num_steps=1
        )
        state["rollout_id"] = "test_rollout"
        state["_messages_uploaded_count"] = 2  # Already uploaded all 2 messages

        mock_session = MagicMock()
        mock_session.sandbox_id = "sandbox_123"
        mock_session.sandbox_fs_root = "/tmp/rlm_test/rlm_fs"
        env_with_history._executor._get_session = MagicMock(return_value=mock_session)
        env_with_history._executor._execute_sandbox_command = AsyncMock()

        await env_with_history._upload_message_history(state)

        # Should NOT have called sandbox command
        env_with_history._executor._execute_sandbox_command.assert_not_called()

    @pytest.mark.asyncio
    async def test_upload_failure_is_non_fatal(self, env_with_history):
        """Verify that sandbox command failure doesn't raise."""
        state = self._make_state_with_trajectory(
            env_with_history, messages_per_step=1, num_steps=1
        )
        state["rollout_id"] = "test_rollout"

        mock_session = MagicMock()
        mock_session.sandbox_id = "sandbox_123"
        mock_session.sandbox_fs_root = "/tmp/rlm_test/rlm_fs"
        env_with_history._executor._get_session = MagicMock(return_value=mock_session)
        env_with_history._executor._execute_sandbox_command = AsyncMock(
            side_effect=RuntimeError("sandbox down")
        )

        # Should not raise
        await env_with_history._upload_message_history(state)

        # Counter should NOT be updated
        assert state["_messages_uploaded_count"] == 0

    @pytest.mark.asyncio
    async def test_system_prompt_includes_history_note_when_enabled(self):
        dataset = make_dataset({})
        env = build_env(
            dataset,
            repl_language="python",
            expose_message_history=True,
            interception_url="http://test.invalid",
        )
        env._ensure_interception_server = AsyncMock()
        env._executor.prepare_filesystem = AsyncMock()
        env._executor.setup = AsyncMock()

        state = {"info": {}, "model": "m", "client": MagicMock()}
        result = await env.setup_state(state)
        try:
            prompt = result["rlm_system_prompt"]
            assert ".messages" in prompt
            assert "JSONL" in prompt
        finally:
            await env.cleanup_rlm_state(result)

    @pytest.mark.asyncio
    async def test_system_prompt_excludes_history_note_when_disabled(self):
        dataset = make_dataset({})
        env = build_env(
            dataset,
            repl_language="python",
            expose_message_history=False,
            interception_url="http://test.invalid",
        )
        env._ensure_interception_server = AsyncMock()
        env._executor.prepare_filesystem = AsyncMock()
        env._executor.setup = AsyncMock()

        state = {"info": {}, "model": "m", "client": MagicMock()}
        result = await env.setup_state(state)
        try:
            prompt = result["rlm_system_prompt"]
            assert ".messages" not in prompt
        finally:
            await env.cleanup_rlm_state(result)

    @pytest.mark.asyncio
    async def test_system_prompt_bash_history_note(self):
        dataset = make_dataset({})
        env = build_env(
            dataset,
            repl_language="bash",
            expose_message_history=True,
            interception_url="http://test.invalid",
        )
        env._ensure_interception_server = AsyncMock()
        env._executor.prepare_filesystem = AsyncMock()
        env._executor.setup = AsyncMock()

        state = {"info": {}, "model": "m", "client": MagicMock()}
        result = await env.setup_state(state)
        try:
            prompt = result["rlm_system_prompt"]
            assert ".messages" in prompt
            assert "cat .messages" in prompt
        finally:
            await env.cleanup_rlm_state(result)

    def test_expose_message_history_defaults_to_false(self):
        dataset = make_dataset({})
        env = build_env(dataset, interception_url="http://test.invalid")
        assert env.expose_message_history is False

    @pytest.mark.asyncio
    async def test_setup_state_initializes_upload_counter(self):
        dataset = make_dataset({})
        env = build_env(
            dataset,
            expose_message_history=True,
            interception_url="http://test.invalid",
        )
        env._ensure_interception_server = AsyncMock()
        env._executor.prepare_filesystem = AsyncMock()
        env._executor.setup = AsyncMock()

        state = {"info": {}, "model": "m", "client": MagicMock()}
        result = await env.setup_state(state)
        try:
            assert result["_messages_uploaded_count"] == 0
        finally:
            await env.cleanup_rlm_state(result)
