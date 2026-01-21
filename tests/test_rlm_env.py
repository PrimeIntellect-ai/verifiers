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
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from datasets import Dataset
from verifiers.envs.experimental import rlm_env as rlm_module
from verifiers.envs.experimental.rlm_env import RLMEnv, RLMWorkerPaths


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
    )


@pytest.fixture
def rlm_env_bash() -> RLMEnv:
    dataset = make_dataset({})
    return build_env(
        dataset,
        max_iterations=10,
        max_output_length=1000,
        repl_language="bash",
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
        env = build_env(dataset)
        env._ensure_interception_server = AsyncMock()
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
        env = build_env(dataset)
        env._ensure_interception_server = AsyncMock()
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
        env = build_env(dataset)
        env._ensure_interception_server = AsyncMock()
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
        env = build_env(dataset)
        env._ensure_interception_server = AsyncMock()
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
        env = build_env(dataset, filesystem_copy_max_bytes=5)
        env._ensure_interception_server = AsyncMock()
        env._executor.setup = AsyncMock()

        state = {"info": {"context_dir": str(src)}, "model": "m", "client": MagicMock()}
        with pytest.raises(ValueError, match="exceeds size limit"):
            await env.setup_state(state)

    @pytest.mark.asyncio
    async def test_setup_state_no_context_creates_empty_dir(self):
        dataset = make_dataset({})
        env = build_env(dataset)
        env._ensure_interception_server = AsyncMock()
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
        env = build_env(dataset)
        env._ensure_interception_server = AsyncMock()
        env._executor.setup = AsyncMock()

        state = {"info": {}, "model": "m", "client": MagicMock()}
        result = await env.setup_state(state)
        try:
            prompt = result["rlm_system_prompt"]
            fs_root = result["rlm_fs_root"]
            assert f"Working directory: {fs_root}" in prompt
            assert "No extra data was provided" in prompt
            assert "can still use this directory" in prompt
        finally:
            await env.cleanup_rlm_state(result)


class TestFilesystemCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_removes_filesystem_by_default(self, tmp_path: Path):
        dataset = make_dataset({"context": "hello"})
        env = build_env(dataset)
        env._ensure_interception_server = AsyncMock()
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
        env = build_env(dataset, retain_filesystem_after_rollout=True)
        env._ensure_interception_server = AsyncMock()
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
        env._executor.setup = AsyncMock()

        state = {"info": {}, "model": "m", "client": MagicMock()}
        result = await env.setup_state(state)
        try:
            prompt = result["rlm_system_prompt"]
            assert "RLM_READY" in prompt
            assert "RLM_CONTENT" in prompt
        finally:
            await env.cleanup_rlm_state(result)


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
        assert '"$?"' in script
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

    def test_tools_not_exposed_as_openai_tools(self):
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

        tool_names = {tool["function"]["name"] for tool in env.oai_tools}
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
            dataset, tools=[shared_tool], root_tools=[root_tool], sub_tools=[sub_tool]
        )
        env._ensure_interception_server = AsyncMock()
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
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.tool_calls = None
        mock_message.content = "Final answer"
        mock_response.choices = [MagicMock(message=mock_message)]
        mock_response.model_dump = MagicMock(
            return_value={"choices": [{"message": {"content": "Final answer"}}]}
        )
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

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
        mock_client = MagicMock()

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function.name = "sample_tool"
        mock_tool_call.function.arguments = '{"x": 2, "y": 3}'

        mock_message1 = MagicMock()
        mock_message1.tool_calls = [mock_tool_call]
        mock_message1.content = None
        mock_message1.model_dump = MagicMock(
            return_value={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "sample_tool",
                            "arguments": '{"x": 2, "y": 3}',
                        },
                    }
                ],
            }
        )

        mock_message2 = MagicMock()
        mock_message2.tool_calls = None
        mock_message2.content = "The result is 5"

        mock_response1 = MagicMock()
        mock_response1.choices = [MagicMock(message=mock_message1)]

        mock_response2 = MagicMock()
        mock_response2.choices = [MagicMock(message=mock_message2)]
        mock_response2.model_dump = MagicMock(
            return_value={"choices": [{"message": {"content": "The result is 5"}}]}
        )

        mock_client.chat.completions.create = AsyncMock(
            side_effect=[mock_response1, mock_response2]
        )

        messages = [{"role": "user", "content": "Add 2 and 3"}]
        state = {}
        await rlm_env_with_sub_tools._run_sub_llm(state, mock_client, "gpt-4", messages)

        assert mock_client.chat.completions.create.call_count == 2


# =============================================================================
# 7. Sub-LLM Request Paths
# =============================================================================


class TestSubLLMRequestPaths:
    @pytest.mark.asyncio
    async def test_interleaved_uses_tokens_endpoint(self, rlm_env):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.chat.completions.create = AsyncMock()

        rlm_env.interleaved_rollouts = True
        messages = [{"role": "user", "content": "hi"}]
        state = {"sampling_args": {"max_tokens": 7, "extra_body": {"foo": "bar"}}}

        with patch(
            "verifiers.envs.experimental.rlm_env.tokenize_vllm",
            new=AsyncMock(return_value=[1, 2, 3]),
        ) as mock_tokenize:
            await rlm_env._call_sub_llm_api(state, mock_client, "gpt-4", messages)

        mock_tokenize.assert_awaited_once_with(
            client=mock_client,
            messages=messages,
            tools=None,
            model="gpt-4",
        )
        mock_client.post.assert_awaited_once()
        args, kwargs = mock_client.post.call_args
        assert args[0] == "/chat/completions/tokens"
        body = kwargs["body"]
        assert body["tokens"] == [1, 2, 3]
        assert body["max_completion_tokens"] == 7
        assert body["return_token_ids"] is True
        assert body["foo"] == "bar"
        assert "max_tokens" not in body
        mock_client.chat.completions.create.assert_not_called()


# =============================================================================
# 8. Root Tool Serialization (pickle)
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
# 9. Context Limit Configuration
# =============================================================================


class TestContextLimitConfiguration:
    def test_default_threshold(self, rlm_env):
        assert rlm_env.context_warning_threshold == 0.80

    def test_custom_threshold(self):
        dataset = make_dataset({})
        env = build_env(dataset, context_warning_threshold=0.70)
        assert env.context_warning_threshold == 0.70


# =============================================================================
# 10. Sub-LLM Metrics with Tools
# =============================================================================


class TestSubLLMMetricsWithTools:
    @pytest.mark.asyncio
    async def test_accumulates_tokens_across_tool_turns(self, rlm_env_with_sub_tools):
        mock_client = MagicMock()

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function.name = "sample_tool"
        mock_tool_call.function.arguments = '{"x": 2, "y": 3}'

        mock_message1 = MagicMock()
        mock_message1.tool_calls = [mock_tool_call]
        mock_message1.content = None
        mock_message1.model_dump = MagicMock(
            return_value={
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "sample_tool",
                            "arguments": '{"x": 2, "y": 3}',
                        },
                    }
                ],
            }
        )

        mock_response1 = MagicMock()
        mock_response1.choices = [MagicMock(message=mock_message1)]
        mock_response1.usage = MagicMock(prompt_tokens=50, completion_tokens=30)

        mock_message2 = MagicMock()
        mock_message2.tool_calls = None
        mock_message2.content = "The result is 5"

        mock_response2 = MagicMock()
        mock_response2.choices = [MagicMock(message=mock_message2)]
        mock_response2.usage = MagicMock(prompt_tokens=100, completion_tokens=20)
        mock_response2.model_dump = MagicMock(
            return_value={
                "choices": [{"message": {"content": "The result is 5"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            }
        )

        mock_client.chat.completions.create = AsyncMock(
            side_effect=[mock_response1, mock_response2]
        )

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
# 11. Sub-LLM Trajectory Steps
# =============================================================================


class TestSubLLMTrajectorySteps:
    @pytest.mark.asyncio
    async def test_include_sub_llm_in_trajectory_default(self, rlm_env):
        assert rlm_env.include_sub_llm_in_trajectory is True


# =============================================================================
# 12. Tunnel Utils (kept for coverage)
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
