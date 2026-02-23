"""Sandbox backend tests for RLMEnv (mocked)."""

import ast
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from datasets import Dataset

from verifiers.envs.experimental import rlm_env as rlm_module
from verifiers.envs.experimental.rlm_env import (
    RLMEnv,
    RLMWorkerPaths,
)


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


def _seed_rollout_dirs(state: dict, tmp_path: Path) -> None:
    rollout_dir = tmp_path / "rlm_rollout"
    fs_root = rollout_dir / "rlm_fs"
    control_dir = rollout_dir / "rlm_control"
    fs_root.mkdir(parents=True, exist_ok=True)
    control_dir.mkdir(parents=True, exist_ok=True)
    state["rlm_rollout_dir"] = str(rollout_dir)
    state["rlm_fs_root"] = str(fs_root)
    state["rlm_control_dir"] = str(control_dir)
    state["rlm_paths"] = {}


class TestExecutorIsRLMExecutor:
    def test_default_executor_is_rlm_executor(self):
        dataset = make_dataset({})
        env = build_env(dataset)
        assert env._executor.__class__.__name__ == "RLMExecutor"


class TestWorkerScripts:
    def test_rendered_python_worker_is_valid(self, tmp_path: Path):
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
        script = rlm_module._render_worker_script(paths, repl_language="python")
        ast.parse(script)
        assert "FilesystemJail" not in script

    def test_rendered_bash_worker_is_valid(self, tmp_path: Path):
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
        assert "FilesystemJail" not in script
        assert "import pty" not in script.lower()


class TestTunnelRouting:
    @pytest.mark.asyncio
    async def test_uses_tunnel_when_no_interception_url(self, tmp_path: Path):
        dataset = make_dataset({})
        env = build_env(dataset, repl_language="bash")
        env._ensure_interception_server = AsyncMock()
        env._executor.prepare_filesystem = AsyncMock()
        env._executor.setup = AsyncMock()

        state = {"info": {}, "model": "m", "client": MagicMock()}
        _seed_rollout_dirs(state, tmp_path)
        env._executor.create_rollout_dirs = MagicMock(side_effect=lambda s=state: None)

        with patch("verifiers.envs.experimental.rlm_env.Tunnel") as TunnelMock:
            tunnel = TunnelMock.return_value
            tunnel.start = AsyncMock(return_value="https://tunnel.example")
            tunnel.stop = AsyncMock()

            result = await env.setup_state(state)

        tunnel.start.assert_awaited_once()
        assert result["interception_url"].startswith("https://tunnel.example")
        assert result["root_tool_url"].startswith("https://tunnel.example")

    @pytest.mark.asyncio
    async def test_skips_tunnel_when_interception_url_provided(self, tmp_path: Path):
        dataset = make_dataset({})
        env = build_env(
            dataset,
            interception_url="https://override.example/base",
            repl_language="bash",
        )
        env._ensure_interception_server = AsyncMock()
        env._executor.prepare_filesystem = AsyncMock()
        env._executor.setup = AsyncMock()

        state = {"info": {}, "model": "m", "client": MagicMock()}
        _seed_rollout_dirs(state, tmp_path)
        env._executor.create_rollout_dirs = MagicMock(side_effect=lambda s=state: None)

        with patch("verifiers.envs.experimental.rlm_env.Tunnel") as TunnelMock:
            result = await env.setup_state(state)

        TunnelMock.assert_not_called()
        assert result["interception_url"].startswith("https://override.example")
        assert result["root_tool_url"].startswith("https://override.example")


class TestCleanupSemantics:
    @pytest.mark.asyncio
    async def test_cleanup_calls_executor(self, tmp_path: Path):
        dataset = make_dataset({})
        env = build_env(
            dataset,
            interception_url="https://override.example/base",
        )
        env._ensure_interception_server = AsyncMock()
        env._executor.prepare_filesystem = AsyncMock()
        env._executor.setup = AsyncMock()
        env._executor.cleanup = AsyncMock()

        state = {
            "info": {},
            "model": "m",
            "client": MagicMock(),
        }
        _seed_rollout_dirs(state, tmp_path)
        env._executor.create_rollout_dirs = MagicMock(side_effect=lambda s=state: None)

        result = await env.setup_state(state)
        await env.cleanup_rlm_state(result)

        env._executor.cleanup.assert_awaited_once()


class TestFilesystemProvisioning:
    @pytest.mark.asyncio
    async def test_prepare_filesystem_uploads_and_sets_paths(self, tmp_path: Path):
        dataset = make_dataset({})
        env = build_env(dataset, repl_language="bash")
        state = {
            "rollout_id": "rlm_test",
            "model": "m",
            "client": MagicMock(),
        }

        env._executor.create_rollout_dirs(state)
        fs_root = Path(state["rlm_fs_root"])
        (fs_root / "data.txt").write_text("hi", encoding="utf-8")

        executor = env._executor
        executor.create_sandbox = AsyncMock(return_value="sbx_1")
        executor._execute_sandbox_command = AsyncMock()
        executor._upload_directory = AsyncMock()

        await executor.prepare_filesystem(state)

        executor.create_sandbox.assert_awaited_once()
        executor._upload_directory.assert_awaited_once()

        assert state["rlm_fs_staging_root"] == str(fs_root)
        assert state["rlm_fs_root_remote"].startswith("/tmp/rlm_rlm_test/rlm_fs")
        assert state["rlm_control_dir_remote"].startswith(
            "/tmp/rlm_rlm_test/rlm_control"
        )
        assert state["rlm_paths_remote"]["base_dir"].startswith(
            "/tmp/rlm_rlm_test/rlm_control"
        )

    @pytest.mark.asyncio
    async def test_write_sandbox_files_uploads_worker_and_context(self, tmp_path: Path):
        dataset = make_dataset({})
        env = build_env(dataset, repl_language="python")
        state = {
            "rollout_id": "rlm_test",
            "rlm_fs_root": "/tmp/rlm_rlm_test/rlm_fs",
            "model": "m",
            "client": MagicMock(),
            "interception_url": "http://example.invalid",
            "root_tool_url": "http://example.invalid",
        }

        executor = env._executor
        executor._sessions.clear()
        session = executor._get_or_create_session(state)
        session.sandbox_id = "sbx_1"
        session.sandbox_control_dir = "/tmp/rlm_rlm_test/rlm_control"
        session.sandbox_fs_root = "/tmp/rlm_rlm_test/rlm_fs"
        session.paths = rlm_module._build_worker_paths(session.sandbox_control_dir)

        executor.sandbox_client.upload_file = AsyncMock()

        await executor._write_sandbox_files(session, state)

        assert executor.sandbox_client.upload_file.await_count == 3
