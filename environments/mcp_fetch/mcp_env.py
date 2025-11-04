"""Fetch MCP environment built on the shared MCPEnv wrapper."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import shlex
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

from datasets import Dataset

import verifiers as vf
from verifiers.envs.mcp_env import MCPEnv
from verifiers.envs.mcp import MCPServerConfig

TASKS_PATH = Path(__file__).resolve().parent / "tasks" / "qa.jsonl"
DEFAULT_FIXTURE_PORT = 31415
DEFAULT_ONLINE_HOSTS: list[str] = ["example.com", "httpbin.org"]

SYSTEM_PROMPT = """You can call a single MCP tool named `fetch`.

Rules:
1. Always call `fetch` to read the requested URL(s); never guess.
2. Use the HTTP method, headers, query params, and byte limits the task specifies.
3. After finishing tool calls reply with `ANSWER: <value>` on its own line.
4. Keep answers concise and deterministic—return raw numbers/strings when possible.
"""


def _normalize_hosts(hosts: Optional[Iterable[str]]) -> list[str]:
    if hosts is None:
        return []
    normalized: list[str] = []
    for host in hosts:
        entry = host.strip()
        if entry and entry not in normalized:
            normalized.append(entry)
    return normalized


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _load_tasks(path: Path = TASKS_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Task file not found: {path}")
    tasks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            tasks.append(record)
    if len(tasks) < 20:
        raise ValueError("Fetch environment requires at least 20 tasks")
    return tasks


def _build_dataset(task_rows: list[dict[str, Any]]) -> Dataset:
    records = []
    for row in task_rows:
        records.append(
            {
                "question": row["question"],
                "answer": str(row.get("expected", "")),
                "task_id": row["id"],
                "verifier": json.dumps(row.get("verifier", {})),
                "meta": json.dumps(row.get("meta", {})),
            }
        )
    return Dataset.from_list(records)


def _normalize_answer(text: str) -> str:
    return " ".join(text.strip().split()).lower()


def _build_accuracy_rubric(parser: vf.Parser) -> vf.Rubric:
    async def accuracy(
        completion, answer: str, parser: vf.Parser, **_: Any
    ) -> float:
        guess = parser.parse_answer(completion) or ""
        return 1.0 if _normalize_answer(guess) == _normalize_answer(answer) else 0.0

    return vf.Rubric(funcs=[accuracy], weights=[1.0], parser=parser)


def _command_list(cmd: Optional[Sequence[str] | str], default: list[str]) -> list[str]:
    if cmd is None:
        return list(default)
    if isinstance(cmd, str):
        return shlex.split(cmd)
    parts = list(cmd)
    if not parts:
        raise ValueError("Command override cannot be empty")
    return parts


class FetchEnv(MCPEnv):
    """Concrete MCPEnv wiring for the deterministic fetch MCP server."""

    name = "mcp_fetch"
    version = "0.3.0"

    def __init__(
        self,
        *,
        server_cmd: Optional[Sequence[str] | str] = None,
        server_env: Optional[Dict[str, str]] = None,
        allow_online: bool = False,
        allow_any_host: bool = False,
        allowed_hosts: Optional[Iterable[str]] = None,
        auto_start_fixtures: bool = True,
        fixture_port: int = DEFAULT_FIXTURE_PORT,
        fixture_cmd: Optional[Sequence[str] | str] = None,
        **kwargs: Any,
    ) -> None:
        self.allow_online = allow_online
        self.allow_any_host = allow_any_host
        self.allowed_hosts = _normalize_hosts(allowed_hosts)
        self.fixture_port = fixture_port
        self._fixture_cmd = _command_list(
            fixture_cmd,
            [
                sys.executable,
                "-m",
                "environments.mcp_fetch.utils.mini_httpd",
                "--port",
                str(fixture_port),
            ],
        )
        self._fixture_proc: subprocess.Popen[str] | None = None
        self._owns_fixture = False

        command_parts = _command_list(
            server_cmd,
            [
                sys.executable,
                "-m",
                "environments.mcp_fetch.tools.fetch_mcp_server",
                "--run-server",
            ],
        )

        host_allowlist = self.allowed_hosts
        offline_hosts = _default_offline_hosts(self.fixture_port)
        if not host_allowlist:
            if allow_any_host:
                host_allowlist = []
            elif allow_online:
                host_allowlist = offline_hosts + DEFAULT_ONLINE_HOSTS
            else:
                host_allowlist = offline_hosts

        env = dict(server_env or {})
        self.host_allowlist = host_allowlist

        if host_allowlist:
            env["MCP_FETCH_ALLOWED_HOSTS"] = ",".join(host_allowlist)
        if allow_any_host:
            env["MCP_FETCH_ALLOW_ANY_HOST"] = "1"

        config = MCPServerConfig(
            name="fetch",
            command=command_parts[0],
            args=command_parts[1:],
            env=env or None,
            description="Deterministic Fetch MCP server",
        )

        super().__init__(mcp_servers=[config], **kwargs)

        if auto_start_fixtures and not allow_online and not allow_any_host:
            self._ensure_fixture_server()

    def _ensure_fixture_server(self) -> None:
        if _port_in_use(self.fixture_port):
            self.logger.info(
                "Reusing existing fixture server on port %s", self.fixture_port
            )
            self._owns_fixture = False
            return

        try:
            self._fixture_proc = subprocess.Popen(
                self._fixture_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as exc:  # pragma: no cover - spawn errors are rare
            raise RuntimeError(f"Failed to launch fixture server: {exc}") from exc

        self._owns_fixture = True
        self.logger.info(
            "Started fixture server with pid %s on port %s",
            self._fixture_proc.pid if self._fixture_proc else "unknown",
            self.fixture_port,
        )

    async def cleanup(self) -> None:
        await super().cleanup()
        self._stop_fixture_server()

    def _stop_fixture_server(self) -> None:
        if not self._owns_fixture or not self._fixture_proc:
            return
        proc = self._fixture_proc
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()
        self._fixture_proc = None
        self._owns_fixture = False


def load_environment(
    *,
    allow_online: bool = False,
    allow_any_host: bool = False,
    allowed_hosts: Optional[Iterable[str]] = None,
    server_cmd: Optional[Sequence[str] | str] = None,
    server_env: Optional[Dict[str, str]] = None,
    task_path: Optional[str | Path] = None,
    dataset: Dataset | None = None,
    rubric: vf.Rubric | None = None,
    system_prompt: str | None = None,
    parser: vf.Parser | None = None,
    **kwargs: Any,
) -> FetchEnv:
    """Factory hook used by verifiers to instantiate the environment."""

    tasks = _load_tasks(Path(task_path) if task_path else TASKS_PATH)
    dataset = dataset or _build_dataset(tasks)

    parser = parser or vf.Parser(
        extract_fn=lambda text: _extract_answer(text),
    )
    rubric = rubric or _build_accuracy_rubric(parser)
    system_prompt = system_prompt or SYSTEM_PROMPT

    return FetchEnv(
        allow_online=allow_online,
        allow_any_host=allow_any_host,
        allowed_hosts=allowed_hosts,
        server_cmd=server_cmd,
        server_env=server_env,
        dataset=dataset,
        rubric=rubric,
        system_prompt=system_prompt,
        parser=parser,
        **kwargs,
    )


def _extract_answer(text: str) -> str:
    marker = "answer:"
    lowered = text.lower()
    if marker not in lowered:
        return text.strip()
    idx = lowered.rfind(marker)
    return text[idx + len(marker) :].strip()
def _default_offline_hosts(port: int) -> list[str]:
    return [f"127.0.0.1:{port}"]
