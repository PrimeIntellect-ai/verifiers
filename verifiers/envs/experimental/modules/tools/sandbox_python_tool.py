from __future__ import annotations

import base64
import json
import textwrap
import time
from typing import cast

from verifiers.errors import SandboxError
from verifiers.types import State, Tool

from verifiers.envs.experimental.resources import Resources
from verifiers.envs.experimental.task import Task
from verifiers.envs.experimental.modules.tools.sandbox_tool import SandboxTool


class SandboxPythonTool(SandboxTool):
    """Persistent Python REPL tool backed by a sandbox."""

    _WORKER_PATH = "/tmp/vf_python_tool_worker.py"
    _WORKER_PID_FILE = "/tmp/vf_python_tool_worker.pid"
    _COMMAND_FIFO = "/tmp/vf_python_tool_cmd"
    _RESPONSE_FIFO = "/tmp/vf_python_tool_res"
    _READY_FLAG = "/tmp/vf_python_tool_ready"

    _WORKER_SCRIPT = textwrap.dedent(
        """
        import ast
        import contextlib
        import io
        import json
        import os
        from pathlib import Path
        import traceback

        COMMAND_FIFO = "{command_fifo}"
        RESPONSE_FIFO = "{response_fifo}"
        READY_FLAG = "{ready_flag}"

        def ensure_fifo(path: str) -> None:
            if os.path.exists(path):
                os.remove(path)
            os.mkfifo(path)

        for fifo_path in (COMMAND_FIFO, RESPONSE_FIFO):
            ensure_fifo(fifo_path)

        Path(READY_FLAG).write_text("ready", encoding="utf-8")

        namespace: dict[str, object] = {{"__name__": "__main__"}}
        execution_count = 0

        while True:
            with open(COMMAND_FIFO, "r", encoding="utf-8") as command_file:
                payload = command_file.read()
            if not payload:
                continue
            request = json.loads(payload)
            if request.get("shutdown"):
                break
            code = request.get("code", "")
            execution_count += 1
            result = {{
                "status": "ok",
                "stdout": "",
                "stderr": "",
                "result": None,
                "execution_count": execution_count,
            }}
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
                    stderr_buffer
                ):
                    module_ast = ast.parse(code, mode="exec")
                    body = list(module_ast.body)
                    trailing_expr = None
                    if body and isinstance(body[-1], ast.Expr):
                        trailing_expr = body.pop()
                    if body:
                        exec_module = ast.Module(body=body, type_ignores=[])
                        exec(compile(exec_module, "<cell>", "exec"), namespace, namespace)
                    if trailing_expr is not None:
                        value = eval(
                            compile(ast.Expression(trailing_expr.value), "<cell>", "eval"),
                            namespace,
                            namespace,
                        )
                        if value is not None:
                            result["result"] = repr(value)
            except Exception:
                result["status"] = "error"
                result["result"] = traceback.format_exc()
            result["stdout"] = stdout_buffer.getvalue()
            result["stderr"] = stderr_buffer.getvalue()
            with open(RESPONSE_FIFO, "w", encoding="utf-8") as response_file:
                response_file.write(json.dumps(result))
        """
    )

    def __init__(
        self,
        name: str = "python",
        pip_install_packages: str = "numpy sympy scipy",
        max_startup_wait_seconds: int = 60,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.pip_install_packages = pip_install_packages
        self.max_startup_wait_seconds = max_startup_wait_seconds

    def schema(self) -> Tool:
        return Tool(
            name=self.name,
            description="Execute code inside a persistent Python REPL.",
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute.",
                    }
                },
                "required": ["code"],
            },
        )

    async def __call__(
        self,
        code: str,
        task: Task,
        state: State,
        resources: Resources,
    ) -> str:
        await self.ensure_worker(task, state, resources)
        exit_code, output = await self.execute_command(
            self.worker_request_command({"code": code}),
            task,
            state,
            resources,
            timeout=self.command_timeout,
        )
        if exit_code == -1:
            return output
        if exit_code != 0:
            raise SandboxError(output)
        response = json.loads(output)
        return self.format_response(state, response)

    async def ensure_worker(
        self, task: Task, state: State, resources: Resources
    ) -> None:
        sandbox_state = self.sandbox_state(state)
        python_state = cast(
            dict[str, object],
            sandbox_state.setdefault(
                "python",
                {"ready": False, "execution_count": 0, "ready_wait_time": 0.0},
            ),
        )
        state.setdefault("python_state", python_state)
        if python_state["ready"]:
            return
        start = time.time()
        exit_code, output = await self.execute_command(
            self.start_worker_command(),
            task,
            state,
            resources,
            timeout=self.max_startup_wait_seconds,
        )
        if exit_code != 0:
            raise SandboxError(output)
        python_state["ready"] = True
        python_state["ready_wait_time"] = time.time() - start

    def start_worker_command(self) -> str:
        worker_b64 = base64.b64encode(
            self._WORKER_SCRIPT.format(
                command_fifo=self._COMMAND_FIFO,
                response_fifo=self._RESPONSE_FIFO,
                ready_flag=self._READY_FLAG,
            ).encode("utf-8")
        ).decode("utf-8")
        pip_install_command = (
            f"pip install -q {self.pip_install_packages}"
            if self.pip_install_packages.strip()
            else ""
        )
        poll_count = max(1, int(self.max_startup_wait_seconds / 0.05))
        return textwrap.dedent(
            f"""
            bash -lc '
            set -euo pipefail
            rm -f "{self._COMMAND_FIFO}" "{self._RESPONSE_FIFO}" "{self._READY_FLAG}"
            {pip_install_command}
            python - <<'PY'
import base64
from pathlib import Path
Path("{self._WORKER_PATH}").write_bytes(base64.b64decode("{worker_b64}"))
PY
            nohup python -u "{self._WORKER_PATH}" > /tmp/vf_python_tool_worker.log 2>&1 &
            echo $! > "{self._WORKER_PID_FILE}"
            for _ in $(seq 1 {poll_count}); do
                if [ -f "{self._READY_FLAG}" ]; then
                    exit 0
                fi
                sleep 0.05
            done
            cat /tmp/vf_python_tool_worker.log >&2 || true
            exit 1
            '
            """
        )

    def worker_request_command(self, payload: dict[str, object]) -> str:
        payload_b64 = base64.b64encode(json.dumps(payload).encode("utf-8")).decode(
            "utf-8"
        )
        return textwrap.dedent(
            f"""
            bash -lc '
            [ -f "{self._WORKER_PID_FILE}" ] && [ -d "/proc/$(cat {self._WORKER_PID_FILE})" ] || {{ echo "WORKER_DEAD"; exit 1; }}
            python - <<'PY'
import base64
import sys
data = base64.b64decode("{payload_b64}").decode("utf-8")
with open("{self._COMMAND_FIFO}", "w", encoding="utf-8") as command_file:
    command_file.write(data)
with open("{self._RESPONSE_FIFO}", "r", encoding="utf-8") as response_file:
    sys.stdout.write(response_file.read())
PY
            '
            """
        )

    def format_response(self, state: State, response: dict[str, object]) -> str:
        python_state = state["python_state"]
        execution_count = response.get("execution_count")
        if execution_count is None:
            execution_count = int(python_state.get("execution_count", 0)) + 1
        python_state["execution_count"] = execution_count

        parts: list[str] = []
        stdout = str(response.get("stdout") or "").rstrip()
        if stdout:
            parts.append(stdout)
        stderr = str(response.get("stderr") or "").rstrip()
        if stderr:
            parts.append(f"stderr:\n{stderr}")
        result_text = response.get("result")
        if response.get("status") == "error" and result_text:
            parts.append(str(result_text).rstrip())
        elif response.get("status") == "ok" and result_text is not None:
            parts.append(f"Out[{execution_count}]: {result_text}")
        if not parts:
            parts.append("(no output)")
        return "\n".join(parts)
