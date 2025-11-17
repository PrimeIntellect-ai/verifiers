import base64
import json
import textwrap
from typing import Any

from typing_extensions import TypedDict

import verifiers as vf
from verifiers.envs.sandbox_env import SandboxEnv
from verifiers.utils.decorators import cleanup


class PythonWorkerState(TypedDict):
    ready: bool
    execution_count: int


class PythonEnv(SandboxEnv):
    """Sandbox-backed environment exposing a persistent Python REPL."""

    _WORKER_PATH = "/tmp/python_worker.py"
    _COMMAND_FIFO = "/tmp/python_env_cmd"
    _RESPONSE_FIFO = "/tmp/python_env_res"
    _READY_FLAG = "/tmp/python_env_ready"

    _WORKER_SCRIPT = textwrap.dedent(
        """
        import ast
        import contextlib
        import io
        import json
        import os
        from pathlib import Path
        import traceback

        WORKER_PATH = os.path.abspath(__file__)
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

    _START_COMMAND_TEMPLATE = textwrap.dedent(
        """
        bash -lc '
        set -euo pipefail

        command_fifo="{command_fifo}"
        response_fifo="{response_fifo}"
        ready_flag="{ready_flag}"
        worker_path="{worker_path}"

        rm -f "$command_fifo" "$response_fifo" "$ready_flag"

        pip install -q numpy sympy scipy

        python - <<'PY'
import base64
from pathlib import Path

Path("{worker_path}").write_bytes(base64.b64decode("{worker_b64}"))
PY

        python -u "$worker_path" &
        tail -f /dev/null
        '
        """
    )

    _READY_WAIT_SCRIPT = textwrap.dedent(
        """
        bash -lc '
        for i in $(seq 1 200); do
          if [ -f "{ready_flag}" ]; then
            exit 0
          fi
          sleep 0.05
        done
        echo "python worker failed to start" >&2
        exit 1
        '
        """
    )

    def __init__(self, **kwargs: Any) -> None:
        start_command = self._START_COMMAND_TEMPLATE.format(
            command_fifo=self._COMMAND_FIFO,
            response_fifo=self._RESPONSE_FIFO,
            ready_flag=self._READY_FLAG,
            worker_path=self._WORKER_PATH,
            worker_b64=base64.b64encode(
                self._WORKER_SCRIPT.format(
                    command_fifo=self._COMMAND_FIFO,
                    response_fifo=self._RESPONSE_FIFO,
                    ready_flag=self._READY_FLAG,
                ).encode("utf-8")
            ).decode("utf-8"),
        )
        super().__init__(
            sandbox_name="python-env",
            docker_image="python:3.11-slim",
            start_command=start_command,
            **kwargs,
        )
        self.add_tool(self.python, args_to_skip=["sandbox_id", "python_state"])
        self.remove_tool(self.bash)  # omit from agent tool list

    async def setup_state(self, state: vf.State, **kwargs: Any) -> vf.State:
        state = await super().setup_state(state, **kwargs)
        state["python_env"] = {
            "ready": False,
            "execution_count": 0,
        }
        return state

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> dict[str, Any]:
        updated_args = dict(tool_args)
        if tool_name != "python":
            return updated_args
        sandbox_id = state["sandbox_id"]
        python_state = state["python_env"]
        updated_args["python_state"] = python_state
        updated_args["sandbox_id"] = sandbox_id
        return updated_args

    async def python(
        self, code: str, sandbox_id: str, python_state: PythonWorkerState
    ) -> str:
        """Execute `code` inside persistent Python REPL."""
        if not python_state["ready"]:
            await self._wait_for_worker_ready(sandbox_id)
            python_state["ready"] = True
        sandbox_response = await self._send_worker_request(sandbox_id, {"code": code})
        return self._format_response(python_state, sandbox_response)

    @cleanup
    async def cleanup_python_env(self, state: vf.State):
        state.pop("python_env", None)

    async def _wait_for_worker_ready(self, sandbox_id: str) -> None:
        wait_script = self._READY_WAIT_SCRIPT.format(ready_flag=self._READY_FLAG)
        await self.bash(wait_script, sandbox_id=sandbox_id)

    async def _send_worker_request(
        self, sandbox_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        payload_json = json.dumps(payload)
        payload_b64 = base64.b64encode(payload_json.encode("utf-8")).decode("utf-8")
        command = textwrap.dedent(
            f"""
            python - <<'PY'
import base64
import json
import sys

data = base64.b64decode('{payload_b64}').decode('utf-8')
with open('{self._COMMAND_FIFO}', 'w', encoding='utf-8') as command_file:
    command_file.write(data)
with open('{self._RESPONSE_FIFO}', 'r', encoding='utf-8') as response_file:
    sys.stdout.write(response_file.read())
PY
            """
        )
        raw_response = await self.bash(command, sandbox_id=sandbox_id)
        if not raw_response:
            raise RuntimeError("Python worker returned no output")
        return json.loads(raw_response)

    def _format_response(
        self, python_state: PythonWorkerState, sandbox_response: dict[str, Any]
    ) -> str:
        execution_count = sandbox_response.get("execution_count")
        if execution_count is None:
            execution_count = python_state.get("execution_count", 0) + 1
        python_state["execution_count"] = execution_count

        parts: list[str] = []
        stdout = (sandbox_response.get("stdout") or "").rstrip()
        if stdout:
            parts.append(stdout)

        stderr = (sandbox_response.get("stderr") or "").rstrip()
        if stderr:
            parts.append(f"stderr:\n{stderr}")

        status = sandbox_response.get("status")
        result_text = sandbox_response.get("result")
        if status == "error" and result_text:
            parts.append(result_text.rstrip())
        elif status == "ok" and result_text is not None:
            parts.append(f"Out[{execution_count}]: {result_text}")

        if not parts:
            parts.append("(no output)")

        return "\n".join(parts)
