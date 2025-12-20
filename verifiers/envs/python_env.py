import base64
import json
import sys
import textwrap
import time
from typing import Any

if sys.version_info < (3, 12):
    from typing_extensions import TypedDict
else:
    from typing import TypedDict

import verifiers as vf
from verifiers.envs.sandbox_env import SandboxEnv, SandboxState
from verifiers.utils.decorators import cleanup


class PythonWorkerState(TypedDict):
    ready: bool
    execution_count: int


class PythonWorkerNotReadyError(vf.SandboxError): ...


class PythonWorkerRequestError(vf.SandboxError): ...


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

        {pip_install_command}

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
        
        while true; do
          if [ -f "{ready_flag}" ]; then
            exit 0
          fi
          sleep 0.05
        done
        '
        """
    )

    def __init__(
        self,
        pip_install_packages: str = "numpy sympy scipy",
        max_startup_wait_seconds: int = 30,
        **kwargs: Any,
    ) -> None:
        pip_install_command = (
            f"pip install -q {pip_install_packages}"
            if pip_install_packages.strip()
            else ""
        )
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
            pip_install_command=pip_install_command,
        )
        self.max_startup_wait_seconds = max_startup_wait_seconds
        super().__init__(
            sandbox_name="python-env",
            docker_image="python:3.11-slim",
            start_command=start_command,
            **kwargs,
        )
        self.add_tool(
            self.python, args_to_skip=["sandbox_id", "sandbox_state", "python_state"]
        )
        self.remove_tool(self.bash)  # omit from agent tool list

    async def setup_state(self, state: vf.State, **kwargs: Any) -> vf.State:
        state = await super().setup_state(state, **kwargs)
        state["python_state"] = {
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
        assert isinstance(tool_args, dict), (
            f"Expected tool_args to be a dict, got {type(tool_args)}: {tool_args}"
        )
        updated_args = dict(tool_args)
        if tool_name == "python":
            updated_args["sandbox_id"] = state["sandbox_id"]
            updated_args["sandbox_state"] = state["sandbox_state"]
            updated_args["python_state"] = state["python_state"]
        return updated_args

    async def python(
        self,
        code: str,
        sandbox_id: str,
        sandbox_state: SandboxState,
        python_state: PythonWorkerState,
    ) -> str:
        """Execute `code` inside persistent Python REPL."""
        if not python_state["ready"]:
            await self._wait_for_worker_ready(sandbox_state, sandbox_id)
            python_state["ready"] = True
        sandbox_response = await self._send_worker_request(
            sandbox_id, sandbox_state, {"code": code}
        )
        return self._format_response(python_state, sandbox_response)

    @cleanup
    async def cleanup_python_state(self, state: vf.State):
        state.pop("python_state", None)

    async def _wait_for_worker_ready(
        self, sandbox_state: SandboxState, sandbox_id: str
    ) -> None:
        s = time.time()
        try:
            await self._wait_for_sandbox_ready(sandbox_state, sandbox_id)
            wait_script = self._READY_WAIT_SCRIPT.format(ready_flag=self._READY_FLAG)
            self.logger.debug(f"Setting up Python worker in sandbox {sandbox_id}")
            result = await self.sandbox_client.execute_command(
                sandbox_id, wait_script, timeout=self.max_startup_wait_seconds
            )
            if result.exit_code != 0:
                raise RuntimeError(result.stderr)
            self.logger.debug(
                f"Waited {time.time() - s:.1f}s for Python worker to be ready"
            )
        except Exception as e:
            raise PythonWorkerNotReadyError(e)

    async def _send_worker_request(
        self, sandbox_id: str, sandbox_state, payload: dict[str, Any]
    ) -> dict[str, Any]:
        try:
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
            raw_response = await self.bash(command, sandbox_id, sandbox_state)
            if not raw_response:
                raise RuntimeError("Python worker returned no output")
            response = json.loads(raw_response)
        except Exception as e:
            raise PythonWorkerRequestError(e)

        return response

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
