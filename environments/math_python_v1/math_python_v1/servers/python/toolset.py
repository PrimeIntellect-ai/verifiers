import ast
import contextlib
import io
import traceback

import verifiers.v1 as vf

from .config import PythonToolsetConfig


def execute_python(code: str, history: list[str]) -> str:
    namespace: dict[str, object] = {}
    for snippet in history:
        exec(compile(snippet, "<history>", "exec"), namespace, namespace)
    tree = ast.parse(code, "<tool>", "exec")
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            prefix = ast.Module(body=tree.body[:-1], type_ignores=[])
            exec(compile(prefix, "<tool>", "exec"), namespace, namespace)
            expression = ast.Expression(tree.body[-1].value)
            result = eval(compile(expression, "<tool>", "eval"), namespace, namespace)
            if result is not None:
                print(repr(result))
        else:
            exec(compile(tree, "<tool>", "exec"), namespace, namespace)
    history.append(code)
    return stdout.getvalue().strip() or "(no output)"


class PythonToolset(vf.Toolset[PythonToolsetConfig]):
    @vf.resource
    def history(self) -> list[str]:
        return []

    @vf.tool(
        args={"history": "resources.history"},
        sets={"python_history": "state.extras.python_history"},
    )
    def python(self, code: str, history: list[str]) -> dict:
        try:
            content = execute_python(code, history)
        except BaseException:
            content = traceback.format_exc()
        return {
            "content": content,
            "python_history": list(history),
        }
