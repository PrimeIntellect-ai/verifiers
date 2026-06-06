import ast
import contextlib
import io
import traceback

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("math-python")
HISTORY: list[str] = []


def execute_python(code: str) -> str:
    namespace: dict[str, object] = {}
    for snippet in HISTORY:
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
    HISTORY.append(code)
    return stdout.getvalue().strip() or "(no output)"


@mcp.tool()
def python(code: str) -> dict:
    try:
        content = execute_python(code)
    except BaseException:
        content = traceback.format_exc()
    return {
        "content": content,
        "scratch": {"python_history": list(HISTORY)},
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
