# tool examples, not actively maintained or intended for extension
# includes: ask, search, calculator, python

import ast
import os
import subprocess
import textwrap


def _get_url_markdown(url: str) -> str:
    """Get contents of URL as nicely formatted markdown."""
    import requests

    try:
        from markdownify import markdownify as md  # type: ignore

        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return md(response.text)
    except Exception as e:
        return f"Error: {str(e)}"


def ask(question: str, url: str) -> str:
    """Ask a question about a web page returned from search results.

    Args:
        question: The question to be answered (by an LLM who will be given the web page contents)
        url: The URL of the web page to query

    Returns:
        A LLM-generated answer to the question based on the web page contents.

    Examples:
        {"question": "What is the capital of France?", "url": "https://en.wikipedia.org/wiki/France"} -> "The capital of France is Paris."
        {"question": "How many people live in the United States?", "url": "https://en.wikipedia.org/wiki/United_States"} -> "The population of the United States is approximately 340 million people."
    """
    BASE_URL = "https://api.deepinfra.com/v1/openai"
    API_KEY = os.getenv("DEEPINFRA_API_KEY")
    MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

    contents = _get_url_markdown(url)[:50000]

    if contents.startswith("Error:"):
        return "Error: Failed to fetch URL contents."

    from openai import OpenAI

    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    prompt = f"""Answer the following question based on the provided web page contents:

    Question: {question}

    Page: {url}

    Page contents:
    {contents}
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
        )
        return response.choices[0].message.content or "Error: No response from model."
    except Exception as e:
        return f"Error: {str(e)}"


def calculator(expression: str) -> str:
    """Evaluates a single line of Python math expression. No imports or variables allowed.

    Args:
        expression (str): A mathematical expression using only numbers and basic operators (+,-,*,/,**,())

    Returns:
        The result of the calculation or an error message

    Examples:
        "2 + 2" -> "4"
        "3 * (17 + 4)" -> "63"
        "100 / 5" -> "20.0"
    """
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: Invalid characters in expression"

    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


def _jupyterize(src: str) -> str:
    src = textwrap.dedent(src)
    tree = ast.parse(src, mode="exec")

    if tree.body and isinstance(tree.body[-1], ast.Expr):
        # Extract the last expression
        last = tree.body.pop()
        body_code = ast.unparse(ast.Module(tree.body, []))
        expr_code = ast.unparse(last.value)  # type: ignore
        # Display the last expression value like Jupyter does
        return f"{body_code}\n_ = {expr_code}\nif _ is not None: print(_)"
    else:
        return src


def python(code: str) -> str:
    """Evaluates Python code like a Jupyter notebook cell, returning output and/or the last expression value.

    Args:
        code (str): A block of Python code

    Returns:
        The output (stdout + last expression if not None) or error message

    Examples:
        {"code": "import numpy as np\nnp.array([1, 2, 3]) + np.array([4, 5, 6])"} -> "[5 7 9]"
        {"code": "x = 5\ny = 10\nx + y"} -> "15"
        {"code": "a, b = 3, 4\na, b"} -> "(3, 4)"
    """

    try:
        # Run the code block in subprocess with 10-second timeout
        result = subprocess.run(
            ["python", "-c", _jupyterize(code)],
            timeout=10,
            text=True,
            capture_output=True,
        )

        output = result.stdout.strip()[:1000]
        error = result.stderr.strip()[:1000]
        if error:
            return error
        return output
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out after 10 seconds"
