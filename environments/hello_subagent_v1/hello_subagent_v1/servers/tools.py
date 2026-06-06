from mcp.server.fastmcp import FastMCP

mcp = FastMCP("hello-subagent")


@mcp.tool()
def ask_subagent(name: str, _vf: dict | None = None) -> dict:
    answer = f"hello {name}"
    if not isinstance(_vf, dict) or not isinstance(_vf["state"], dict):
        raise TypeError("ask_subagent requires harness context.")
    scratch = _vf["state"]["scratch"]
    existing = scratch.get("subagent_calls") if isinstance(scratch, dict) else []
    calls = list(existing) if isinstance(existing, list) else []
    calls.append({"name": name, "answer": answer})
    return {"content": answer, "scratch": {"subagent_calls": calls}}


if __name__ == "__main__":
    mcp.run(transport="stdio")
