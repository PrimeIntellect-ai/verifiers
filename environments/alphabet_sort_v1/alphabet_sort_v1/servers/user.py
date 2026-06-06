from mcp.server.fastmcp import FastMCP

mcp = FastMCP("alphabet-sort-user")


@mcp.tool()
def respond(task: dict, state: dict, transcript: list[dict]) -> dict:
    _ = state
    info = task.get("info") or {}
    follow_ups = info.get("follow_ups") or []
    assistant_count = 0
    for turn in transcript:
        completion = turn.get("completion") or []
        assistant_count += sum(
            1 for message in completion if message.get("role") == "assistant"
        )
    if assistant_count <= 0 or assistant_count > len(follow_ups):
        return {"messages": []}
    return {
        "messages": [{"role": "user", "content": str(follow_ups[assistant_count - 1])}],
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
