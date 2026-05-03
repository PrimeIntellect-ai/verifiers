import json

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mcp-search-v1")

DOCUMENTS = {
    "v1_composition": {
        "title": "v1 Taskset Harness Composition",
        "content": (
            "verifiers.v1 composes reusable tasksets with reusable harnesses "
            "through vf.Env. Promoted config fields resolve tools, users, "
            "signals, cleanup, endpoints, and sandboxes."
        ),
    },
    "runtime": {
        "title": "Runtime Boundary",
        "content": (
            "Task and State stay serializable. Harness runtime owns live "
            "clients, tool handles, MCP sessions, and sandbox leases."
        ),
    },
}


@mcp.tool()
def search_documents(query: str) -> str:
    matches = []
    normalized = query.lower()
    for document_id, document in DOCUMENTS.items():
        text = f"{document['title']} {document['content']}".lower()
        if any(token in text for token in normalized.split()):
            matches.append({"document_id": document_id, "title": document["title"]})
    return json.dumps(
        matches
        or [
            {"document_id": key, "title": value["title"]}
            for key, value in DOCUMENTS.items()
        ]
    )


@mcp.tool()
def read_document(document_id: str) -> str:
    if document_id not in DOCUMENTS:
        raise ValueError(f"Unknown document_id: {document_id}")
    document = DOCUMENTS[document_id]
    return f"{document['title']}\n\n{document['content']}"


if __name__ == "__main__":
    mcp.run(transport="stdio")
