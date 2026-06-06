from mcp.server.fastmcp import FastMCP

from wiki_search_v1.taskset import (
    WikiIndex,
    load_wiki,
    read_section,
    search_pages,
    view_sections,
)

mcp = FastMCP("wiki-search")
wiki_index: WikiIndex | None = None


def wiki() -> WikiIndex:
    global wiki_index
    if wiki_index is None:
        wiki_index = load_wiki()
    return wiki_index


@mcp.tool()
async def search_pages_tool(query: str) -> list[dict[str, str]]:
    return await search_pages(query, wiki())


@mcp.tool()
async def view_sections_tool(page_id: str) -> list[dict[str, str]]:
    return await view_sections(page_id, wiki())


@mcp.tool()
async def read_section_tool(section_id: str) -> str:
    return await read_section(section_id, wiki())


if __name__ == "__main__":
    mcp.run(transport="stdio")
