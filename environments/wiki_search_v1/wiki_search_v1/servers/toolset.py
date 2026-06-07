import verifiers.v1 as vf

from wiki_search_v1.taskset import (
    WikiIndex,
    load_wiki,
    read_section,
    search_pages,
    view_sections,
)

wiki_index: WikiIndex | None = None


def wiki() -> WikiIndex:
    global wiki_index
    if wiki_index is None:
        wiki_index = load_wiki()
    return wiki_index


class WikiToolset(vf.Toolset):
    @vf.tool
    async def search_pages_tool(self, query: str) -> list[dict[str, str]]:
        return await search_pages(query, wiki())

    @vf.tool
    async def view_sections_tool(self, page_id: str) -> list[dict[str, str]]:
        return await view_sections(page_id, wiki())

    @vf.tool
    async def read_section_tool(self, section_id: str) -> str:
        return await read_section(section_id, wiki())
