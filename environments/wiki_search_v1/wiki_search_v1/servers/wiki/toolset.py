import verifiers.v1 as vf

from wiki_search_v1.taskset import (
    WikiIndex,
    load_wiki,
    read_section,
    search_pages,
    view_sections,
)

from .config import WikiToolsetConfig


class WikiToolset(vf.Toolset[WikiToolsetConfig]):
    @vf.resource
    def wiki(self) -> WikiIndex:
        return load_wiki(self.config)

    @vf.tool(args={"wiki": "resources.wiki"})
    async def search_pages_tool(
        self, query: str, wiki: WikiIndex
    ) -> list[dict[str, str]]:
        return await search_pages(query, wiki)

    @vf.tool(args={"wiki": "resources.wiki"})
    async def view_sections_tool(
        self, page_id: str, wiki: WikiIndex
    ) -> list[dict[str, str]]:
        return await view_sections(page_id, wiki)

    @vf.tool(args={"wiki": "resources.wiki"})
    async def read_section_tool(self, section_id: str, wiki: WikiIndex) -> str:
        return await read_section(section_id, wiki)
