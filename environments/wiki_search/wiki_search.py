import asyncio
import os
from typing import cast

import chromadb
from chromadb.api.types import Embeddable, EmbeddingFunction
from chromadb.utils import embedding_functions
from datasets import load_dataset
from openai import AsyncOpenAI

import verifiers as vf
from verifiers.rubrics.judge_rubric import JudgeRubric

CHROMA_DB_DIR = ".chroma_db"
_chroma_semaphore: asyncio.Semaphore | None = None


def _get_chroma_semaphore() -> asyncio.Semaphore:
    global _chroma_semaphore
    if _chroma_semaphore is None:
        _chroma_semaphore = asyncio.Semaphore(100)
    return _chroma_semaphore


def load_environment(
    max_turns: int = 10,
    judge_model: str = "gpt-4.1-mini",
    judge_base_url: str = "https://api.openai.com/v1",
    judge_api_key_var: str = "OPENAI_API_KEY",
    embed_model: str = "text-embedding-3-small",
    embed_base_url: str = "https://api.openai.com/v1",
    embed_api_key_var: str = "OPENAI_API_KEY",
    corpus_dataset: str = "willcb/rare-wiki-pages",
    corpus_split: str = "train",
    chroma_db_dir: str = CHROMA_DB_DIR,
) -> vf.Environment:
    # ensure Chroma server is running in client/server mode
    # ensure_chroma_server(chroma_db_dir)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        model_name=embed_model,
        api_base=embed_base_url,
        api_key=os.getenv(embed_api_key_var, "EMPTY"),
    )
    client = chromadb.PersistentClient(path=chroma_db_dir)
    collection = client.get_or_create_collection(
        name="wiki_titles",
        embedding_function=cast(EmbeddingFunction[Embeddable], openai_ef),
    )
    # load corpus into memory and build page_id -> row index
    corpus = load_dataset(corpus_dataset, split=corpus_split)
    page_id_to_title: dict[str, str] = {}
    page_id_to_content: dict[str, str] = {}
    for row in corpus:
        row = cast(dict, row)
        pid = row["id"]
        title = row["title"]
        content = row["content"]
        page_id_to_title[pid] = title
        page_id_to_content[pid] = content

    # initialize chroma collection
    def init_chroma() -> None:
        # upsert missing pages
        all_ids = list(page_id_to_title.keys())
        existing: set[str] = set()
        for i in range(0, len(all_ids), 500):
            batch = all_ids[i : i + 500]
            got = collection.get(ids=batch)
            existing.update(got.get("ids", []))
        missing = [pid for pid in all_ids if pid not in existing]
        if missing:
            documents = []
            metadatas = []
            for pid in missing:
                title = str(page_id_to_title[pid]).strip()
                if not title:
                    raise ValueError(f"Empty title for page_id {pid}")
                documents.append(title)
                metadatas.append({"title": title})
            bs = 100
            for i in range(0, len(missing), bs):
                collection.upsert(
                    ids=missing[i : i + bs],
                    documents=documents[i : i + bs],
                    metadatas=metadatas[i : i + bs],
                )

    init_chroma()

    # helper function to normalize section ids
    def normalize_id(text: str) -> str:
        """Normalize free text into an id: lowercased with spaces as underscores.

        Mirrors the section id normalization used elsewhere in this module.
        """
        return text.strip().lower().replace(" ", "_")

    # define tools
    async def search_pages(query: str) -> list[dict]:
        """Search for top 10 relevant articles using title embedding similarity.

        args:
            query (str): The query to search for.

        returns:
            list[dict]: A list of dicts with page_id and title.

        example:
            "basketball" -> [{"page_id": "basketball", "title": "Basketball"}, {"page_id": "basketball_rules", "title": "Basketball Rules"}, ...]
        """
        async with _get_chroma_semaphore():
            results = await asyncio.to_thread(
                collection.query, query_texts=[query], n_results=10
            )
        if not results:
            raise ValueError(f"No results found for query: {query}")
        if not results["metadatas"]:
            raise ValueError(f"No results metadata found for query: {query}")
        output = []
        for i in range(len(results["ids"][0])):
            output.append(
                {
                    "page_id": results["ids"][0][i],
                    "title": results["metadatas"][0][i]["title"],
                }
            )

        return output

    async def view_sections(page_id: str) -> list[dict]:
        """View the sections of a page.

        args:
            page_id (str): The ID of the page to view.

        returns:
            list[dict]: A list of dicts with section_id and section_name.

        example:
            "basketball" -> [{"section_id": "basketball:history", "section_name": "History"}, ...]
        """
        content = page_id_to_content[page_id]
        sections = []
        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("#"):
                section_name = line.lstrip("#").strip()
                section_id = f"{page_id}:{normalize_id(section_name)}"
                sections.append(
                    {
                        "section_id": section_id,
                        "section_name": section_name,
                        "start_line": i,
                    }
                )

        # if no sections found, return the whole page as one section
        if not sections:
            sections.append(
                {
                    "section_id": f"{page_id}:full",
                    "section_name": "Full Page",
                    "start_line": 0,
                }
            )

        return [
            {"section_id": s["section_id"], "section_name": s["section_name"]}
            for s in sections
        ]

    async def read_section(section_id: str) -> str:
        """Read a section of a page.

        args:
            section_id (str): The ID of the section to read.

        returns:
            str: The content of the section.

        example:
            "baseball:finnish_baseball" -> "Finnish baseball is a sport that is played in Finland..."
        """
        if ":" not in section_id:
            raise ValueError(
                "Invalid section_id format. Expected: page_id:section_name"
            )
        page_id, section_name_id = section_id.split(":", 1)

        # get Markdown content
        content = page_id_to_content[page_id]
        lines = content.split("\n")

        # special case for "full" section
        if section_name_id == "full":
            return content

        # find section
        section_start = None
        section_end = None

        for i, line in enumerate(lines):
            if line.startswith("#"):
                current_section = normalize_id(line.lstrip("#").strip())
                if current_section == section_name_id and section_start is None:
                    section_start = i
                elif section_start is not None and section_end is None:
                    section_end = i
                    break

        if section_start is not None:
            if section_end is None:
                section_end = len(lines)
            return "\n".join(lines[section_start:section_end])
        else:
            raise ValueError(f"Section not found: {section_id}")

    tools = [
        search_pages,
        view_sections,
        read_section,
    ]
    parser = vf.Parser()
    dataset = load_dataset("willcb/wiki-trivia-questions-v4", split="train")
    tool_rubric = vf.ToolRubric(tools=tools)

    JUDGE_PROMPT = """Given a ground truth answer \
    and a response, determine if the response is both correct and coherent.

    Question:
    ```
    {question}
    ```

    Ground truth answer:
    ```
    {answer}
    ```

    Response:
    ```
    {response}
    ```

    Respond either "yes" or "no" only.
    
    If a response contains incoherent text, respond with "no" even if the correct answer is also present.
    """
    judge_client = AsyncOpenAI(
        base_url=judge_base_url, api_key=os.getenv(judge_api_key_var)
    )
    judge_rubric = JudgeRubric(
        judge_client=judge_client,
        judge_model=judge_model,
        parser=parser,
        judge_prompt=JUDGE_PROMPT,
    )

    async def judge_reward_func(judge, prompt, completion, answer, state) -> float:
        judge_response = await judge(prompt, completion, answer, state)
        if "yes" in judge_response.lower():
            return 1.0
        else:
            return 0.0

    system_prompt = "Use the provided Wikipedia search tools to help answer questions."
    judge_rubric.add_reward_func(judge_reward_func, weight=1.0)
    rubric = vf.RubricGroup(rubrics=[tool_rubric, judge_rubric])
    vf_env = vf.ToolEnv(
        dataset=dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        tools=tools,
        max_turns=max_turns,
    )
    return vf_env
