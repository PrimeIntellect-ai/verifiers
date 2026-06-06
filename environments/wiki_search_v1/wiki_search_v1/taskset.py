from __future__ import annotations

import asyncio
import os
import sys
from typing import TYPE_CHECKING, cast

from datasets import load_dataset

import verifiers.v1 as vf

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection

CHROMA_DB_DIR = ".chroma_db"
SYSTEM_PROMPT = "Use the provided Wikipedia search tools to help answer questions."
_chroma_semaphore: asyncio.Semaphore | None = None


class WikiIndex:
    def __init__(
        self,
        *,
        collection: Collection,
        page_id_to_title: dict[str, str],
        page_id_to_content: dict[str, str],
    ):
        self.collection = collection
        self.page_id_to_title = page_id_to_title
        self.page_id_to_content = page_id_to_content


class WikiSearchTasksetConfig(vf.TasksetConfig):
    max_turns: int = 10
    corpus_dataset: str = "willcb/rare-wiki-pages"
    corpus_split: str = "train"
    chroma_db_dir: str = CHROMA_DB_DIR
    embed_model: str = "text-embedding-3-small"
    embed_base_url: str = "https://api.openai.com/v1"
    embed_api_key_var: str = "OPENAI_API_KEY"


class WikiSearchTask(vf.Task):
    question: str
    answer: str


class WikiSearchTaskset(vf.Taskset[WikiSearchTasksetConfig]):
    task_type = WikiSearchTask

    def load_tasks(self, split: vf.TaskSplit = "train") -> vf.Tasks:
        _ = split
        dataset = load_dataset("willcb/wiki-trivia-questions-v4", split="train")
        for index, row in enumerate(dataset):
            record = cast(dict[str, object], row)
            yield {
                "row_id": index,
                "question": str(record["question"]),
                "answer": str(record["answer"]),
                "max_turns": self.config.max_turns,
                "prompt": [{"role": "user", "content": str(record["question"])}],
            }

    def load_system_prompt(self, config: WikiSearchTasksetConfig) -> vf.SystemPrompt:
        _ = config
        return SYSTEM_PROMPT

    def load_toolsets(self, config: WikiSearchTasksetConfig) -> vf.Toolsets:
        return [
            vf.Toolset(
                name="wiki",
                scope="env",
                server=vf.MCPServerSpec(
                    command=[sys.executable, "-m", "wiki_search_v1.servers.tools"],
                    env={
                        "VF_WIKI_CORPUS_DATASET": config.corpus_dataset,
                        "VF_WIKI_CORPUS_SPLIT": config.corpus_split,
                        "VF_WIKI_CHROMA_DB_DIR": config.chroma_db_dir,
                        "VF_WIKI_EMBED_MODEL": config.embed_model,
                        "VF_WIKI_EMBED_BASE_URL": config.embed_base_url,
                        "VF_WIKI_EMBED_API_KEY_VAR": config.embed_api_key_var,
                    },
                ),
            )
        ]

    @vf.reward(weight=1.0)
    async def answer_in_response(self, task: WikiSearchTask, state: vf.State) -> float:
        messages = vf.get_messages(state.completion, role="assistant")
        response = str(messages[-1].content or "") if messages else ""
        return float(task.answer.lower() in response.lower())


def get_chroma_semaphore() -> asyncio.Semaphore:
    global _chroma_semaphore
    if _chroma_semaphore is None:
        _chroma_semaphore = asyncio.Semaphore(100)
    return _chroma_semaphore


def load_wiki() -> WikiIndex:
    import chromadb
    from chromadb.api.types import Embeddable, EmbeddingFunction
    from chromadb.utils import embedding_functions

    corpus_dataset = os.environ["VF_WIKI_CORPUS_DATASET"]
    corpus_split = os.environ["VF_WIKI_CORPUS_SPLIT"]
    chroma_db_dir = os.environ["VF_WIKI_CHROMA_DB_DIR"]
    embed_model = os.environ["VF_WIKI_EMBED_MODEL"]
    embed_base_url = os.environ["VF_WIKI_EMBED_BASE_URL"]
    embed_api_key_var = os.environ["VF_WIKI_EMBED_API_KEY_VAR"]
    page_id_to_title: dict[str, str] = {}
    page_id_to_content: dict[str, str] = {}
    corpus = load_dataset(corpus_dataset, split=corpus_split)
    for row in corpus:
        record = cast(dict[str, object], row)
        page_id = str(record["id"])
        page_id_to_title[page_id] = str(record["title"])
        page_id_to_content[page_id] = str(record["content"])

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
    init_chroma(collection, page_id_to_title)
    return WikiIndex(
        collection=collection,
        page_id_to_title=page_id_to_title,
        page_id_to_content=page_id_to_content,
    )


def init_chroma(collection: Collection, page_id_to_title: dict[str, str]) -> None:
    all_ids = list(page_id_to_title)
    existing: set[str] = set()
    for index in range(0, len(all_ids), 500):
        batch = all_ids[index : index + 500]
        got = collection.get(ids=batch)
        existing.update(got.get("ids", []))
    missing = [page_id for page_id in all_ids if page_id not in existing]
    if not missing:
        return
    documents = []
    metadatas = []
    for page_id in missing:
        title = page_id_to_title[page_id].strip()
        if not title:
            raise ValueError(f"Empty title for page_id {page_id}")
        documents.append(title)
        metadatas.append({"title": title})
    for index in range(0, len(missing), 100):
        collection.upsert(
            ids=missing[index : index + 100],
            documents=documents[index : index + 100],
            metadatas=metadatas[index : index + 100],
        )


def normalize_id(text: str) -> str:
    return text.strip().lower().replace(" ", "_")


async def search_pages(query: str, wiki: WikiIndex) -> list[dict[str, str]]:
    async with get_chroma_semaphore():
        results = await asyncio.to_thread(
            wiki.collection.query,
            query_texts=[query],
            n_results=10,
        )
    if not results or not results["metadatas"]:
        raise ValueError(f"No results found for query: {query}")
    output: list[dict[str, str]] = []
    for index in range(len(results["ids"][0])):
        output.append(
            {
                "page_id": str(results["ids"][0][index]),
                "title": str(results["metadatas"][0][index]["title"]),
            }
        )
    return output


async def view_sections(page_id: str, wiki: WikiIndex) -> list[dict[str, str]]:
    content = wiki.page_id_to_content[page_id]
    sections: list[dict[str, str | int]] = []
    lines = content.split("\n")
    for index, line in enumerate(lines):
        if line.startswith("#"):
            section_name = line.lstrip("#").strip()
            sections.append(
                {
                    "section_id": f"{page_id}:{normalize_id(section_name)}",
                    "section_name": section_name,
                    "start_line": index,
                }
            )
    if not sections:
        sections.append(
            {
                "section_id": f"{page_id}:full",
                "section_name": "Full Page",
                "start_line": 0,
            }
        )
    return [
        {
            "section_id": str(section["section_id"]),
            "section_name": str(section["section_name"]),
        }
        for section in sections
    ]


async def read_section(section_id: str, wiki: WikiIndex) -> str:
    if ":" not in section_id:
        raise ValueError("Invalid section_id format. Expected: page_id:section_name")
    page_id, section_name_id = section_id.split(":", 1)
    content = wiki.page_id_to_content[page_id]
    if section_name_id == "full":
        return content
    lines = content.split("\n")
    section_start = None
    section_end = None
    for index, line in enumerate(lines):
        if line.startswith("#"):
            current_section = normalize_id(line.lstrip("#").strip())
            if current_section == section_name_id and section_start is None:
                section_start = index
            elif section_start is not None and section_end is None:
                section_end = index
                break
    if section_start is None:
        raise ValueError(f"Section not found: {section_id}")
    return "\n".join(lines[section_start : section_end or len(lines)])


def load_taskset(config: WikiSearchTasksetConfig) -> WikiSearchTaskset:
    return WikiSearchTaskset(config=config)
