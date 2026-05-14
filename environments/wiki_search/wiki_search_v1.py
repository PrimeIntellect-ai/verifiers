import asyncio
import os
from typing import cast

import chromadb
from chromadb.api.types import Embeddable, EmbeddingFunction
from chromadb.utils import embedding_functions
from datasets import load_dataset
from openai import AsyncOpenAI

import verifiers as vf

CHROMA_DB_DIR = ".chroma_db"
_chroma_semaphore: asyncio.Semaphore | None = None

SYSTEM_PROMPT = "Use the provided Wikipedia search tools to help answer questions."
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


def get_chroma_semaphore() -> asyncio.Semaphore:
    global _chroma_semaphore
    if _chroma_semaphore is None:
        _chroma_semaphore = asyncio.Semaphore(100)
    return _chroma_semaphore


def load_wiki(
    corpus_dataset: str,
    corpus_split: str,
    chroma_db_dir: str,
    embed_model: str,
    embed_base_url: str,
    embed_api_key_var: str,
) -> vf.ConfigData:
    page_id_to_title: dict[str, str] = {}
    page_id_to_content: dict[str, str] = {}
    corpus = load_dataset(corpus_dataset, split=corpus_split)
    for row in corpus:
        row = cast(dict, row)
        page_id_to_title[row["id"]] = row["title"]
        page_id_to_content[row["id"]] = row["content"]

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
    return {
        "collection": collection,
        "page_id_to_title": page_id_to_title,
        "page_id_to_content": page_id_to_content,
    }


def init_chroma(collection, page_id_to_title: dict[str, str]) -> None:
    all_ids = list(page_id_to_title)
    existing: set[str] = set()
    for i in range(0, len(all_ids), 500):
        batch = all_ids[i : i + 500]
        got = collection.get(ids=batch)
        existing.update(got.get("ids", []))
    missing = [page_id for page_id in all_ids if page_id not in existing]
    if not missing:
        return
    documents = []
    metadatas = []
    for page_id in missing:
        title = str(page_id_to_title[page_id]).strip()
        if not title:
            raise ValueError(f"Empty title for page_id {page_id}")
        documents.append(title)
        metadatas.append({"title": title})
    for i in range(0, len(missing), 100):
        collection.upsert(
            ids=missing[i : i + 100],
            documents=documents[i : i + 100],
            metadatas=metadatas[i : i + 100],
        )


def normalize_id(text: str) -> str:
    return text.strip().lower().replace(" ", "_")


async def search_pages(query: str, wiki) -> list[dict]:
    """Search for top 10 relevant articles using title embedding similarity."""
    async with get_chroma_semaphore():
        results = await asyncio.to_thread(
            wiki["collection"].query, query_texts=[query], n_results=10
        )
    if not results or not results["metadatas"]:
        raise ValueError(f"No results found for query: {query}")
    output = []
    for i in range(len(results["ids"][0])):
        output.append(
            {
                "page_id": results["ids"][0][i],
                "title": results["metadatas"][0][i]["title"],
            }
        )
    return output


async def view_sections(page_id: str, wiki) -> list[dict]:
    """View the sections of a page."""
    content = wiki["page_id_to_content"][page_id]
    sections = []
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("#"):
            section_name = line.lstrip("#").strip()
            sections.append(
                {
                    "section_id": f"{page_id}:{normalize_id(section_name)}",
                    "section_name": section_name,
                    "start_line": i,
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
        {"section_id": section["section_id"], "section_name": section["section_name"]}
        for section in sections
    ]


async def read_section(section_id: str, wiki) -> str:
    """Read a section of a page."""
    if ":" not in section_id:
        raise ValueError("Invalid section_id format. Expected: page_id:section_name")
    page_id, section_name_id = section_id.split(":", 1)
    content = wiki["page_id_to_content"][page_id]
    if section_name_id == "full":
        return content
    lines = content.split("\n")
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
    if section_start is None:
        raise ValueError(f"Section not found: {section_id}")
    return "\n".join(lines[section_start : section_end or len(lines)])


def build_source(max_turns: int = 10):
    def source():
        dataset = load_dataset("willcb/wiki-trivia-questions-v4", split="train")
        for index, row in enumerate(dataset):
            row = cast(dict, row)
            yield {
                **row,
                "example_id": index,
                "max_turns": max_turns,
                "prompt": [{"role": "user", "content": row["question"]}],
            }

    return source


def judge_reward_factory(
    judge_model: str,
    judge_base_url: str,
    judge_api_key_var: str,
):
    @vf.reward(weight=1.0)
    async def judge_reward_func(task, state) -> float:
        completion = state.get("completion") or []
        messages = vf.get_messages(completion, role="assistant")
        response = str(messages[-1].content or "") if messages else ""
        prompt = JUDGE_PROMPT.format(
            question=task["question"],
            answer=task["answer"],
            response=response,
        )
        judge_client = AsyncOpenAI(
            base_url=judge_base_url,
            api_key=os.getenv(judge_api_key_var, ""),
        )
        try:
            result = await judge_client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
            )
        finally:
            await judge_client.close()
        text = result.choices[0].message.content or ""
        return 1.0 if "yes" in text.lower() else 0.0

    return judge_reward_func


def load_toolset(
    corpus_dataset: str = "willcb/rare-wiki-pages",
    corpus_split: str = "train",
    chroma_db_dir: str = CHROMA_DB_DIR,
    embed_model: str = "text-embedding-3-small",
    embed_base_url: str = "https://api.openai.com/v1",
    embed_api_key_var: str = "OPENAI_API_KEY",
    config=None,
):
    def load_wiki_index() -> vf.ConfigData:
        return load_wiki(
            corpus_dataset=corpus_dataset,
            corpus_split=corpus_split,
            chroma_db_dir=chroma_db_dir,
            embed_model=embed_model,
            embed_base_url=embed_base_url,
            embed_api_key_var=embed_api_key_var,
        )

    return vf.Toolset(
        tools=[search_pages, view_sections, read_section],
        objects={"wiki": load_wiki_index},
        bindings={
            "search_pages.wiki": "objects.wiki",
            "view_sections.wiki": "objects.wiki",
            "read_section.wiki": "objects.wiki",
        },
        config=config,
    )


def load_taskset(
    max_turns: int = 10,
    judge_model: str = "gpt-4.1-mini",
    judge_base_url: str = "https://api.openai.com/v1",
    judge_api_key_var: str = "OPENAI_API_KEY",
    corpus_dataset: str = "willcb/rare-wiki-pages",
    corpus_split: str = "train",
    chroma_db_dir: str = CHROMA_DB_DIR,
    embed_model: str = "text-embedding-3-small",
    embed_base_url: str = "https://api.openai.com/v1",
    embed_api_key_var: str = "OPENAI_API_KEY",
    config=None,
):
    return vf.Taskset(
        source=build_source(max_turns=max_turns),
        system_prompt=SYSTEM_PROMPT,
        rewards=[
            judge_reward_factory(
                judge_model=judge_model,
                judge_base_url=judge_base_url,
                judge_api_key_var=judge_api_key_var,
            )
        ],
        toolsets=[
            load_toolset(
                corpus_dataset=corpus_dataset,
                corpus_split=corpus_split,
                chroma_db_dir=chroma_db_dir,
                embed_model=embed_model,
                embed_base_url=embed_base_url,
                embed_api_key_var=embed_api_key_var,
            )
        ],
        config=config,
    )


def load_v1_environment(
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
) -> vf.Env:
    return vf.Env(
        taskset=load_taskset(
            max_turns=max_turns,
            judge_model=judge_model,
            judge_base_url=judge_base_url,
            judge_api_key_var=judge_api_key_var,
            corpus_dataset=corpus_dataset,
            corpus_split=corpus_split,
            chroma_db_dir=chroma_db_dir,
            embed_model=embed_model,
            embed_base_url=embed_base_url,
            embed_api_key_var=embed_api_key_var,
        )
    )
