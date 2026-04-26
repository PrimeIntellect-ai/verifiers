import asyncio
import os
from typing import Any, cast

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


def load_toolset(
    embed_model: str = "text-embedding-3-small",
    embed_base_url: str = "https://api.openai.com/v1",
    embed_api_key_var: str = "OPENAI_API_KEY",
    corpus_dataset: str = "willcb/rare-wiki-pages",
    corpus_split: str = "train",
    chroma_db_dir: str = CHROMA_DB_DIR,
) -> vf.Toolset:
    corpus_state: dict = {
        "loaded": False,
        "page_id_to_title": {},
        "page_id_to_content": {},
    }

    def load_corpus():
        if corpus_state["loaded"]:
            return
        corpus = load_dataset(corpus_dataset, split=corpus_split)
        for row in corpus:
            row = cast(dict, row)
            pid = row["id"]
            corpus_state["page_id_to_title"][pid] = row["title"]
            corpus_state["page_id_to_content"][pid] = row["content"]
        corpus_state["loaded"] = True

    chroma_state: dict[str, Any] = {
        "client": None,
        "collection": None,
    }

    def get_chroma_client() -> Any:
        import chromadb

        if chroma_state["client"] is None:
            chroma_state["client"] = chromadb.PersistentClient(path=chroma_db_dir)
        return chroma_state["client"]

    @vf.teardown(priority=10)
    async def teardown_chroma() -> None:
        chroma_state["client"] = None
        chroma_state["collection"] = None

    def get_collection():
        from chromadb.utils import embedding_functions

        load_corpus()
        if chroma_state["collection"] is None:
            openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                model_name=embed_model,
                api_base=embed_base_url,
                api_key=os.getenv(embed_api_key_var, "EMPTY"),
            )
            client = get_chroma_client()
            chroma_state["collection"] = client.get_or_create_collection(
                name="wiki_titles",
                embedding_function=openai_ef,
            )
            init_chroma(chroma_state["collection"])
        return chroma_state["collection"]

    def init_chroma(collection) -> None:
        page_id_to_title = corpus_state["page_id_to_title"]
        all_ids = list(page_id_to_title.keys())
        existing: set[str] = set()
        for i in range(0, len(all_ids), 500):
            batch = all_ids[i : i + 500]
            got = collection.get(ids=batch)
            existing.update(got.get("ids", []))
        missing = [pid for pid in all_ids if pid not in existing]
        if not missing:
            return
        documents = []
        metadatas = []
        for pid in missing:
            title = str(page_id_to_title[pid]).strip()
            if not title:
                raise ValueError(f"Empty title for page_id {pid}")
            documents.append(title)
            metadatas.append({"title": title})
        batch_size = 100
        for i in range(0, len(missing), batch_size):
            collection.upsert(
                ids=missing[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )

    def normalize_id(text: str) -> str:
        return text.strip().lower().replace(" ", "_")

    wiki: dict[str, Any] = {
        "load_corpus": load_corpus,
        "get_collection": get_collection,
        "normalize_id": normalize_id,
        "corpus_state": corpus_state,
    }

    async def search_pages(query: str, wiki) -> list[dict]:
        """Search for top 10 relevant articles using title embedding similarity."""
        collection = wiki["get_collection"]()
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

    async def view_sections(page_id: str, wiki) -> list[dict]:
        """View the sections of a page."""
        load_corpus = wiki["load_corpus"]
        normalize_id = wiki["normalize_id"]
        corpus_state = wiki["corpus_state"]
        load_corpus()
        content = corpus_state["page_id_to_content"][page_id]
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
            {
                "section_id": section["section_id"],
                "section_name": section["section_name"],
            }
            for section in sections
        ]

    async def read_section(section_id: str, wiki) -> str:
        """Read a section of a page."""
        if ":" not in section_id:
            raise ValueError(
                "Invalid section_id format. Expected: page_id:section_name"
            )
        page_id, section_name_id = section_id.split(":", 1)
        load_corpus = wiki["load_corpus"]
        normalize_id = wiki["normalize_id"]
        corpus_state = wiki["corpus_state"]
        load_corpus()
        content = corpus_state["page_id_to_content"][page_id]
        lines = content.split("\n")
        if section_name_id == "full":
            return content

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
        if section_end is None:
            section_end = len(lines)
        return "\n".join(lines[section_start:section_end])

    return vf.Toolset(
        bindings={"wiki": wiki},
        channels={"teardown": teardown_chroma},
        name="wiki_search",
        tools=[search_pages, view_sections, read_section],
    )


def load_rubric(
    judge_model: str = "gpt-4.1-mini",
    judge_base_url: str = "https://api.openai.com/v1",
    judge_api_key_var: str = "OPENAI_API_KEY",
) -> vf.Rubric:
    parser = vf.Parser()
    judge_client = AsyncOpenAI(
        base_url=judge_base_url, api_key=os.getenv(judge_api_key_var, "")
    )
    judge_rubric = JudgeRubric(
        judge_client=judge_client,
        judge_model=judge_model,
        parser=parser,
        judge_prompt=JUDGE_PROMPT,
    )

    async def judge_reward_func(judge, prompt, completion, answer, state) -> float:
        judge_response = await judge(prompt, completion, answer, state)
        return float("yes" in judge_response.lower())

    judge_rubric.add_reward_func(judge_reward_func, weight=1.0)
    return judge_rubric


def load_taskset(
    judge_model: str = "gpt-4.1-mini",
    judge_base_url: str = "https://api.openai.com/v1",
    judge_api_key_var: str = "OPENAI_API_KEY",
    embed_model: str = "text-embedding-3-small",
    embed_base_url: str = "https://api.openai.com/v1",
    embed_api_key_var: str = "OPENAI_API_KEY",
    corpus_dataset: str = "willcb/rare-wiki-pages",
    corpus_split: str = "train",
    chroma_db_dir: str = CHROMA_DB_DIR,
) -> vf.Taskset:
    return vf.Taskset(
        source=lambda: load_dataset("willcb/wiki-trivia-questions-v4", split="train"),
        rubric=lambda: load_rubric(
            judge_model=judge_model,
            judge_base_url=judge_base_url,
            judge_api_key_var=judge_api_key_var,
        ),
        tools=lambda: load_toolset(
            embed_model=embed_model,
            embed_base_url=embed_base_url,
            embed_api_key_var=embed_api_key_var,
            corpus_dataset=corpus_dataset,
            corpus_split=corpus_split,
            chroma_db_dir=chroma_db_dir,
        ),
    )


def load_harness(max_turns: int = 10) -> vf.Harness:
    return vf.Harness(
        system_prompt="Use the provided Wikipedia search tools to help answer questions.",
        run=vf.RunConfig(max_turns=max_turns),
    )


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
    return vf.Env(
        taskset=load_taskset(
            judge_model=judge_model,
            judge_base_url=judge_base_url,
            judge_api_key_var=judge_api_key_var,
            embed_model=embed_model,
            embed_base_url=embed_base_url,
            embed_api_key_var=embed_api_key_var,
            corpus_dataset=corpus_dataset,
            corpus_split=corpus_split,
            chroma_db_dir=chroma_db_dir,
        ),
        harness=load_harness(max_turns=max_turns),
    )


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
