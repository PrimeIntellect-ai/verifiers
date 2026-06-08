from __future__ import annotations

import re
from typing import cast

from datasets import load_dataset

import verifiers.v1 as vf

from .servers.wiki import WikiToolsetConfig

SYSTEM_PROMPT = "Use the provided Wikipedia search tools to help answer questions."
TOKEN_RE = re.compile(r"[a-z0-9]+")


class WikiIndex:
    def __init__(
        self,
        *,
        page_id_to_title: dict[str, str],
        page_id_to_content: dict[str, str],
    ):
        self.page_id_to_title = page_id_to_title
        self.page_id_to_content = page_id_to_content


class WikiSearchTasksetConfig(vf.TasksetConfig):
    max_turns: int = 10
    toolsets: vf.ToolsetConfigs = {"wiki": WikiToolsetConfig()}


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

    @vf.reward(weight=1.0)
    async def answer_in_response(self, task: WikiSearchTask, state: vf.State) -> float:
        messages = [
            message for message in state.completion if message.role == "assistant"
        ]
        response = str(messages[-1].content or "") if messages else ""
        return float(task.answer.lower() in response.lower())


def load_wiki(config: WikiToolsetConfig) -> WikiIndex:
    page_id_to_title: dict[str, str] = {}
    page_id_to_content: dict[str, str] = {}
    corpus = load_dataset(config.corpus_dataset, split=config.corpus_split)
    for row in corpus:
        record = cast(dict[str, object], row)
        page_id = str(record["id"])
        page_id_to_title[page_id] = str(record["title"])
        page_id_to_content[page_id] = str(record["content"])
    return WikiIndex(
        page_id_to_title=page_id_to_title,
        page_id_to_content=page_id_to_content,
    )


def normalize_id(text: str) -> str:
    return text.strip().lower().replace(" ", "_")


def tokenize(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


async def search_pages(query: str, wiki: WikiIndex) -> list[dict[str, str]]:
    query_tokens = tokenize(query)
    if not query_tokens:
        raise ValueError("Search query must contain at least one alphanumeric token.")
    ranked: list[tuple[int, str, str]] = []
    for page_id, title in wiki.page_id_to_title.items():
        title_score = len(query_tokens & tokenize(title))
        content_score = len(query_tokens & tokenize(wiki.page_id_to_content[page_id]))
        score = 5 * title_score + content_score
        ranked.append((-score, title.lower(), page_id))
    ranked.sort()
    return [
        {
            "page_id": page_id,
            "title": wiki.page_id_to_title[page_id],
        }
        for _, _, page_id in ranked[:10]
    ]


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
