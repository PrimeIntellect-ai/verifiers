from __future__ import annotations

import verifiers.v1 as vf
from verifiers.v1.utils.tool_utils import load_tools_from_state


def load_wiki():
    return {
        "paris": {
            "title": "Paris",
            "sections": {
                "lead": "Paris is the capital and most populous city of France.",
                "landmarks": "The Eiffel Tower is a landmark in Paris.",
            },
        },
        "ada_lovelace": {
            "title": "Ada Lovelace",
            "sections": {
                "lead": "Ada Lovelace is often regarded as the first computer programmer.",
                "work": "She wrote notes on Charles Babbage's Analytical Engine.",
            },
        },
    }


async def search_pages(query, wiki):
    query = query.lower()
    return [
        {"page_id": page_id, "title": page["title"]}
        for page_id, page in wiki.items()
        if query in page_id or query in page["title"].lower()
    ]


async def view_sections(page_id, wiki):
    return [
        {"section_id": f"{page_id}:{section_id}", "title": section_id}
        for section_id in wiki[page_id]["sections"]
    ]


async def read_section(section_id, wiki):
    page_id, section = section_id.split(":", 1)
    return wiki[page_id]["sections"][section]


@vf.teardown(priority=10)
async def teardown_wiki(wiki):
    wiki.clear()


@vf.reward(weight=1.0)
async def judge_answer(task, state) -> float:
    return float(task["answer"].lower() in state["answer"].lower())


def source():
    return [
        {
            "prompt": "Use wiki search to answer: what city is the capital of France?",
            "query": "Paris",
            "answer": "Paris",
        },
        {
            "prompt": "Use wiki search to answer: who wrote notes on the Analytical Engine?",
            "query": "Ada Lovelace",
            "answer": "Ada Lovelace",
        },
    ]


def load_toolset(config=None):
    return vf.Toolset(
        tools=[search_pages, view_sections, read_section],
        objects={"wiki": load_wiki},
        bindings={
            "search_pages.wiki": "objects.wiki",
            "view_sections.wiki": "objects.wiki",
            "read_section.wiki": "objects.wiki",
        },
        teardown=[teardown_wiki],
        config=config,
    )


async def wiki_program(task, state):
    tools = load_tools_from_state(state)
    pages = await tools["search_pages"](query=task["query"])
    sections = await tools["view_sections"](page_id=pages[0]["page_id"])
    content = await tools["read_section"](section_id=sections[0]["section_id"])
    state["tool_results"] = {
        "pages": pages,
        "sections": sections,
        "content": content,
    }
    state["answer"] = task["answer"]
    state["completion"] = [{"role": "assistant", "content": state["answer"]}]
    return state


def load_taskset(config=None):
    return vf.Taskset(
        source=source,
        rewards=[judge_answer],
        toolsets=[load_toolset(getattr(config, "toolset", None))],
        config=config,
    )


def load_harness(config=None):
    return vf.Harness(
        program=wiki_program,
        config=config,
    )


def load_environment(config=None):
    return vf.Env(
        taskset=load_taskset(getattr(config, "taskset", None)),
        harness=load_harness(getattr(config, "harness", None)),
    )
