import os

import chromadb  # type: ignore
from chromadb.utils import embedding_functions  # type: ignore
from datasets import load_dataset
from openai import OpenAI

import verifiers as vf
from verifiers.rubrics.judge_rubric import JudgeRubric

WIKI_DIR = "data/wiki"
CHROMA_DB_DIR = ".chroma_db"


def load_environment(
    max_turns: int = 10,
    judge_model: str = "gpt-4.1-mini",
    judge_base_url: str = "https://api.openai.com/v1",
    judge_api_key_var: str = "OPENAI_API_KEY",
    embed_model: str = "text-embedding-3-small",
    embed_base_url: str = "https://api.openai.com/v1",
    embed_api_key_var: str = "OPENAI_API_KEY",
    wiki_dir: str = WIKI_DIR,
    chroma_db_dir: str = CHROMA_DB_DIR,
) -> vf.Environment:
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv(embed_api_key_var, "EMPTY"),
        model_name=embed_model,
        base_url=embed_base_url,
    )
    db_client = chromadb.PersistentClient(path=chroma_db_dir)
    collection = db_client.get_collection("wiki_titles", embedding_function=openai_ef)  # type: ignore

    def search_pages(query: str) -> list[dict]:
        """Search for top 10 relevant articles using title embedding similarity.

        Args:
            query (str): The query to search for.

        Returns:
            list[dict]: A list of dicts with page_id and title.

        Examples:
            "basketball" -> [{"page_id": "basketball", "title": "Basketball"}, {"page_id": "basketball_rules", "title": "Basketball Rules"}, ...]
        """
        results = collection.query(query_texts=[query], n_results=10)

        # Format results
        output = []
        for i in range(len(results["ids"][0])):
            output.append(
                {
                    "page_id": results["ids"][0][i],
                    "title": results["metadatas"][0][i]["title"],  # type: ignore
                }
            )

        return output

    def view_sections(page_id: str) -> list[dict]:
        """View the sections of a page.

        Args:
            page_id (str): The ID of the page to view.

        Returns:
            list[dict]: A list of dicts with section_id and section_name.

        Examples:
            "basketball" -> [{"section_id": "basketball:history", "section_name": "History"}, ...]
        """
        # Find the file for this page_id
        results = collection.get(ids=[page_id])
        if not results["ids"]:
            raise ValueError(f"Page not found: {page_id}")

        filename = results["metadatas"][0]["title"] + ".md"  # type: ignore
        filepath = os.path.join(wiki_dir, filename)  # type: ignore

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        sections = []

        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.startswith("#"):
                # Extract section name (remove # and whitespace)
                section_name = line.lstrip("#").strip()
                # Create section ID
                section_id = f"{page_id}:{section_name.lower().replace(' ', '_')}"
                sections.append(
                    {
                        "section_id": section_id,
                        "section_name": section_name,
                        "start_line": i,
                    }
                )

        # If no sections found, return the whole page as one section
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

    def read_section(section_id: str) -> str:
        """Read a section of a page.

        Args:
            section_id (str): The ID of the section to read.

        Returns:
            str: The content of the section.

        Examples:
            "baseball:finnish_baseball" -> "Finnish baseball is a sport that is played in Finland..."
        """
        # Parse section_id
        if ":" not in section_id:
            raise ValueError(
                "Invalid section_id format. Expected: page_id:section_name"
            )

        page_id, section_name_id = section_id.split(":", 1)

        # Get the file
        results = collection.get(ids=[page_id])
        if not results["ids"]:
            raise ValueError(f"Page not found: {page_id}")

        filename = results["metadatas"][0]["title"] + ".md"  # type: ignore
        filepath = os.path.join(wiki_dir, filename)

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        lines = content.split("\n")

        # Special case for "full" section
        if section_name_id == "full":
            return content

        # Find the section
        section_start = None
        section_end = None

        for i, line in enumerate(lines):
            if line.startswith("#"):
                current_section = line.lstrip("#").strip().lower().replace(" ", "_")
                if current_section == section_name_id and section_start is None:
                    section_start = i
                elif section_start is not None and section_end is None:
                    section_end = i
                    break

        # If section found
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

    dataset = load_dataset("willcb/wiki-trivia-questions", split="train")

    parser = vf.ThinkParser()
    vf_env = vf.ToolEnv(
        dataset=dataset,
        parser=parser,
        tools=tools,
        max_turns=max_turns,
    )

    judge_client = OpenAI(base_url=judge_base_url, api_key=os.getenv(judge_api_key_var))
    judge_rubric = JudgeRubric(
        judge_client=judge_client, judge_model=judge_model, parser=vf_env.parser
    )

    async def judge_reward_func(judge, prompt, completion, answer, state) -> float:
        judge_response = await judge(prompt, completion, answer, state)
        if "yes" in judge_response.lower():
            return 1.0
        else:
            return 0.0

    judge_rubric.add_reward_func(judge_reward_func, weight=1.0)
    vf_env.rubric = vf.RubricGroup(rubrics=[judge_rubric, vf_env.rubric])

    return vf_env
