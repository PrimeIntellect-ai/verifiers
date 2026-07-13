"""Remote-tool example backed by the public DeepWiki MCP server.

Unlike the locally authored `glossary` and worker-shared `wiki_search` examples,
`DeepWikiToolset` declares no local `@tool` methods. Its config supplies an existing
streamable-HTTP URL, so verifiers connects the harness directly to that remote server.
The harness runtime therefore needs outbound network access.
"""

import verifiers.v1 as vf

from deepwiki_v1.servers.deepwiki import DEEPWIKI_URL, DeepWikiToolset

# (repository, expected primary language) pairs chosen for unambiguous answers.
TASKS = [
    ("modelcontextprotocol/python-sdk", "python"),
    ("tokio-rs/tokio", "rust"),
]


class DeepWikiTaskConfig(vf.TaskConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig(url=DEEPWIKI_URL)


class DeepWikiTaskData(vf.TaskData):
    answer: str
    """The language the repo is written in (must appear in the model's reply)."""


class DeepWikiTask(vf.Task[DeepWikiTaskData, vf.State, DeepWikiTaskConfig]):
    tools = (DeepWikiToolset,)

    @vf.reward(weight=1.0)
    async def answered(self, trace: vf.Trace) -> float:
        last = trace.last_reply
        return float(self.data.answer.lower() in (last or "").lower())


class DeepWikiConfig(vf.TasksetConfig):
    task: DeepWikiTaskConfig = DeepWikiTaskConfig()


class DeepWikiTaskset(vf.Taskset[DeepWikiTask, DeepWikiConfig]):
    def load(self) -> list[DeepWikiTask]:
        return [
            DeepWikiTask(
                DeepWikiTaskData(
                    idx=i,
                    name=repo,
                    prompt=(
                        f"Use the `deepwiki_ask_question` tool to ask what programming "
                        f'language the "{repo}" GitHub repository is primarily written in. '
                        "Then reply with just the language name."
                    ),
                    answer=language,
                ),
                self.config.task,
            )
            for i, (repo, language) in enumerate(TASKS)
        ]
