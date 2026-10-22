"""Remote-tool example backed by the public DeepWiki MCP server.

Unlike the locally authored `glossary` and worker-shared `wiki_search` examples,
`DeepWikiToolset` declares no local `@tool` methods. Its config supplies an existing
streamable-HTTP URL, so verifiers connects the harness directly to that remote server.
The harness runtime therefore needs outbound network access.
"""

import verifiers.v1 as vf
from deepwiki.servers.deepwiki import DEEPWIKI_URL, DeepWikiToolset

# (repository, expected primary language) pairs chosen for unambiguous answers.
TASKS = [
    ("modelcontextprotocol/python-sdk", "python"),
    ("tokio-rs/tokio", "rust"),
]


class DeepWikiTaskConfig(vf.TaskConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig(url=DEEPWIKI_URL)


class DeepWikiTaskData(vf.TaskData):
    answer: str
    """The language the repo is written in (the model's whole reply, ignoring case)."""


def asked_deepwiki(trace: vf.Trace) -> bool:
    """Whether a `deepwiki_ask_question` call got a result back."""
    call_ids = {
        call.id
        for message in trace.assistant_messages
        for call in message.tool_calls or []
        if call.name == "deepwiki_ask_question"
    }
    return any(message.tool_call_id in call_ids for message in trace.tool_messages)


class DeepWikiTask(vf.Task[DeepWikiTaskData, vf.State, DeepWikiTaskConfig]):
    @classmethod
    def toolsets(cls, config: DeepWikiTaskConfig) -> list[vf.Toolset]:
        return [DeepWikiToolset(config.tools)]

    @vf.reward(weight=1.0)
    async def answered(self, trace: vf.Trace) -> float:
        # Match the whole reply, not a substring, so listing every language can't pass, and
        # require the tool this example exists to exercise.
        reply = trace.last_reply.strip("`*_. \n").lower()
        return float(asked_deepwiki(trace) and reply == self.data.answer)


class DeepWikiConfig(vf.TasksetConfig):
    task: DeepWikiTaskConfig = DeepWikiTaskConfig()


class DeepWikiTaskset(vf.Taskset[DeepWikiTask, DeepWikiConfig]):
    def load(self) -> list[DeepWikiTask]:
        return [
            DeepWikiTask(
                DeepWikiTaskData(
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
            for repo, language in TASKS
        ]
