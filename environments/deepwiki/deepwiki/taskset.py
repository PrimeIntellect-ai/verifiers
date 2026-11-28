"""Remote-tool example backed by the public DeepWiki MCP server.

Unlike the locally authored `glossary` and worker-shared `wiki_search` examples,
`DeepWikiToolset` declares no local `@tool` methods. Its config supplies an existing
streamable-HTTP URL, so verifiers connects the harness directly to that remote server.
The harness runtime therefore needs outbound network access.
"""

import json

import verifiers.v1 as vf
from deepwiki.servers.deepwiki import DEEPWIKI_URL, DeepWikiToolset

# (repository, expected primary language) pairs chosen for unambiguous answers.
TASKS = [
    ("modelcontextprotocol/python-sdk", "python"),
    ("tokio-rs/tokio", "rust"),
]
# The server's question tool, as the model sees it. DeepWiki renamed `ask_question` to
# `ask_wiki_question`; accept both so the reward doesn't depend on the server version.
ASK_TOOL = "deepwiki_ask_wiki_question"
ASK_TOOLS = {ASK_TOOL, "deepwiki_ask_question"}


class DeepWikiTaskConfig(vf.TaskConfig):
    tools: vf.ToolsetConfig = vf.ToolsetConfig(url=DEEPWIKI_URL)


class DeepWikiTaskData(vf.TaskData):
    answer: str
    """The language the repo is written in (the model's whole reply, ignoring case)."""


def _asks_about(call: vf.ToolCall, repo: str) -> bool:
    """Whether a question call names `repo` (`repoName` is one repository or a list)."""
    try:
        arguments = json.loads(call.arguments)
    except json.JSONDecodeError:
        return False
    names = arguments.get("repoName") if isinstance(arguments, dict) else None
    names = [names] if isinstance(names, str) else names
    return isinstance(names, list) and repo.lower() in (
        str(name).lower() for name in names
    )


def _text(content: vf.MessageContent) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(
        part.text for part in content if isinstance(part, vf.TextContentPart)
    )


def asked_deepwiki(trace: vf.Trace, repo: str) -> bool:
    """Whether a question about `repo` got an answer back. Harness and server errors come
    back as text starting with "error", e.g. `error: invalid JSON in tool arguments`."""
    call_ids = {
        call.id
        for message in trace.assistant_messages
        for call in message.tool_calls or []
        if call.name in ASK_TOOLS and _asks_about(call, repo)
    }
    return any(
        message.tool_call_id in call_ids
        and not _text(message.content).strip().lower().startswith("error")
        for message in trace.tool_messages
    )


class DeepWikiTask(vf.Task[DeepWikiTaskData, vf.State, DeepWikiTaskConfig]):
    @classmethod
    def toolsets(cls, config: DeepWikiTaskConfig) -> list[vf.Toolset]:
        return [DeepWikiToolset(config.tools)]

    @vf.reward(weight=1.0)
    async def answered(self, trace: vf.Trace) -> float:
        # Match the whole reply, not a substring, so listing every language can't pass, and
        # require the tool this example exists to exercise.
        reply = trace.last_reply.strip("`*_. \n").lower()
        return float(
            asked_deepwiki(trace, self.data.name) and reply == self.data.answer.lower()
        )


class DeepWikiConfig(vf.TasksetConfig):
    task: DeepWikiTaskConfig = DeepWikiTaskConfig()


class DeepWikiTaskset(vf.Taskset[DeepWikiTask, DeepWikiConfig]):
    def load(self) -> list[DeepWikiTask]:
        return [
            DeepWikiTask(
                DeepWikiTaskData(
                    name=repo,
                    prompt=(
                        f"Use the `{ASK_TOOL}` tool to ask what programming "
                        f'language the "{repo}" GitHub repository is primarily written in. '
                        "Then reply with just the language name."
                    ),
                    answer=language,
                ),
                self.config.task,
            )
            for repo, language in TASKS
        ]
