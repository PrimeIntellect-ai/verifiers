import verifiers.v1 as vf

BLOCKED_WORD = "platform.openai.com"


class WebSearchInterceptionTask(vf.Task):
    # Native search has already run here, but its buffered result has not reached Codex.
    @vf.on_response
    def terminate_on_result(self, response: vf.Response) -> vf.Terminate | None:
        if (
            any(
                event.name == "web_search"
                for event in response.message.provider_tools or []
            )
            and BLOCKED_WORD in response.message.model_dump_json().casefold()
        ):
            # Stop the rollout instead of delivering the matching search result.
            return vf.Terminate(
                reason=f"Web-search response contained {BLOCKED_WORD!r}", reward=1.0
            )


class WebSearchInterceptionTaskset(vf.Taskset[WebSearchInterceptionTask]):
    def load(self) -> list[WebSearchInterceptionTask]:
        prompt = (
            "Use your native web search to find the official OpenAI Responses API "
            "documentation. Cite the source and include the search query."
        )
        return [
            WebSearchInterceptionTask(
                vf.TaskData(idx=0, prompt=prompt), self.config.task
            )
        ]
