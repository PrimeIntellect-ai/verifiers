import verifiers.v1 as vf

BLOCKED_WORD = "openai.com"


class WebSearchInterceptionTask(vf.Task):
    @vf.stop
    def stop_on_result(self, response: vf.Response) -> bool:
        # Native web-search citations remain on the buffered provider response.
        sources = (
            annotation.get("url", "")
            for item in response.message.provider_state or []
            if item.get("type") == "message"
            for part in item.get("content") or []
            for annotation in part.get("annotations") or []
            if annotation.get("type") == "url_citation"
        )
        return any(BLOCKED_WORD in source.casefold() for source in sources)

    @vf.reward
    async def blocked(self, trace: vf.Trace) -> float:
        return float(trace.stop_condition == "stop_on_result")


class WebSearchInterceptionTaskset(vf.Taskset[WebSearchInterceptionTask]):
    def load(self) -> list[WebSearchInterceptionTask]:
        return [
            WebSearchInterceptionTask(
                vf.TaskData(
                    idx=0,
                    prompt=(
                        "Use native web search to find the official OpenAI Responses "
                        "API documentation. Cite the source and include the query."
                    ),
                ),
                self.config.task,
            )
        ]
