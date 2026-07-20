"""Textify E2E fixture: images in the task prompt and a user-simulator turn."""

import verifiers.v1 as vf

from tool_response_image_v1 import EXPECTED_URL


class TextifyState(vf.State):
    turns: int = 0
    done: bool = False


class TextifyUser(vf.User[vf.UserConfig, TextifyState]):
    async def respond(self, message: str) -> vf.Messages:
        self.state.turns += 1
        if self.state.turns > 1:
            self.state.done = True
            return []
        return [
            vf.UserMessage(
                content=[
                    vf.TextContentPart(text="Reply with done."),
                    vf.ImageUrlContentPart(
                        image_url=vf.ImageUrlSource(url=EXPECTED_URL)
                    ),
                ]
            )
        ]


class TextifyData(vf.TaskData):
    pass


class TextifyTaskConfig(vf.TaskConfig):
    user: vf.UserConfig = vf.UserConfig()


class TextifyConfig(vf.TasksetConfig):
    task: TextifyTaskConfig = TextifyTaskConfig()


class TextifyTask(vf.Task[TextifyData, TextifyState, TextifyTaskConfig]):
    user = TextifyUser

    @vf.stop
    async def done(self, trace: vf.Trace[TextifyData, TextifyState]) -> bool:
        return trace.state.done

    @vf.reward(weight=1.0)
    async def images_rendered(
        self, trace: vf.Trace[TextifyData, TextifyState]
    ) -> float:
        branches = trace.branches
        if not branches:
            return 0.0
        parts = [
            part
            for message in branches[-1].messages
            if isinstance(message.content, list)
            for part in message.content
        ]
        rendered = [
            part
            for part in parts
            if part.type == "text" and part.text.startswith("```image[ascii]")
        ]
        return float(len(rendered) == 2 and all(p.type != "image_url" for p in parts))


class TextifyTaskset(vf.Taskset[TextifyTask, TextifyConfig]):
    def load(self) -> list[TextifyTask]:
        return [
            TextifyTask(
                TextifyData(
                    idx=0,
                    prompt=[
                        vf.UserMessage(
                            content=[
                                vf.TextContentPart(text="Reply with ready."),
                                vf.ImageUrlContentPart(
                                    image_url=vf.ImageUrlSource(url=EXPECTED_URL)
                                ),
                            ]
                        )
                    ],
                    system_prompt="Follow each instruction briefly.",
                ),
                self.config.task,
            )
        ]


__all__ = ["TextifyTaskset"]


if __name__ == "__main__":
    TextifyUser.run()
