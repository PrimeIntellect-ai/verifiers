"""Textify E2E fixture: malformed user image after a committed model turn."""

import verifiers.v1 as vf


class TextifyErrorUser(vf.User[vf.UserConfig, vf.State]):
    async def respond(self, message: str) -> vf.Messages:
        return [
            vf.UserMessage(
                content=[
                    vf.ImageUrlContentPart(
                        image_url=vf.ImageUrlSource(
                            url="data:image/png;base64,not-valid!"
                        )
                    )
                ]
            )
        ]


class TextifyErrorData(vf.TaskData):
    pass


class TextifyErrorTaskConfig(vf.TaskConfig):
    user: vf.UserConfig = vf.UserConfig()


class TextifyErrorConfig(vf.TasksetConfig):
    task: TextifyErrorTaskConfig = TextifyErrorTaskConfig()


class TextifyErrorTask(vf.Task[TextifyErrorData, vf.State, TextifyErrorTaskConfig]):
    user = TextifyErrorUser


class TextifyErrorTaskset(vf.Taskset[TextifyErrorTask, TextifyErrorConfig]):
    def load(self) -> list[TextifyErrorTask]:
        return [
            TextifyErrorTask(
                TextifyErrorData(
                    idx=0,
                    prompt="Reply with ready.",
                    system_prompt="Reply briefly.",
                ),
                self.config.task,
            )
        ]


__all__ = ["TextifyErrorTaskset"]


if __name__ == "__main__":
    TextifyErrorUser.run()
