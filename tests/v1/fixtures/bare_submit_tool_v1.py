import verifiers.v1 as vf

PHRASE = "hello world"
ECHO_TOKEN = "ok-7f3"


class BareSubmitToolset(vf.Toolset[vf.ToolsetConfig]):
    TOOL_PREFIX = None

    @vf.tool
    def submit__now(self) -> str:
        """Return a stamped value through a bare tool named exactly `submit__now`."""
        return f"{PHRASE} [{ECHO_TOKEN}]"


if __name__ == "__main__":
    BareSubmitToolset.run()
