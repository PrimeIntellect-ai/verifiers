import verifiers.v1 as vf


class CalcToolset(vf.Toolset[vf.ToolsetConfig]):
    TOOL_PREFIX = "calc"  # the model sees `calc_add`

    @vf.tool
    def add(self, a: int, b: int) -> str:
        """Add two integers and return their sum."""
        return str(a + b)


if __name__ == "__main__":
    CalcToolset.run()
