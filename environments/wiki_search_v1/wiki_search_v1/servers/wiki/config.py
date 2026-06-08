import verifiers.v1 as vf


class WikiToolsetConfig(vf.ToolsetConfig):
    scope: vf.Scope = "env"
    startup_timeout_seconds: float = 60.0
    corpus_dataset: str = "willcb/rare-wiki-pages"
    corpus_split: str = "train"
