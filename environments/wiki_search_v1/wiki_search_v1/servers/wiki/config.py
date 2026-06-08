import verifiers.v1 as vf

CHROMA_DB_DIR = ".chroma_db"


class WikiToolsetConfig(vf.ToolsetConfig):
    scope: vf.Scope = "env"
    startup_timeout_seconds: float = 180.0
    corpus_dataset: str = "willcb/rare-wiki-pages"
    corpus_split: str = "train"
    chroma_db_dir: str = CHROMA_DB_DIR
    embed_model: str = "text-embedding-3-small"
    embed_base_url: str = "https://api.openai.com/v1"
    embed_api_key_var: str = "OPENAI_API_KEY"
