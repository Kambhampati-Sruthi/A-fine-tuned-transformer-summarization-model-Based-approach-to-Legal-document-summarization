"""Central configuration (12-factor: everything overridable via env / .env)."""
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- storage / security
    upload_dir: Path = ROOT / "data" / "uploads"
    index_dir: Path = ROOT / "data" / "indexes"
    output_dir: Path = ROOT / "outputs"
    database_url: str = f"sqlite:///{ROOT / 'app.db'}"
    max_upload_mb: int = 20
    max_pages: int = 400
    app_api_key: str = ""            # if set, every /api request needs X-API-Key
    cors_origins: str = "http://localhost:5173"
    retention_days: int = 7          # purge uploaded files/results older than this

    # --- models (all optional; the system degrades gracefully without them)
    summarizer_mode: str = "extractive"   # extractive | abstractive | hybrid
    model_name: str = "allenai/led-base-16384"
    model_path: str = ""                  # path to a fine-tuned checkpoint (overrides model_name)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = ""              # e.g. a cross-encoder; empty = disabled
    nli_model: str = ""                   # e.g. an MNLI cross-encoder; empty = lexical verifier

    # --- LLM adapter (OpenAI-compatible chat-completions endpoint) ---
    llm_provider: str = "none"            # none | openai_compatible
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""

    # --- pipeline limits
    chunk_max_words: int = 220
    chunk_overlap_words: int = 30
    retrieval_top_k: int = 5
    max_verify_retries: int = 2           # bounded agent loop
    ocr_enabled: bool = False


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    for d in (s.upload_dir, s.index_dir, s.output_dir):
        Path(d).mkdir(parents=True, exist_ok=True)
    return s
