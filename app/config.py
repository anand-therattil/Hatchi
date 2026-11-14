"""Application settings, loaded from environment variables / a .env file."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- backends ---------------------------------------------------------
    storage_backend: Literal["services", "memory"] = "services"
    vector_backend: Literal["http", "persistent", "ephemeral"] = "http"
    llm_backend: Literal["vllm", "ollama", "mock"] = "vllm"
    embedding_backend: Literal["granite", "fake"] = "granite"

    # --- infrastructure ---------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    mongo_url: str = "mongodb://localhost:27017"
    mongo_db: str = "hatchi"
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    chroma_path: str = "./.chroma"
    chroma_collection_prefix: str = "hatchi"

    # --- models -----------------------------------------------------------
    # vLLM: `vllm serve <model> --port 8002` (OpenAI-compatible). VLLM_MODEL must match the served model name.
    vllm_base_url: str = "http://localhost:8002"
    vllm_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    vllm_api_key: str = ""
    # Ollama stays available as a fallback provider (LLM_BACKEND=ollama).
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    llm_timeout_seconds: float = 120.0
    llm_temperature: float = 0.4
    llm_max_tokens: int = 512
    embedding_model: str = "ibm-granite/granite-embedding-97m-multilingual-r2"
    embedding_query_prefix: str = ""
    embedding_doc_prefix: str = ""

    # --- conversation window / lifecycle ------------------------------------
    history_max_messages: int = 12      # recent messages kept verbatim per session
    history_fold_count: int = 6         # oldest messages folded into the summary on overflow
    session_idle_seconds: int = 1800    # idle sessions are finalised by the sweeper
    session_ttl_seconds: int = 86400    # Redis key TTL (backstop only)
    sweep_interval_seconds: int = 60
    sweeper_enabled: bool = True
    ws_disconnect_grace_seconds: float = 60.0

    # --- retrieval --------------------------------------------------------
    # Cosine scores from real embedding models are compressed: with Granite-97M a *relevant* memory scored ~0.90
    # and an *unrelated* one ~0.79. So absolute cut-offs must be high, and a relative margin (how far below the
    # best hit a result may fall) does most of the work. PROVISIONAL - run `python -m scripts.calibrate_similarity`.
    memory_top_k: int = 5
    memory_recall_max: int = 8
    memory_min_similarity: float = 0.85
    semantic_margin: float = 0.10       # semantic-only hits must be within this of the best hit
    category_boost: float = 0.5
    summary_top_k: int = 2
    summary_min_similarity: float = 0.85
    summary_recall_min_similarity: float = 0.75   # lower bar when the user explicitly asks about the past
    fact_dedupe_similarity: float = 0.9

    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
